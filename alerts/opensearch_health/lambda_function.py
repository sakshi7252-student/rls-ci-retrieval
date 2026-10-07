"""
Alert Lambda: OpenSearch cluster health watchdog.

"OpenSearch Health Alerts: Implement immediate alerting for OpenSearch write
blocks, circuit breaker triggers, and non-green cluster health warnings."

Polls CloudWatch metrics directly (no native CloudWatch Alarms involved --
all detection logic lives here so it's visible/debuggable like the other
alert lambdas) for the OpenSearch domain named by DOMAIN_NAME. No SQS/queue
checks here -- those live in alerts/queue_capacity (separate resource).

Checks:
  1. CLUSTER_STATUS_RED: ClusterStatus.red == 1 in the last period -- fires
     immediately, no sustained requirement (data unavailable).
  2. CLUSTER_STATUS_YELLOW_SUSTAINED: ClusterStatus.yellow == 1 for every
     period in the last YELLOW_SUSTAINED_MINUTES minutes (yellow is common/
     transient during shard rebalancing, so a single blip is not paged).
  3. INDEX_WRITES_BLOCKED: ClusterIndexWritesBlocked == 1 in the last period
     -- fires immediately (cluster-wide write stop).
  4. JVM_MEMORY_PRESSURE: JVMMemoryPressure/MasterJVMMemoryPressure stayed
     above JVM_MEMORY_PRESSURE_THRESHOLD for every period in the last
     JVM_SUSTAINED_MINUTES minutes. AWS/ES has no literal "circuit breaker
     tripped" metric -- this is the documented proxy, since the real-memory
     circuit breaker trips at ~95% JVM heap by default.
  5. KNN_CACHE_CAPACITY_REACHED: KNNCacheCapacityReached == 1 -- per-node KNN
     graph cache hit its capacity limit (vector search starts degrading).
  6. KNN_CIRCUIT_BREAKER_TRIGGERED: KNNCircuitBreakerTriggered == 1 -- the
     cluster-level KNN circuit breaker has actually tripped (distinct from
     and more severe than #5).
  7. STORAGE_WARNING/CRITICAL/EMERGENCY: FreeStorageSpace (per-node, MB)
     below STORAGE_*_GIB thresholds -- early warning before the cluster
     ever reaches ClusterIndexWritesBlocked.
  8. NODE_COUNT_DECREASED: Nodes < EXPECTED_NODE_COUNT (only checked if that
     env var is set).
  9. SHARDS_UNASSIGNED: Shards.unassigned > 0 in the last period.
  10. CPU_PRESSURE: CPUUtilization/MasterCPUUtilization stayed above
      CPU_THRESHOLD for every period in the last CPU_SUSTAINED_MINUTES.
  11. NATIVE_MEMORY_PRESSURE: NativeMemoryPressure stayed above
      NATIVE_MEMORY_PRESSURE_THRESHOLD for NATIVE_MEMORY_SUSTAINED_MINUTES --
      JVM pressure alone can look fine while native/off-heap (KNN graphs)
      memory is maxed out.
  12. WRITE_REJECTIONS_DETECTED: Coordinating/Primary/ReplicaWriteRejected
      sum > 0 in the last period -- indexing pressure rejecting writes.
  13. THREADPOOL_PRESSURE: ThreadpoolWriteQueue/ThreadpoolSearchQueue above
      their threshold, or any ThreadpoolWriteRejected/ThreadpoolSearchRejected.
  14. SNAPSHOT_FAILURE: AutomatedSnapshotFailure == 1 (no successful
      automated snapshot in the last 36h per AWS's own definition).
  15. CRITICAL_INDEX_MISSING / INDEX_HEALTH_RED / INDEX_HEALTH_YELLOW /
      INDEX_WRITE_BLOCKED / INDEX_READ_ONLY / INDEX_READ_ONLY_ALLOW_DELETE /
      INDEX_METADATA_BLOCKED: direct OpenSearch REST API check (GET
      _cluster/health/<index> + GET <index>/_settings) for each index in
      CRITICAL_INDICES -- catches per-index problems a cluster-wide
      CloudWatch metric can miss entirely. Each index.blocks.* flag maps to
      its own alert type (see INDEX_BLOCK_ALERT_TYPES) instead of being
      collapsed into one generic "write blocked" alert, e.g. a metadata
      block alone would otherwise be misreported as a write block.
  16. SHARDS_INITIALIZING/SHARDS_RELOCATING: Shards.initializing/relocating
      > 0 -- shard-movement signal that complements SHARDS_UNASSIGNED.
  17. MASTER_UNREACHABLE: MasterReachableFromNode == 0 in the last period.
  18. OLD_GEN_JVM_PRESSURE: OldGenJVMMemoryPressure sustained above
      OLD_GEN_JVM_THRESHOLD -- old-gen fills up before full heap pressure
      shows, so this is an earlier warning than JVM_MEMORY_PRESSURE.
  19. EBS_IO_PRESSURE: VolumeStalledIOCheck == 1 (immediate), or
      ReadLatency/WriteLatency above EBS_LATENCY_THRESHOLD_MS, or
      ReadThroughput/WriteThroughput/ReadIOPS/WriteIOPS below/above their
      thresholds, or BurstBalance below BURST_BALANCE_THRESHOLD (skipped
      for gp3/large-gp2 volumes where AWS always reports 0 -- see
      BURST_BALANCE_THRESHOLD=0 default, which disables this sub-check).
  20. KNN_GRAPH_ERRORS: KNNGraphIndexErrors/KNNEvictionCount sum > 0 --
      informational signal, not necessarily an outage.
  21. SEARCH_LATENCY_HIGH: SearchLatency sustained above
      SEARCH_LATENCY_THRESHOLD_MS.
  22. HTTP_5XX_DETECTED / HTTP_4XX_SPIKE / TLS_NEGOTIATION_ERROR /
      INVALID_HOST_HEADER: 5xx sum > 0; 4xx sum above HTTP_4XX_SPIKE_THRESHOLD;
      TLSNegotiationError/InvalidHostHeaderRequests sum > 0.
  23. MONITORING_DATA_MISSING: fired whenever a sustained-window metric
      (yellow/JVM/old-gen/CPU/native-memory/search-latency) returns fewer
      datapoints than the window expects -- missing CloudWatch data must
      not silently read as "healthy".

NOT covered here: SNS alert deduplication. Every breach found in a run is
sent immediately -- there's no cross-invocation state store (S3/DynamoDB)
to suppress repeat alerts for an ongoing issue. Needs an explicit decision
on where that state lives before adding it.
"""

import os
import time
import logging
import boto3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.sns_alert import send_sns_alert

logger = logging.getLogger()
logger.setLevel(logging.INFO)

RESOURCE = "opensearch-health"

DOMAIN_NAME = os.environ.get("DOMAIN_NAME")
YELLOW_SUSTAINED_MINUTES = int(os.environ.get("YELLOW_SUSTAINED_MINUTES", "3"))
JVM_MEMORY_PRESSURE_THRESHOLD = float(os.environ.get("JVM_MEMORY_PRESSURE_THRESHOLD", "92"))
JVM_SUSTAINED_MINUTES = int(os.environ.get("JVM_SUSTAINED_MINUTES", "3"))

STORAGE_WARNING_GIB = float(os.environ.get("STORAGE_WARNING_GIB", "50"))
STORAGE_CRITICAL_GIB = float(os.environ.get("STORAGE_CRITICAL_GIB", "20"))
STORAGE_EMERGENCY_GIB = float(os.environ.get("STORAGE_EMERGENCY_GIB", "5"))
EXPECTED_NODE_COUNT = os.environ.get("EXPECTED_NODE_COUNT")  # unset = skip check
CPU_THRESHOLD = float(os.environ.get("CPU_THRESHOLD", "90"))
CPU_SUSTAINED_MINUTES = int(os.environ.get("CPU_SUSTAINED_MINUTES", "5"))
NATIVE_MEMORY_PRESSURE_THRESHOLD = float(os.environ.get("NATIVE_MEMORY_PRESSURE_THRESHOLD", "90"))
NATIVE_MEMORY_SUSTAINED_MINUTES = int(os.environ.get("NATIVE_MEMORY_SUSTAINED_MINUTES", "3"))
THREADPOOL_WRITE_QUEUE_THRESHOLD = int(os.environ.get("THREADPOOL_WRITE_QUEUE_THRESHOLD", "100"))
THREADPOOL_SEARCH_QUEUE_THRESHOLD = int(os.environ.get("THREADPOOL_SEARCH_QUEUE_THRESHOLD", "100"))

OLD_GEN_JVM_THRESHOLD = float(os.environ.get("OLD_GEN_JVM_THRESHOLD", "80"))
OLD_GEN_JVM_SUSTAINED_MINUTES = int(os.environ.get("OLD_GEN_JVM_SUSTAINED_MINUTES", "3"))
EBS_LATENCY_THRESHOLD_MS = float(os.environ.get("EBS_LATENCY_THRESHOLD_MS", "50"))
EBS_IOPS_THRESHOLD = float(os.environ.get("EBS_IOPS_THRESHOLD", "0"))  # 0 = disabled by default, set per-volume baseline
BURST_BALANCE_THRESHOLD = float(os.environ.get("BURST_BALANCE_THRESHOLD", "0"))  # 0 = disabled; gp3/large gp2 report 0 always
SEARCH_LATENCY_THRESHOLD_MS = float(os.environ.get("SEARCH_LATENCY_THRESHOLD_MS", "2000"))
SEARCH_LATENCY_SUSTAINED_MINUTES = int(os.environ.get("SEARCH_LATENCY_SUSTAINED_MINUTES", "3"))
HTTP_4XX_SPIKE_THRESHOLD = int(os.environ.get("HTTP_4XX_SPIKE_THRESHOLD", "50"))

OPENSEARCH_ENDPOINT = os.environ.get("OPENSEARCH_ENDPOINT")
OPEN_SEARCH_REGION  = os.environ.get("OPEN_SEARCH_REGION", os.environ.get("AWS_REGION", "eu-west-1"))
CRITICAL_INDICES = [s.strip() for s in os.environ.get(
    "CRITICAL_INDICES", "document-chunks,semantic-objects-v3,ci-objects,qc-document-terms"
).split(",") if s.strip()]
INDEX_BLOCK_NAMES = ("write", "read_only", "read_only_allow_delete", "metadata")
INDEX_BLOCK_ALERT_TYPES = {
    "write": "INDEX_WRITE_BLOCKED",
    "read_only": "INDEX_READ_ONLY",
    "read_only_allow_delete": "INDEX_READ_ONLY_ALLOW_DELETE",
    "metadata": "INDEX_METADATA_BLOCKED",
}

_os_client = None


def _get_opensearch_client():
    """Lazy singleton -- only constructed if CRITICAL_INDICES/OPENSEARCH_ENDPOINT are configured."""
    global _os_client
    if _os_client is None:
        from opensearchpy import OpenSearch, RequestsHttpConnection
        from requests_aws4auth import AWS4Auth
        frozen = boto3.Session().get_credentials().get_frozen_credentials()
        awsauth = AWS4Auth(frozen.access_key, frozen.secret_key, OPEN_SEARCH_REGION, "es", session_token=frozen.token)
        _os_client = OpenSearch(
            hosts=[{"host": OPENSEARCH_ENDPOINT, "port": 443}],
            http_auth=awsauth, use_ssl=True, verify_certs=True,
            connection_class=RequestsHttpConnection, timeout=15,
        )
    return _os_client

cloudwatch = boto3.client("cloudwatch")
sts = boto3.client("sts")


def _get_datapoints(account_id: str, metric_name: str, statistic: str, period_seconds: int, lookback_seconds: int):
    """Returns datapoint values ordered oldest-to-newest for the lookback window."""
    now = int(time.time())
    response = cloudwatch.get_metric_statistics(
        Namespace="AWS/ES",
        MetricName=metric_name,
        Dimensions=[
            {"Name": "DomainName", "Value": DOMAIN_NAME},
            {"Name": "ClientId", "Value": account_id},
        ],
        StartTime=now - lookback_seconds,
        EndTime=now,
        Period=period_seconds,
        Statistics=[statistic],
    )
    datapoints = sorted(response.get("Datapoints", []), key=lambda d: d["Timestamp"])
    return [d[statistic] for d in datapoints]


def _sustained_breach(values: list[float], threshold: float, min_periods: int) -> bool:
    """All expected periods present and every one of them breaches threshold."""
    if len(values) < min_periods:
        return False
    return all(v > threshold for v in values[-min_periods:])


def _latest_breach(values: list[float], threshold: float) -> bool:
    return bool(values) and values[-1] > threshold


def _check_data_presence(check_fn, metric_name: str, values: list, expected_periods: int) -> None:
    """Missing CloudWatch datapoints must not silently read as "healthy"."""
    if len(values) < expected_periods:
        check_fn("MONITORING_DATA_MISSING", {
            "metric": metric_name, "expected_datapoints": expected_periods,
            "received_datapoints": len(values), "severity": "P2",
        })


def lambda_handler(event, context):
    start_time = time.monotonic()

    if not DOMAIN_NAME:
        logger.error("DOMAIN_NAME env var is not set")
        return {"statusCode": 500, "resource": RESOURCE, "error": "DOMAIN_NAME not set"}

    try:
        account_id = sts.get_caller_identity()["Account"]
        totals: dict[str, int] = {}

        def check(alert_type: str, row: dict):
            row["schema"] = DOMAIN_NAME
            send_sns_alert([row], alert_type, RESOURCE)
            totals[alert_type] = totals.get(alert_type, 0) + 1
            logger.info(f"{alert_type} fired for {DOMAIN_NAME}: {row}")

        # 1 + 2. Cluster status.
        red_start = time.monotonic()
        red = _get_datapoints(account_id, "ClusterStatus.red", "Maximum", 60, 60)
        logger.info(f"ClusterStatus.red check completed in {time.monotonic() - red_start:.2f}s")
        if _latest_breach(red, 0):
            check("CLUSTER_STATUS_RED", {"metric": "ClusterStatus.red", "value": red[-1]})

        yellow_start = time.monotonic()
        yellow = _get_datapoints(account_id, "ClusterStatus.yellow", "Maximum", 60, YELLOW_SUSTAINED_MINUTES * 60)
        logger.info(f"ClusterStatus.yellow check completed in {time.monotonic() - yellow_start:.2f}s ({len(yellow)} datapoints)")
        _check_data_presence(check, "ClusterStatus.yellow", yellow, YELLOW_SUSTAINED_MINUTES)
        if _sustained_breach(yellow, 0, YELLOW_SUSTAINED_MINUTES):
            check("CLUSTER_STATUS_YELLOW_SUSTAINED", {
                "metric": "ClusterStatus.yellow",
                "sustained_minutes": YELLOW_SUSTAINED_MINUTES,
                "recent_values": yellow[-YELLOW_SUSTAINED_MINUTES:],
            })

        # 3. Index writes blocked.
        blocked_start = time.monotonic()
        blocked = _get_datapoints(account_id, "ClusterIndexWritesBlocked", "Maximum", 60, 60)
        logger.info(f"ClusterIndexWritesBlocked check completed in {time.monotonic() - blocked_start:.2f}s")
        if _latest_breach(blocked, 0):
            check("INDEX_WRITES_BLOCKED", {"metric": "ClusterIndexWritesBlocked", "value": blocked[-1]})

        # 4. JVM memory pressure (circuit-breaker proxy).
        for metric_name in ("JVMMemoryPressure", "MasterJVMMemoryPressure"):
            jvm_start = time.monotonic()
            values = _get_datapoints(account_id, metric_name, "Maximum", 60, JVM_SUSTAINED_MINUTES * 60)
            logger.info(f"{metric_name} check completed in {time.monotonic() - jvm_start:.2f}s ({len(values)} datapoints)")
            _check_data_presence(check, metric_name, values, JVM_SUSTAINED_MINUTES)
            if _sustained_breach(values, JVM_MEMORY_PRESSURE_THRESHOLD, JVM_SUSTAINED_MINUTES):
                check("JVM_MEMORY_PRESSURE", {
                    "metric": metric_name,
                    "threshold": JVM_MEMORY_PRESSURE_THRESHOLD,
                    "sustained_minutes": JVM_SUSTAINED_MINUTES,
                    "recent_values": values[-JVM_SUSTAINED_MINUTES:],
                })

        # 18. Old-gen JVM pressure -- fills up before full heap pressure shows.
        oldgen_start = time.monotonic()
        oldgen_values = _get_datapoints(account_id, "OldGenJVMMemoryPressure", "Maximum", 60, OLD_GEN_JVM_SUSTAINED_MINUTES * 60)
        logger.info(f"OldGenJVMMemoryPressure check completed in {time.monotonic() - oldgen_start:.2f}s ({len(oldgen_values)} datapoints)")
        _check_data_presence(check, "OldGenJVMMemoryPressure", oldgen_values, OLD_GEN_JVM_SUSTAINED_MINUTES)
        if _sustained_breach(oldgen_values, OLD_GEN_JVM_THRESHOLD, OLD_GEN_JVM_SUSTAINED_MINUTES):
            check("OLD_GEN_JVM_PRESSURE", {
                "metric": "OldGenJVMMemoryPressure", "threshold": OLD_GEN_JVM_THRESHOLD,
                "sustained_minutes": OLD_GEN_JVM_SUSTAINED_MINUTES,
                "recent_values": oldgen_values[-OLD_GEN_JVM_SUSTAINED_MINUTES:], "severity": "P1",
            })

        # 5 + 6. KNN: per-node cache capacity vs. the actual cluster-level circuit breaker.
        knn_start = time.monotonic()
        knn_cache = _get_datapoints(account_id, "KNNCacheCapacityReached", "Maximum", 60, 60)
        logger.info(f"KNNCacheCapacityReached check completed in {time.monotonic() - knn_start:.2f}s")
        if _latest_breach(knn_cache, 0):
            check("KNN_CACHE_CAPACITY_REACHED", {"metric": "KNNCacheCapacityReached", "value": knn_cache[-1], "severity": "P2"})

        knn_cb_start = time.monotonic()
        knn_cb = _get_datapoints(account_id, "KNNCircuitBreakerTriggered", "Maximum", 60, 60)
        logger.info(f"KNNCircuitBreakerTriggered check completed in {time.monotonic() - knn_cb_start:.2f}s")
        if _latest_breach(knn_cb, 0):
            check("KNN_CIRCUIT_BREAKER_TRIGGERED", {"metric": "KNNCircuitBreakerTriggered", "value": knn_cb[-1], "severity": "P0"})

        # 7. Storage: tiered early warning before ClusterIndexWritesBlocked ever fires.
        storage_start = time.monotonic()
        storage_mb = _get_datapoints(account_id, "FreeStorageSpace", "Minimum", 60, 60)
        logger.info(f"FreeStorageSpace check completed in {time.monotonic() - storage_start:.2f}s")
        if storage_mb:
            free_gib = storage_mb[-1] / 1024.0
            if free_gib <= STORAGE_EMERGENCY_GIB:
                check("STORAGE_EMERGENCY", {"metric": "FreeStorageSpace", "free_gib": round(free_gib, 2), "threshold_gib": STORAGE_EMERGENCY_GIB, "severity": "P0"})
            elif free_gib <= STORAGE_CRITICAL_GIB:
                check("STORAGE_CRITICAL", {"metric": "FreeStorageSpace", "free_gib": round(free_gib, 2), "threshold_gib": STORAGE_CRITICAL_GIB, "severity": "P1"})
            elif free_gib <= STORAGE_WARNING_GIB:
                check("STORAGE_WARNING", {"metric": "FreeStorageSpace", "free_gib": round(free_gib, 2), "threshold_gib": STORAGE_WARNING_GIB, "severity": "P2"})

        # 8. Node count.
        if EXPECTED_NODE_COUNT:
            nodes_start = time.monotonic()
            nodes = _get_datapoints(account_id, "Nodes", "Minimum", 60, 60)
            logger.info(f"Nodes check completed in {time.monotonic() - nodes_start:.2f}s")
            expected = int(EXPECTED_NODE_COUNT)
            if nodes and nodes[-1] < expected:
                check("NODE_COUNT_DECREASED", {"metric": "Nodes", "expected": expected, "actual": nodes[-1], "severity": "P0"})

        # 9. Unassigned shards.
        shards_start = time.monotonic()
        unassigned = _get_datapoints(account_id, "Shards.unassigned", "Maximum", 60, 60)
        logger.info(f"Shards.unassigned check completed in {time.monotonic() - shards_start:.2f}s")
        if _latest_breach(unassigned, 0):
            check("SHARDS_UNASSIGNED", {"metric": "Shards.unassigned", "value": unassigned[-1], "severity": "P1"})

        # 16. Initializing/relocating shards.
        for metric_name, alert_type in (("Shards.initializing", "SHARDS_INITIALIZING"), ("Shards.relocating", "SHARDS_RELOCATING")):
            values = _get_datapoints(account_id, metric_name, "Maximum", 60, 60)
            if _latest_breach(values, 0):
                check(alert_type, {"metric": metric_name, "value": values[-1], "severity": "P2"})

        # 17. Master reachability.
        master_start = time.monotonic()
        master_reachable = _get_datapoints(account_id, "MasterReachableFromNode", "Minimum", 60, 60)
        logger.info(f"MasterReachableFromNode check completed in {time.monotonic() - master_start:.2f}s")
        if master_reachable and master_reachable[-1] == 0:
            check("MASTER_UNREACHABLE", {"metric": "MasterReachableFromNode", "value": master_reachable[-1], "severity": "P0"})

        # 10. CPU pressure.
        for metric_name in ("CPUUtilization", "MasterCPUUtilization"):
            cpu_start = time.monotonic()
            values = _get_datapoints(account_id, metric_name, "Average", 60, CPU_SUSTAINED_MINUTES * 60)
            logger.info(f"{metric_name} check completed in {time.monotonic() - cpu_start:.2f}s ({len(values)} datapoints)")
            _check_data_presence(check, metric_name, values, CPU_SUSTAINED_MINUTES)
            if _sustained_breach(values, CPU_THRESHOLD, CPU_SUSTAINED_MINUTES):
                check("CPU_PRESSURE", {
                    "metric": metric_name, "threshold": CPU_THRESHOLD,
                    "sustained_minutes": CPU_SUSTAINED_MINUTES,
                    "recent_values": values[-CPU_SUSTAINED_MINUTES:], "severity": "P1",
                })

        # 11. Native (off-heap) memory pressure -- JVM alone can look fine while this is maxed.
        native_start = time.monotonic()
        native_values = _get_datapoints(account_id, "NativeMemoryPressure", "Maximum", 60, NATIVE_MEMORY_SUSTAINED_MINUTES * 60)
        logger.info(f"NativeMemoryPressure check completed in {time.monotonic() - native_start:.2f}s ({len(native_values)} datapoints)")
        _check_data_presence(check, "NativeMemoryPressure", native_values, NATIVE_MEMORY_SUSTAINED_MINUTES)
        if _sustained_breach(native_values, NATIVE_MEMORY_PRESSURE_THRESHOLD, NATIVE_MEMORY_SUSTAINED_MINUTES):
            check("NATIVE_MEMORY_PRESSURE", {
                "metric": "NativeMemoryPressure", "threshold": NATIVE_MEMORY_PRESSURE_THRESHOLD,
                "sustained_minutes": NATIVE_MEMORY_SUSTAINED_MINUTES,
                "recent_values": native_values[-NATIVE_MEMORY_SUSTAINED_MINUTES:], "severity": "P1",
            })

        # 12. Write rejections (indexing pressure).
        for metric_name in ("CoordinatingWriteRejected", "PrimaryWriteRejected", "ReplicaWriteRejected"):
            wr_start = time.monotonic()
            values = _get_datapoints(account_id, metric_name, "Sum", 60, 60)
            logger.info(f"{metric_name} check completed in {time.monotonic() - wr_start:.2f}s")
            if _latest_breach(values, 0):
                check("WRITE_REJECTIONS_DETECTED", {"metric": metric_name, "value": values[-1], "severity": "P1"})

        # 13. Thread pool queue/rejection pressure.
        tp_queue_start = time.monotonic()
        write_queue = _get_datapoints(account_id, "ThreadpoolWriteQueue", "Maximum", 60, 60)
        search_queue = _get_datapoints(account_id, "ThreadpoolSearchQueue", "Maximum", 60, 60)
        logger.info(f"Threadpool queue check completed in {time.monotonic() - tp_queue_start:.2f}s")
        if _latest_breach(write_queue, THREADPOOL_WRITE_QUEUE_THRESHOLD):
            check("THREADPOOL_PRESSURE", {"metric": "ThreadpoolWriteQueue", "value": write_queue[-1], "threshold": THREADPOOL_WRITE_QUEUE_THRESHOLD, "severity": "P2"})
        if _latest_breach(search_queue, THREADPOOL_SEARCH_QUEUE_THRESHOLD):
            check("THREADPOOL_PRESSURE", {"metric": "ThreadpoolSearchQueue", "value": search_queue[-1], "threshold": THREADPOOL_SEARCH_QUEUE_THRESHOLD, "severity": "P2"})
        for metric_name in ("ThreadpoolWriteRejected", "ThreadpoolSearchRejected"):
            values = _get_datapoints(account_id, metric_name, "Sum", 60, 60)
            if _latest_breach(values, 0):
                check("THREADPOOL_PRESSURE", {"metric": metric_name, "value": values[-1], "severity": "P1"})

        # 14. Automated snapshot failure.
        snap_start = time.monotonic()
        snapshot_failure = _get_datapoints(account_id, "AutomatedSnapshotFailure", "Maximum", 60, 60)
        logger.info(f"AutomatedSnapshotFailure check completed in {time.monotonic() - snap_start:.2f}s")
        if _latest_breach(snapshot_failure, 0):
            check("SNAPSHOT_FAILURE", {"metric": "AutomatedSnapshotFailure", "value": snapshot_failure[-1], "severity": "P1"})

        # 19. EBS I/O pressure.
        ebs_start = time.monotonic()
        stalled_io = _get_datapoints(account_id, "VolumeStalledIOCheck", "Maximum", 60, 60)
        if _latest_breach(stalled_io, 0):
            check("EBS_IO_PRESSURE", {"metric": "VolumeStalledIOCheck", "value": stalled_io[-1], "severity": "P0"})
        for metric_name in ("ReadLatency", "WriteLatency"):
            values = _get_datapoints(account_id, metric_name, "Average", 60, 60)
            if _latest_breach(values, EBS_LATENCY_THRESHOLD_MS):
                check("EBS_IO_PRESSURE", {"metric": metric_name, "value": values[-1], "threshold_ms": EBS_LATENCY_THRESHOLD_MS, "severity": "P1"})
        for metric_name in ("ReadThroughput", "WriteThroughput"):
            values = _get_datapoints(account_id, metric_name, "Average", 60, 60)
            logger.info(f"{metric_name}: {values[-1] if values else 'n/a'}")  # trend-only, no alert threshold yet
        if EBS_IOPS_THRESHOLD:
            for metric_name in ("ReadIOPS", "WriteIOPS"):
                values = _get_datapoints(account_id, metric_name, "Average", 60, 60)
                if _latest_breach(values, EBS_IOPS_THRESHOLD):
                    check("EBS_IO_PRESSURE", {"metric": metric_name, "value": values[-1], "threshold": EBS_IOPS_THRESHOLD, "severity": "P2"})
        if BURST_BALANCE_THRESHOLD:
            burst = _get_datapoints(account_id, "BurstBalance", "Minimum", 60, 60)
            if burst and burst[-1] < BURST_BALANCE_THRESHOLD:
                check("EBS_IO_PRESSURE", {"metric": "BurstBalance", "value": burst[-1], "threshold": BURST_BALANCE_THRESHOLD, "severity": "P1"})
        logger.info(f"EBS I/O checks completed in {time.monotonic() - ebs_start:.2f}s")

        # 20. KNN graph errors/evictions -- informational, not necessarily an outage.
        for metric_name in ("KNNGraphIndexErrors", "KNNEvictionCount"):
            values = _get_datapoints(account_id, metric_name, "Sum", 60, 60)
            if _latest_breach(values, 0):
                check("KNN_GRAPH_ERRORS", {"metric": metric_name, "value": values[-1], "severity": "P2"})

        # 21. Search latency.
        search_lat_start = time.monotonic()
        search_latency = _get_datapoints(account_id, "SearchLatency", "Average", 60, SEARCH_LATENCY_SUSTAINED_MINUTES * 60)
        logger.info(f"SearchLatency check completed in {time.monotonic() - search_lat_start:.2f}s ({len(search_latency)} datapoints)")
        _check_data_presence(check, "SearchLatency", search_latency, SEARCH_LATENCY_SUSTAINED_MINUTES)
        if _sustained_breach(search_latency, SEARCH_LATENCY_THRESHOLD_MS, SEARCH_LATENCY_SUSTAINED_MINUTES):
            check("SEARCH_LATENCY_HIGH", {
                "metric": "SearchLatency", "threshold_ms": SEARCH_LATENCY_THRESHOLD_MS,
                "sustained_minutes": SEARCH_LATENCY_SUSTAINED_MINUTES,
                "recent_values": search_latency[-SEARCH_LATENCY_SUSTAINED_MINUTES:], "severity": "P2",
            })

        # 22. HTTP response codes + TLS/host-header errors.
        http_5xx = _get_datapoints(account_id, "5xx", "Sum", 60, 60)
        if _latest_breach(http_5xx, 0):
            check("HTTP_5XX_DETECTED", {"metric": "5xx", "value": http_5xx[-1], "severity": "P1"})
        http_4xx = _get_datapoints(account_id, "4xx", "Sum", 60, 60)
        if _latest_breach(http_4xx, HTTP_4XX_SPIKE_THRESHOLD):
            check("HTTP_4XX_SPIKE", {"metric": "4xx", "value": http_4xx[-1], "threshold": HTTP_4XX_SPIKE_THRESHOLD, "severity": "P2"})
        for metric_name, alert_type in (("TLSNegotiationError", "TLS_NEGOTIATION_ERROR"), ("InvalidHostHeaderRequests", "INVALID_HOST_HEADER")):
            values = _get_datapoints(account_id, metric_name, "Sum", 60, 60)
            if _latest_breach(values, 0):
                check(alert_type, {"metric": metric_name, "value": values[-1], "severity": "P2"})

        # 15. Critical index existence/health/write-blocks (REST API, not CloudWatch).
        if CRITICAL_INDICES and OPENSEARCH_ENDPOINT:
            idx_start = time.monotonic()
            try:
                client = _get_opensearch_client()
                existing_names = {row["index"] for row in client.cat.indices(format="json")}
            except Exception as exc:
                logger.error(f"Critical index check failed to query OpenSearch: {exc}")
                existing_names = None
            if existing_names is not None:
                for index in CRITICAL_INDICES:
                    if index not in existing_names:
                        check("CRITICAL_INDEX_MISSING", {"index": index, "severity": "P0"})
                        continue
                    try:
                        health = client.cluster.health(index=index)
                        if health.get("status") == "red":
                            check("INDEX_HEALTH_RED", {"index": index, "severity": "P0"})
                        elif health.get("status") == "yellow":
                            check("INDEX_HEALTH_YELLOW", {"index": index, "severity": "P2"})
                        settings = client.indices.get_settings(index=index)
                        idx_settings = next(iter(settings.values()), {}).get("settings", {}).get("index", {})
                        blocks = idx_settings.get("blocks", {})
                        for block_name in INDEX_BLOCK_NAMES:
                            if str(blocks.get(block_name, "false")).lower() == "true":
                                alert_type = INDEX_BLOCK_ALERT_TYPES[block_name]
                                check(alert_type, {"index": index, "block": f"index.blocks.{block_name}", "severity": "P0"})
                    except Exception as exc:
                        logger.error(f"Critical index check failed for {index}: {exc}")
            logger.info(f"Critical index checks completed in {time.monotonic() - idx_start:.2f}s")

        execution_time = time.monotonic() - start_time
        logger.info(f"{RESOURCE} scan completed in {execution_time:.2f}s, totals={totals}")

        return {
            "statusCode": 200,
            "resource": RESOURCE,
            "domain": DOMAIN_NAME,
            "execution_time_seconds": execution_time,
            **{f"{alert_type.lower()}_found": count for alert_type, count in totals.items()},
        }
    except Exception as e:
        logger.error(f"{RESOURCE} scan failed: {str(e)}")
        try:
            send_sns_alert([{"schema": DOMAIN_NAME or RESOURCE, "error": str(e), "severity": "P0"}], "WATCHDOG_EXECUTION_FAILED", RESOURCE)
        except Exception as sns_exc:
            logger.error(f"Failed to send WATCHDOG_EXECUTION_FAILED alert: {str(sns_exc)}")
        return {"statusCode": 500, "resource": RESOURCE, "error": str(e)}

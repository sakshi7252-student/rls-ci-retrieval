"""
Shared SNS alert sender for the stuck/exhausted job alert Lambdas.
One SNS message per (schema, alert_type) group, mirroring the existing
nlp_job_detail failure/stuck alert convention.
"""

import os
import json
import logging
import boto3
from datetime import datetime, timezone

logger = logging.getLogger()

SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN")
ENVIRONMENT = os.environ.get("ENVIRONMENT", "unknown")

sns = boto3.client("sns")

sent = False
def send_sns_alert(rows: list[dict], alert_type: str, resource: str) -> None:
    """
    rows: each dict must include a "schema" key plus whatever identifying
    columns the caller selected (e.g. fileId/ciId, status, attempt count, age).
    alert_type: "MISSED_STUCK" or "EXHAUSTED_RETRIES".
    resource: short resource label, e.g. "file-extraction", "ci-indexing",
    "document-indexing".
    """
    if not sent:

        sns.publish(
        TopicArn=SNS_TOPIC_ARN,
        Message="TEST FROM LAMBDA - MMIS SNS DELIVERY",
        Subject="Lambda SNS Test",
        )
        sent = True
    
    if not rows:
        return

    rows_by_schema: dict[str, list[dict]] = {}
    for row in rows:
        rows_by_schema.setdefault(row["schema"], []).append(row)

    for schema, schema_rows in rows_by_schema.items():
        message = {
            "Alert": (
                f"{resource} {alert_type} detected in {ENVIRONMENT} "
                f"in tenant {schema}. {len(schema_rows)} row(s) affected."
            ),
            "resource": resource,
            "alert_type": alert_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "schema": schema,
            "rows": schema_rows,
        }
        subject = f"[{ENVIRONMENT}] {resource} {alert_type}: {len(schema_rows)} row(s) in {schema}"

        try:
            sns.publish(
                TopicArn=SNS_TOPIC_ARN,
                Message=json.dumps(message, indent=2, default=str),
                Subject=subject[:100],  # SNS subject hard limit
            )
            logger.info(f"Alert sent for {len(schema_rows)} {alert_type} rows in schema {schema}")
        except Exception as e:
            logger.error(f"Error sending SNS alert: {str(e)}")

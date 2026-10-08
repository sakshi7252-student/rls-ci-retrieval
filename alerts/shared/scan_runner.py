"""
Shared scan runner for the stuck/exhausted job alert Lambdas.
Mirrors the existing nlp_job_detail alert lambda's robustness: per-schema
timing logs, a schema-count cap to avoid Lambda timeout on many tenants,
and a top-level try/except so an unexpected failure returns a clean 500
instead of a raw stack trace.
"""

import time
import logging

from .pg_client import get_db_connection, get_all_schemas, check_table_exists, run_schema_query, get_main_account
from .sns_alert import send_sns_alert

logger = logging.getLogger()

MAX_SCHEMAS = 100


def run_alert_scan(
    event,
    resource: str,
    required_table: str,
    queries: dict[str, str],
    require_cim_access: bool = False,
) -> dict:
    """
    queries: {"MISSED_STUCK": sql_template, "EXHAUSTED_RETRIES": sql_template, ...}
    Each sql_template must contain a `{schema}` placeholder for schema-qualification.
    require_cim_access: when True, skip a schema whose main.accounts.accessToCIM
    is not true, before any of the required_table/queries checks run.
    """
    start_time = time.monotonic()
    try:
        connection = get_db_connection()
        connection.autocommit = True

        schema_start = time.monotonic()
        schemas = get_all_schemas(connection)
        logger.info(f"Found {len(schemas)} schemas to process in {time.monotonic() - schema_start:.2f}s")

        process_all = event.get("process_all_schemas", False) if isinstance(event, dict) else False
        if len(schemas) > MAX_SCHEMAS and not process_all:
            logger.info(f"Processing only the first {MAX_SCHEMAS} schemas to avoid timeout")
            schemas = schemas[:MAX_SCHEMAS]

        totals = {alert_type: 0 for alert_type in queries}

        for schema in schemas:
            if require_cim_access:
                account = get_main_account(connection, schema)
                if not account.get("accessToCIM"):
                    logger.info(f"Schema {schema} has no CIM access, skipping")
                    continue

            table_check_start = time.monotonic()
            exists = check_table_exists(connection, schema, required_table)
            logger.info(f"Table check for {schema} completed in {time.monotonic() - table_check_start:.2f}s")

            if not exists:
                logger.info(f"Table {required_table} does not exist in schema {schema}, skipping")
                continue

            for alert_type, sql_template in queries.items():
                query_start = time.monotonic()
                rows = run_schema_query(connection, schema, sql_template.format(schema=schema))
                logger.info(f"{alert_type} query for {schema} completed in {time.monotonic() - query_start:.2f}s")

                if not rows:
                    continue

                for row in rows:
                    row["schema"] = schema
                logger.info(f"Found {len(rows)} {alert_type} rows in schema {schema}")
                send_sns_alert(rows, alert_type, resource)
                totals[alert_type] += len(rows)

        connection.close()

        execution_time = time.monotonic() - start_time
        logger.info(f"{resource} alert scan completed in {execution_time:.2f}s")

        return {
            "statusCode": 200,
            "resource": resource,
            "execution_time_seconds": execution_time,
            **{f"{alert_type.lower()}_found": count for alert_type, count in totals.items()},
        }
    except Exception as e:
        logger.error(f"{resource} alert scan failed: {str(e)}")
        return {"statusCode": 500, "resource": resource, "error": str(e)}

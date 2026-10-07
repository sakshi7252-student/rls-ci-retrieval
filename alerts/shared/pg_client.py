"""
Shared Postgres helpers for the stuck/exhausted job alert Lambdas.
Connects to the rls-backend multi-tenant database (each tenant = one schema).
"""

import os
import logging
import pg8000

logger = logging.getLogger()

DB_HOST = os.environ.get("DB_HOST")
DB_NAME = os.environ.get("DB_NAME")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
ENVIRONMENT = os.environ.get("ENVIRONMENT", "unknown")


def get_db_connection():
    logger.info(f"Connecting to database {DB_NAME} on host {DB_HOST}")
    connection = pg8000.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        database=DB_NAME,
    )
    return connection


def get_all_schemas(connection) -> list[str]:
    cursor = connection.cursor()
    cursor.execute("""
        SELECT schema_name
        FROM information_schema.schemata
        WHERE schema_name NOT LIKE 'pg_%'
        AND schema_name != 'information_schema'
    """)
    schemas = [row[0] for row in cursor.fetchall()]
    cursor.close()
    return schemas


def check_table_exists(connection, schema: str, table: str) -> bool:
    try:
        cursor = connection.cursor()
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = %s
                AND table_name = %s
            )
        """, (schema, table))
        exists = cursor.fetchone()[0]
        cursor.close()
        return exists
    except Exception as e:
        logger.error(f"Error checking table existence in schema {schema}: {str(e)}")
        try:
            connection.rollback()
        except Exception:
            pass
        return False


def run_schema_query(connection, schema: str, query: str) -> list[dict]:
    """
    Run a schema-qualified query (query must already have `{schema}` formatted in)
    and return rows as list of dicts using cursor.description column names.
    On error, roll back so the next schema/table check isn't left in a broken
    transaction state, and return an empty list.
    """
    try:
        cursor = connection.cursor()
        cursor.execute(query)
        columns = [c[0] for c in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        cursor.close()
        return rows
    except Exception as e:
        logger.error(f"Error running query in schema {schema}: {str(e)}")
        try:
            connection.rollback()
        except Exception:
            pass
        return []

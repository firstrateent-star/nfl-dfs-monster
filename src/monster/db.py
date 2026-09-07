from __future__ import annotations
import psycopg


def connect(database_url: str):
    return psycopg.connect(database_url)


def check(database_url: str) -> dict:
    with connect(database_url) as conn, conn.cursor() as cur:
        cur.execute("select current_database(), current_user")
        db, user = cur.fetchone()
        cur.execute("select count(*) from information_schema.tables where table_schema='monster'")
        tables = cur.fetchone()[0]
        return {"database": db, "user": user, "monster_tables": tables}

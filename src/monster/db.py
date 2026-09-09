from __future__ import annotations

import psycopg


def check(database_url: str) -> dict:
    with psycopg.connect(database_url) as conn, conn.cursor() as cur:
        cur.execute("select current_database(), current_user, version()")
        db, user, version = cur.fetchone()
        return {"database": db, "user": user, "version": version}

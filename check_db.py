import sqlite3
from datetime import datetime

DB_PATH = "instance/sentinel_audit.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def insert_analysis(text, result):

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT,
        toxicity TEXT,
        sentiment TEXT,
        bias TEXT,
        overall_risk TEXT,
        identity_mention INTEGER,
        created_at TEXT
    )
    """)

    cur.execute("""
    INSERT INTO history
    (text,toxicity,sentiment,bias,overall_risk,identity_mention,created_at)
    VALUES (?,?,?,?,?,?,?)
    """,(
        text,
        result["toxicity"],
        result["sentiment"],
        result["bias"],
        result["overall_risk"],
        int(result["identity_mention"]),
        datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()


def get_history(limit=20, offset=0):

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT * FROM history ORDER BY id DESC LIMIT ? OFFSET ?",
        (limit, offset)
    )

    rows = cur.fetchall()

    conn.close()

    return rows
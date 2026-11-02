
import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATABASE_PATH = (
    PROJECT_ROOT / "database" / "inspection_history.db"
)


def get_connection():
    """Create a database connection."""

    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row

    return connection


def initialize_database():
    """Create the inspection table."""

    connection = get_connection()

    try:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS inspections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                operator TEXT NOT NULL,
                part_id TEXT NOT NULL,
                anomaly_score REAL,
                ai_signal TEXT,
                decision TEXT NOT NULL,
                p1_status TEXT,
                p2_status TEXT,
                p3_status TEXT,
                p4_status TEXT,
                comments TEXT
            )
        """)

        connection.commit()

    finally:
        connection.close()


def save_inspection(record):
    """Save an inspection record."""

    connection = get_connection()

    try:
        cursor = connection.execute("""
            INSERT INTO inspections (
                timestamp,
                operator,
                part_id,
                anomaly_score,
                ai_signal,
                decision,
                p1_status,
                p2_status,
                p3_status,
                p4_status,
                comments
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record["timestamp"],
            record["operator"],
            record["part_id"],
            record["anomaly_score"],
            record["ai_signal"],
            record["decision"],
            record["P1"],
            record["P2"],
            record["P3"],
            record["P4"],
            record["comments"]
        ))

        connection.commit()

        return cursor.lastrowid

    finally:
        connection.close()


def get_all_inspections():
    """Retrieve all saved inspections."""

    connection = get_connection()

    try:
        cursor = connection.execute("""
            SELECT *
            FROM inspections
            ORDER BY id DESC
        """)

        return [
            dict(row)
            for row in cursor.fetchall()
        ]

    finally:
        connection.close()


def get_inspection_count():
    """Get the total number of inspections."""

    connection = get_connection()

    try:
        cursor = connection.execute(
            "SELECT COUNT(*) FROM inspections"
        )

        return cursor.fetchone()[0]

    finally:
        connection.close()

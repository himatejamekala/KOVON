"""
create_db.py

Creates users.db (SQLite) with a `users` table and seeds it with sample data.
Run this once before starting app.py:

    python create_db.py
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.db")


def create_and_seed_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS users")

    cursor.execute(
        """
        CREATE TABLE users (
            user_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            membership_tier TEXT NOT NULL
        )
        """
    )

    sample_users = [
        (101, "Riya Sharma", "Gold"),
        (102, "Aman Verma", "Silver"),
        (103, "Neha Iyer", "Platinum"),
    ]

    cursor.executemany(
        "INSERT INTO users (user_id, name, membership_tier) VALUES (?, ?, ?)",
        sample_users,
    )

    conn.commit()
    conn.close()
    print(f"Database created and seeded at: {DB_PATH}")


if __name__ == "__main__":
    create_and_seed_db()

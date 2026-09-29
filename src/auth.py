"""
Authentication module for RAG Knowledge Assistant.
Uses SQLite for persistent user storage and PBKDF2-HMAC-SHA256 for secure password hashing.
No external dependencies — uses Python built-in hashlib and sqlite3.
"""

import sqlite3
import hashlib
import os
import re
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from src.config import BASE_DIR
from utils.logging_utils import get_logger

logger = get_logger("auth")

AUTH_DB_PATH = BASE_DIR / "auth.db"


def _get_connection() -> sqlite3.Connection:
    """Returns a connection to the authentication SQLite database."""
    conn = sqlite3.connect(str(AUTH_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_auth_db() -> None:
    """Initialize the authentication database with the users table."""
    conn = _get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE COLLATE NOCASE,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    logger.info("Authentication database initialized.")


def _hash_password(password: str) -> str:
    """Hash a password using PBKDF2-HMAC-SHA256 with a random 32-byte salt (100k iterations)."""
    salt = os.urandom(32)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return salt.hex() + ":" + key.hex()


def _verify_password(stored_hash: str, password: str) -> bool:
    """Verify a plaintext password against its stored PBKDF2 hash."""
    try:
        salt_hex, key_hex = stored_hash.split(":")
        salt = bytes.fromhex(salt_hex)
        key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
        return key.hex() == key_hex
    except (ValueError, AttributeError):
        return False


def validate_email(email: str) -> bool:
    """Validate email format using regex."""
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email.strip()))


def validate_password(password: str) -> Tuple[bool, str]:
    """Validate password strength (min 6 chars, at least one digit)."""
    if len(password) < 6:
        return False, "Password must be at least 6 characters long."
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one digit."
    return True, ""


def create_user(full_name: str, email: str, password: str) -> Tuple[bool, str]:
    """
    Register a new user account.

    Returns:
        (success: bool, message: str)
    """
    full_name = full_name.strip()
    email = email.strip().lower()

    if not full_name:
        return False, "Full name is required."
    if not validate_email(email):
        return False, "Please enter a valid email address."

    valid, msg = validate_password(password)
    if not valid:
        return False, msg

    try:
        conn = _get_connection()
        existing = conn.execute(
            "SELECT id FROM users WHERE email = ?", (email,)
        ).fetchone()
        if existing:
            conn.close()
            return False, "An account with this email already exists."

        password_hash = _hash_password(password)
        conn.execute(
            "INSERT INTO users (full_name, email, password_hash) VALUES (?, ?, ?)",
            (full_name, email, password_hash),
        )
        conn.commit()
        conn.close()
        logger.info(f"New user registered: {email}")
        return True, "Account created successfully! Please log in."
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        return False, "An error occurred during registration. Please try again."


def authenticate_user(email: str, password: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Authenticate a user by email and password.

    Returns:
        (success: bool, user_info: dict | None)
    """
    if not email or not password:
        return False, None

    try:
        conn = _get_connection()
        user = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email.strip().lower(),)
        ).fetchone()
        conn.close()

        if user and _verify_password(user["password_hash"], password):
            return True, {
                "id": user["id"],
                "full_name": user["full_name"],
                "email": user["email"],
                "created_at": user["created_at"],
            }
        return False, None
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return False, None


# Initialize database on module import
init_auth_db()

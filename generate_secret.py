#!/usr/bin/env python3
"""
Generate a secure SECRET_KEY for Flask production deployment
Usage: python generate_secret.py
"""

import secrets
import string

def generate_secret_key(length=32):
    """Generate a cryptographically secure random string"""
    alphabet = string.ascii_letters + string.digits + string.punctuation
    return ''.join(secrets.choice(alphabet) for _ in range(length))

if __name__ == "__main__":
    key = generate_secret_key()
    print("=" * 60)
    print("SECURE SECRET_KEY FOR FLASK PRODUCTION")
    print("=" * 60)
    print(f"\nCopy this to your Render environment variable 'SECRET_KEY':\n")
    print(f"{key}\n")
    print("=" * 60)

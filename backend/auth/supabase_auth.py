"""Supabase JWT verification and FastAPI auth dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import requests
from fastapi import Depends, Header, HTTPException, status

from config.config import (
    AUTH_REQUIRED,
    SUPABASE_ANON_KEY,
    SUPABASE_SERVICE_ROLE_KEY,
    SUPABASE_URL,
)


@dataclass
class AuthUser:
    id: str
    email: Optional[str] = None


def _extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip()


def _fetch_supabase_user(access_token: str) -> Optional[AuthUser]:
    if not SUPABASE_URL:
        return None
    api_key = SUPABASE_ANON_KEY or SUPABASE_SERVICE_ROLE_KEY
    if not api_key:
        return None

    url = f"{SUPABASE_URL.rstrip('/')}/auth/v1/user"
    resp = requests.get(
        url,
        headers={
            "Authorization": f"Bearer {access_token}",
            "apikey": api_key,
        },
        timeout=10,
    )
    if resp.status_code != 200:
        return None
    payload = resp.json() or {}
    user_id = payload.get("id")
    if not user_id:
        return None
    return AuthUser(id=user_id, email=payload.get("email"))


def get_current_user(
    authorization: Optional[str] = Header(default=None),
) -> AuthUser:
    token = _extract_bearer_token(authorization)

    if not token:
        if AUTH_REQUIRED:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing Bearer token",
            )
        # Dev fallback identity when auth is disabled.
        return AuthUser(id="anonymous")

    user = _fetch_supabase_user(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    return user


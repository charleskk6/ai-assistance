"""Bearer-token auth.

Constant-time comparison, and the token is never logged or echoed back.
"""

from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, Request, status


async def require_token(
    request: Request, authorization: str | None = Header(default=None)
) -> None:
    expected: str = request.app.state.settings.local_assistant_token
    if not expected:  # pragma: no cover - startup refuses this state
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Server has no token configured.")

    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(token.strip(), expected):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Missing or invalid bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

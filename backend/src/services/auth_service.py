"""
Auth Service - Community Edition Stub

Community Edition has no authentication.
Enterprise Edition includes full user management and authentication.
"""

from typing import Optional

class DummyUser:
    """Dummy user for Community Edition"""
    id: int = 1
    username: str = "community_user"
    email: str = "user@community.local"
    is_active: bool = True

async def get_current_user() -> DummyUser:
    """Returns dummy user for Community Edition"""
    return DummyUser()

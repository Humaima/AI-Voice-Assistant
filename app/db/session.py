"""
Async SQLAlchemy engine and session management (Phase 5).

Kept separate from app/db/models.py so tests can build an engine
pointed at in-memory SQLite (fast, no external service) while
production uses the real Postgres URL from settings — same ORM models,
different backing database.
"""
from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.db.models import Base

settings = get_settings()


def build_engine(database_url: str | None = None, **engine_kwargs) -> AsyncEngine:
    """Build an async engine. Defaults to settings.database_url (real
    Postgres); tests pass an explicit sqlite+aiosqlite URL (plus e.g.
    poolclass=StaticPool, needed so in-memory SQLite is shared across
    the multiple connections an async sessionmaker opens) instead."""
    return create_async_engine(database_url or settings.database_url, future=True, **engine_kwargs)


def build_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def create_all_tables(engine: AsyncEngine) -> None:
    """Create tables from the ORM models directly — fine for local dev
    and for this project's current single-table schema. Production
    deployments with evolving schemas should use Alembic migrations
    instead (Phase 10); this is intentionally not a replacement for
    that, just a fast way to get a working dev database now."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

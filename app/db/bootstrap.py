"""
One-time setup: creates the `messages` table used by
PostgresConversationStore (Phase 5).

Deliberately NOT run automatically on app startup — Phases 2-4 work
fine with zero database configured, and eagerly connecting to Postgres
in the FastAPI lifespan would break that for anyone not yet using
persistent memory. Run this explicitly once before your first call to
an endpoint that uses conversation memory:

    python -m app.db.bootstrap

Uses CREATE TABLE IF NOT EXISTS semantics (via SQLAlchemy's
create_all), so it's safe to run more than once. For an evolving
production schema, replace this with real Alembic migrations
(Phase 10) — this script is a fast path to a working dev database now,
not a migration system.
"""
import asyncio

from app.core.logging_config import configure_logging, get_logger
from app.db.session import build_engine, create_all_tables

configure_logging()
logger = get_logger(__name__)


async def main() -> None:
    engine = build_engine()
    logger.info("Creating tables (if not already present)...")
    await create_all_tables(engine)
    await engine.dispose()
    logger.info("Done. The 'messages' table is ready.")


if __name__ == "__main__":
    asyncio.run(main())

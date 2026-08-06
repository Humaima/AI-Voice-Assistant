"""
Quick-start setup for local development: creates the `messages` table
used by PostgresConversationStore (Phase 5), via SQLAlchemy's
create_all (CREATE TABLE IF NOT EXISTS semantics — safe to run more
than once, but doesn't track schema changes over time).

    python -m app.db.bootstrap

Phase 10 added real Alembic migrations (migrations/ directory) — use
`alembic upgrade head` instead of this script for anything beyond
local quick-start: production deployments, or once you've made schema
changes and need them versioned/tracked. This script remains here
purely for the fast "just get a working dev database" path Phases 5-9
were built and tested against — it isn't being deprecated, just isn't
the recommended path for anything beyond local dev anymore.

Deliberately NOT run automatically on app startup — Phases 2-4 work
fine with zero database configured, and eagerly connecting to Postgres
in the FastAPI lifespan would break that for anyone not yet using
persistent memory.
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

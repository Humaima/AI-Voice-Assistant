#!/bin/sh
# Container entrypoint (Phase 10).
#
# Runs database migrations automatically on every container start,
# before the server starts accepting traffic. This exists specifically
# so migrations don't depend on any platform-specific "extra" feature
# (like Render's Shell tab, which turned out to require a paid plan) —
# this works on literally any platform's free tier, since it's just
# the container's own normal startup process.
#
# Safe to run on every startup: Alembic tracks which revisions have
# already been applied (in the alembic_version table) and is a no-op
# if there's nothing new to apply.
#
# Known limitation, stated honestly: if you ever scale to multiple
# instances starting simultaneously, they could race to apply the same
# migration at once. Not a concern at this project's current
# single-instance free-tier scale — worth knowing if that ever changes,
# at which point a dedicated one-off "release" step (separate from the
# web process) would be the more correct approach.
set -e

echo "Running database migrations..."
alembic upgrade head

echo "Starting server..."
# exec replaces this shell process with uvicorn (rather than running
# it as a child process), so uvicorn becomes PID 1 and receives
# shutdown signals directly — needed for the container to actually
# stop promptly when the platform tries to stop/restart it.
exec uvicorn app.main:app --host 0.0.0.0 --port 8000

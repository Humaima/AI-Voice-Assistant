"""
Structured logging setup.

We use plain stdlib logging (not print) so that log level, format, and
destination are configured once and respected everywhere, including by
uvicorn's own loggers. JSON-ish key=value format keeps logs greppable
and easy to ship to a log aggregator later (Phase 10).
"""
import logging
import sys

from app.core.config import get_settings


def configure_logging() -> None:
    settings = get_settings()

    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    logging.basicConfig(
        level=settings.log_level.upper(),
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    # Keep third-party libraries quieter than our own app logs by default.
    for noisy_logger in ("httpx", "chromadb", "sentence_transformers"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging configured | env=%s | level=%s",
        settings.environment,
        settings.log_level,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

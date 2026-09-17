"""
ai_db.logger
Minimal structured logger. Activated with AI_DB_DEBUG=1 env var.
All normal operation is silent; debug output goes to stderr only.
"""
import logging
import os

_logger = logging.getLogger("ai_db")

if os.environ.get("AI_DB_DEBUG"):
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[ai-db %(levelname)s] %(message)s"))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.DEBUG)
else:
    _logger.addHandler(logging.NullHandler())

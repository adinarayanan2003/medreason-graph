from __future__ import annotations

import json
import logging
import sys
from typing import Any


def configure_logging(verbose: bool = False) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, stream=sys.stderr, format="%(message)s")


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    if not logger.isEnabledFor(logging.INFO):
        return
    logger.info(json.dumps({"event": event, **fields}, sort_keys=True))


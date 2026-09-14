import logging
import time
from typing import Optional

from retflow.config import get_logger


class RateLimitedLogger:
    """A logger that rate-limits to one message every X seconds or Y call."""

    def __init__(self, time_interval: int = 5, call_interval: Optional[int] = None):
        """Filter calls to ``log`` based on the time and the number of calls.

        The logger only allows one message to be logged every ``time_interval``
        (measured in seconds). If ``call_interval`` is given, the logger will
        only allow one message to be logged every ``call_interval`` calls.

        Args:
            time_interval: Limit messages to one every time_interval
            call_interval: Limit messages to one every call_interval
        """
        self.last_log: float | None = None
        self.ignored_calls = 0
        self.time_interval = time_interval
        self.call_interval = call_interval

    def _should_log(self) -> bool:
        if self.last_log is None:
            return True

        enough_time = time.perf_counter() - self.last_log > self.time_interval
        enough_calls = (
            self.call_interval is not None and self.ignored_calls >= self.call_interval
        )

        return enough_time or enough_calls

    def log(self, msg, level=logging.INFO, force=False) -> None:
        """Log a message if the rate limit is not exceeded.

        Args:
            msg: Message to log
            level: Level to log at (default: INFO)
            force: Force logging (disregard rate limit)
        """

        if force:
            get_logger().log(level=level, msg=msg)
        elif self._should_log():
            get_logger().log(level=level, msg=msg)
            self.last_log = time.perf_counter()
            self.ignored_calls = 0
        else:
            self.ignored_calls += 1

"""Question limits for a public deployment: a number per visitor per hour, and a total per day.

Every answered question spends the Groq and Mistral quota of whoever owns the keys, and the free tier covers
about 115 checked questions a day. Without a limit, one visitor or one script can use that up in minutes.

Both limits are sliding windows held in memory, so they reset when the process restarts and only work with one
API process, which is all this service supports anyway. The per-visitor limit trusts the visitor id it is
given: it stops ordinary overuse, not someone who forges ids. The daily total is the limit that can't be
dodged.
"""
import threading
import time
from collections import deque
from collections.abc import Callable

HOUR, DAY = 3600.0, 86400.0
MAX_VISITORS = 10_000  # the most visitor ids remembered; the oldest are dropped past this


class AskLimiter:
    def __init__(self, per_visitor_per_hour: int = 0, per_day: int = 0, clock: Callable[[], float] = time.monotonic):
        """0 turns a limit off."""
        self.per_visitor, self.per_day, self.clock = per_visitor_per_hour, per_day, clock
        self._lock = threading.Lock()
        self._visitors: dict[str, deque[float]] = {}
        self._all: deque[float] = deque()

    @property
    def enabled(self) -> bool:
        return bool(self.per_visitor or self.per_day)

    def check(self, visitor: str) -> tuple[str, int] | None:
        """None if the visitor may ask now; otherwise (which limit, seconds until a slot frees up)."""
        now = self.clock()
        with self._lock:
            self._prune(self._all, now - DAY)
            if self.per_day and len(self._all) >= self.per_day:
                return "daily", int(self._all[0] + DAY - now) + 1
            asked = self._visitors.get(visitor)
            if asked is not None:
                self._prune(asked, now - HOUR)
                if not asked:
                    del self._visitors[visitor]
                elif self.per_visitor and len(asked) >= self.per_visitor:
                    return "visitor", int(asked[0] + HOUR - now) + 1
        return None

    def record(self, visitor: str) -> None:
        """Count one question that cost quota."""
        now = self.clock()
        with self._lock:
            self._all.append(now)
            if self.per_visitor:
                if visitor not in self._visitors and len(self._visitors) >= MAX_VISITORS:
                    del self._visitors[next(iter(self._visitors))]
                self._visitors.setdefault(visitor, deque()).append(now)

    @staticmethod
    def _prune(times: deque[float], before: float) -> None:
        while times and times[0] <= before:
            times.popleft()

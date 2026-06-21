from __future__ import annotations

import asyncio
from collections import deque
from threading import Lock
from typing import Any


class EventBus:
    def __init__(self, max_history: int = 1000) -> None:
        self._history: deque[dict[str, Any]] = deque(maxlen=max_history)
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._lock = Lock()

    def publish(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._history.append(event)
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            try:
                subscriber.put_nowait(event)
            except asyncio.QueueFull:
                pass

    def recent(self, run_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            events = list(self._history)
        if run_id is not None:
            events = [event for event in events if event.get("run_id") == run_id]
        return events[-int(limit):]

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
        with self._lock:
            self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        with self._lock:
            self._subscribers.discard(queue)


default_event_bus = EventBus()

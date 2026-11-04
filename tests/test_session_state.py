"""State synchronisation across Home Assistant (re)connections.

Regression tests for a dial bound to a timer that kept showing the timer
as stopped while it was running: leaked per-session tasks restarted the dial
refresh loops with a stale snapshot, and changes landing between the
initial ``get_states`` and the ``subscribe_events`` were lost.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import Mock

import pytest
import websockets
import websockets.exceptions
from StreamDeck.Devices.StreamDeckPlus import StreamDeckPlus

import home_assistant_streamdeck_yaml as h

TIMER = "timer.minuteur_projecteur_projection"
T1 = "2026-11-04T15:00:00.000000+00:00"
T2 = "2026-11-04T15:56:13.829622+00:00"


def _timer(state: str, last_updated: str) -> dict[str, Any]:
    attributes: dict[str, Any] = {"duration": "5:00:00", "remaining": "5:00:00"}
    if state == "active":
        attributes["finishes_at"] = "2099-01-01T00:00:00+00:00"
    return {
        "entity_id": TIMER,
        "state": state,
        "attributes": attributes,
        "last_changed": last_updated,
        "last_updated": last_updated,
    }


def _event(
    new_state: dict[str, Any] | None,
    old_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "type": "event",
        "id": 0,
        "event": {
            "event_type": "state_changed",
            "data": {"entity_id": TIMER, "new_state": new_state, "old_state": old_state},
        },
    }


class FakeWebSocket:
    """Scripted Home Assistant websocket.

    ``before_snapshot`` are events delivered after the subscription is
    acknowledged but before the ``get_states`` result. Once the script is
    exhausted, ``recv`` raises ``ConnectionClosedError`` like a dropped link.
    """

    def __init__(
        self,
        snapshot: list[dict[str, Any]],
        before_snapshot: list[dict[str, Any]],
    ) -> None:
        """Script a session."""
        self.snapshot = snapshot
        self.before_snapshot = before_snapshot
        self.queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self.sent: list[dict[str, Any]] = []

    async def send(self, payload: str) -> None:
        """Answer the subscription and the snapshot request."""
        msg = json.loads(payload)
        self.sent.append(msg)
        if msg["type"] == "subscribe_events":
            await self.queue.put(
                {"type": "result", "id": msg["id"], "success": True, "result": None},
            )
        elif msg["type"] == "get_states":
            for event in self.before_snapshot:
                await self.queue.put(event)
            await self.queue.put(
                {"type": "result", "id": msg["id"], "success": True, "result": self.snapshot},
            )
            await self.queue.put(None)  # then the link drops

    async def recv(self) -> str:
        """Return the next scripted message or drop the link."""
        msg = await self.queue.get()
        if msg is None:
            raise websockets.exceptions.ConnectionClosedError(None, None)
        return json.dumps(msg)


def _deck() -> Mock:
    deck = Mock(spec=StreamDeckPlus)
    deck.key_image_format.return_value = {
        "size": (StreamDeckPlus.KEY_PIXEL_WIDTH, StreamDeckPlus.KEY_PIXEL_HEIGHT),
        "format": StreamDeckPlus.KEY_IMAGE_FORMAT,
        "flip": StreamDeckPlus.KEY_FLIP,
        "rotation": StreamDeckPlus.KEY_ROTATION,
    }
    deck.touchscreen_image_format.return_value = {
        "size": (StreamDeckPlus.TOUCHSCREEN_PIXEL_WIDTH, StreamDeckPlus.TOUCHSCREEN_PIXEL_HEIGHT),
        "format": StreamDeckPlus.TOUCHSCREEN_IMAGE_FORMAT,
        "flip": StreamDeckPlus.TOUCHSCREEN_FLIP,
        "rotation": StreamDeckPlus.TOUCHSCREEN_ROTATION,
    }
    deck.key_count.return_value = 8
    deck.dial_count.return_value = 4
    deck.is_visual.return_value = True
    deck.__enter__ = Mock(return_value=deck)
    deck.__exit__ = Mock(return_value=False)
    return deck


def _config() -> h.Config:
    dial = h.Dial(
        entity_id=TIMER,
        service="homeassistant.update_entity",
        dial_event_type="TURN",
        refresh_every_secs=1,
        attributes={"min": 0, "max": 100000, "step": 1},
        text='{{ "ACTIVE" if is_state("' + TIMER + '", "active") else "OFF" }}',
    )
    return h.Config(pages=[h.Page(name="home", buttons=[], dials=[dial])], auto_reload=False)


def _session_tasks() -> list[asyncio.Task]:
    names = {"process_websocket_messages", "call_update_timers", "watch_configuration_file"}
    return [
        t
        for t in asyncio.all_tasks()
        if not t.done() and getattr(t.get_coro(), "__name__", "") in names
    ]


async def test_get_states_matches_its_own_result_and_buffers_events() -> None:
    """The snapshot is the result of our own request; events are buffered."""
    ws = FakeWebSocket(
        snapshot=[_timer("idle", T1)],
        before_snapshot=[_event(_timer("active", T2))],
    )
    await ws.send(
        json.dumps({"type": "subscribe_events", "event_type": "state_changed", "id": 999}),
    )
    pending: list[dict[str, Any]] = []
    snapshot = await h.get_states(ws, pending)
    # The subscription acknowledgement (a ``result`` with ``null``) is not
    # mistaken for the snapshot.
    assert snapshot[TIMER]["state"] == "idle"
    assert len(pending) == 1


def test_merge_pending_events_never_rolls_back() -> None:
    """A buffered event applies only when it is not older than the snapshot."""
    newer = {TIMER: _timer("idle", T1)}
    h._merge_pending_events(newer, [_event(_timer("active", T2))])
    assert newer[TIMER]["state"] == "active"

    older = {TIMER: _timer("active", T2)}
    h._merge_pending_events(older, [_event(_timer("idle", T1))])
    assert older[TIMER]["state"] == "active"


async def test_reconnect_with_timer_already_active(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disconnect, then reconnect while the timer is already running."""
    sockets = [
        # 1st session: timer idle, then the link drops.
        FakeWebSocket(snapshot=[_timer("idle", T1)], before_snapshot=[]),
        # 2nd session: the timer is started between the subscription and
        # the snapshot (snapshot still says idle, the event is newer).
        FakeWebSocket(
            snapshot=[_timer("idle", T1)],
            before_snapshot=[_event(_timer("active", T2))],
        ),
    ]

    @asynccontextmanager
    async def fake_setup_ws(*_args: Any, **_kwargs: Any):  # noqa: ANN202
        yield sockets.pop(0)

    monkeypatch.setattr(h, "setup_ws", fake_setup_ws)
    config = _config()
    deck = _deck()
    state: h.StateDict = {}

    try:
        with pytest.raises(websockets.exceptions.ConnectionClosed):
            await h._run_connection_session(
                "ha",
                "token",
                "wss",
                config,
                deck,
                complete_state=state,
            )
        await asyncio.sleep(0)
        # No task of the dead session survives it.
        assert _session_tasks() == []
        # Source unavailable: the last known state is kept, nothing is
        # invented (no empty state that would render as "off").
        assert state[TIMER]["state"] == "idle"

        with pytest.raises(websockets.exceptions.ConnectionClosed):
            await h._run_connection_session(
                "ha",
                "token",
                "wss",
                config,
                deck,
                complete_state=state,
            )
        await asyncio.sleep(0)
        assert _session_tasks() == []
        assert state[TIMER]["state"] == "active"
        dial = config.pages[0].dials[0]
        assert dial.rendered_template_dial(state).text == "ACTIVE"
        # The dial refresh loops run against the very dict that the next
        # sessions refresh in place, never against an old copy.
        assert len(config._dial_refresh_tasks) == 1
        coro = config._dial_refresh_tasks[0].get_coro()
        assert coro.cr_frame.f_locals["complete_state"] is state
    finally:
        for task in config._dial_refresh_tasks:
            task.cancel()
        await asyncio.gather(*config._dial_refresh_tasks, return_exceptions=True)


async def test_subscription_precedes_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """``subscribe_events`` is sent before ``get_states``."""
    ws = FakeWebSocket(snapshot=[_timer("active", T2)], before_snapshot=[])

    @asynccontextmanager
    async def fake_setup_ws(*_args: Any, **_kwargs: Any):  # noqa: ANN202
        yield ws

    monkeypatch.setattr(h, "setup_ws", fake_setup_ws)
    config = _config()
    try:
        with pytest.raises(websockets.exceptions.ConnectionClosed):
            await h._run_connection_session(
                "ha",
                "token",
                "wss",
                config,
                _deck(),
                complete_state={},
            )
        assert [m["type"] for m in ws.sent[:2]] == ["subscribe_events", "get_states"]
    finally:
        for task in config._dial_refresh_tasks:
            task.cancel()
        await asyncio.gather(*config._dial_refresh_tasks, return_exceptions=True)

"""Adapter contract tests with minimal AstrBot API doubles."""

import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


class Star:
    def __init__(self, context):
        self.context = context


class Filters:
    def __getattr__(self, name):
        return lambda *args, **kwargs: lambda fn: fn


def load_plugin():
    package = types.ModuleType("pace_contract")
    package.__path__ = [str(ROOT)]
    api = types.ModuleType("astrbot.api")
    api.AstrBotConfig = dict
    api.logger = Mock()
    event_api = types.ModuleType("astrbot.api.event")
    event_api.AstrMessageEvent = object
    event_api.filter = Filters()
    star_api = types.ModuleType("astrbot.api.star")
    star_api.Context = object
    star_api.Star = Star
    doubles = {
        "astrbot": types.ModuleType("astrbot"), "astrbot.api": api,
        "astrbot.api.event": event_api, "astrbot.api.star": star_api,
        "pace_contract": package,
    }
    with patch.dict(sys.modules, doubles):
        spec = importlib.util.spec_from_file_location("pace_contract.main", ROOT / "main.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module


MODULE = load_plugin()


def plain(text):
    return types.SimpleNamespace(type="Plain", text=text)


class Event:
    unified_msg_origin = "qq:GroupMessage:1"

    def __init__(self, text="鲸鱼，你好", raw=None, sender="A", private=False):
        self.text = text
        self.components = [plain(raw if raw is not None else text)]
        self.sender = sender
        self.private = private
        self.extras = {}
        self.stopped = False
        self.result = types.SimpleNamespace(
            is_llm_result=lambda: True, async_stream=None,
        )
        self.fail_send = False
        self.transport_calls = 0

    def get_self_id(self): return "bot"
    def get_sender_id(self): return self.sender
    def get_messages(self): return self.components
    def get_message_str(self): return self.text
    def is_private_chat(self): return self.private
    def get_extra(self, key, default=None): return self.extras.get(key, default)
    def set_extra(self, key, value): self.extras[key] = value
    def stop_event(self): self.stopped = True
    def get_result(self): return self.result
    def plain_result(self, text): return text

    async def send(self, chain):
        if self.fail_send:
            raise RuntimeError("transport failed")
        self.transport_calls += 1

    async def send_streaming(self, generator, *args, **kwargs):
        async for chain in generator:
            await self.send(chain)


class PluginTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.context = types.SimpleNamespace(get_config=Mock(return_value={"wake_prefix": ["/", ""]}))
        self.plugin = MODULE.GroupPacePlugin(self.context, {
            "reply_probability": 0, "address_words": ["鲸鱼"],
        })
        self.now = Mock(return_value=1000)
        self.clock = patch.object(MODULE, "time", types.SimpleNamespace(monotonic=self.now))
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.chain = types.SimpleNamespace(chain=[plain("你好")])

    async def test_original_prefix_and_session_config(self):
        event = Event(text="你好", raw="/你好")
        await self.plugin.pace_gate(event)
        self.assertFalse(event.stopped)
        self.context.get_config.assert_called_with(umo=event.unified_msg_origin)
        event = Event(text="你好", raw="!你好")
        self.context.get_config.return_value = {"wake_prefix": ["!"]}
        await self.plugin.pace_gate(event)
        self.assertFalse(event.stopped)

    async def test_empty_wake_flag_does_not_bypass_probability(self):
        event = Event("普通闲聊")
        event.is_at_or_wake_command = True
        await self.plugin.pace_gate(event)
        self.assertTrue(event.stopped)

    async def test_address_prefix_and_contains(self):
        self.assertFalse(self.plugin._is_addressed(Event(), "今天看了鲸鱼纪录片"))
        self.assertFalse(self.plugin._is_addressed(Event(), "鲸鱼纪录片好看"))
        self.assertTrue(self.plugin._is_addressed(Event(), "鲸鱼，帮我看看"))
        self.plugin.config["address_match_mode"] = "contains"
        self.assertTrue(self.plugin._is_addressed(Event(), "今天看了鲸鱼纪录片"))
        for component in [types.SimpleNamespace(type="At", qq="bot"), types.SimpleNamespace(type="Reply", sender_id="bot")]:
            event = Event("普通消息")
            event.components.append(component)
            self.assertTrue(self.plugin._is_addressed(event, event.text))

    async def test_invalid_config_is_finite_and_safe(self):
        for invalid in (123, True, {}, None):
            self.plugin.config["address_words"] = invalid
            self.assertEqual(self.plugin._cfg_list("address_words"), [])
        for invalid in (float("nan"), float("inf"), "oops", True):
            self.plugin.config["reply_probability"] = invalid
            self.assertEqual(self.plugin._settings().probability, 0.2)
        self.plugin.config["enabled"] = "false"
        self.assertTrue(self.plugin._cfg_bool("enabled", True))

    async def test_private_and_disabled_bypass(self):
        await self.plugin.pace_gate(Event(private=True))
        self.assertFalse(self.plugin.controller.sessions)
        self.plugin.config["enabled"] = False
        await self.plugin.pace_gate(Event())
        self.assertFalse(self.plugin.controller.sessions)

    async def test_delayed_transport_and_other_sender(self):
        event = Event()
        original_send = event.send
        await self.plugin.pace_gate(event)
        await self.plugin.track_delivery(event)
        state = self.plugin.controller.sessions[event.unified_msg_origin]
        self.assertIsNone(state.last_reply)
        self.now.return_value = 1060
        await event.send(self.chain)
        await self.plugin.finish_delivery(event)
        self.assertEqual(state.last_reply, 1060)
        self.assertEqual(event.send, original_send)
        self.now.return_value = 1061
        followup = Event("接着说")
        await self.plugin.pace_gate(followup)
        self.assertFalse(followup.stopped)
        stranger = Event("别人闲聊", sender="B")
        await self.plugin.pace_gate(stranger)
        self.assertTrue(stranger.stopped)

    async def test_failed_send_releases_without_window_or_cooldown(self):
        self.plugin.config["reply_probability"] = 1
        event = Event("普通消息")
        await self.plugin.pace_gate(event)
        await self.plugin.track_delivery(event)
        event.fail_send = True
        with self.assertRaises(RuntimeError):
            await event.send(self.chain)
        await self.plugin.finish_delivery(event)
        state = self.plugin.controller.sessions[event.unified_msg_origin]
        self.assertIsNone(state.last_random)
        self.assertEqual(state.sent, 0)
        self.assertFalse(state.pending)
        self.assertFalse(state.conversations)

    async def test_error_result_never_opens_window(self):
        event = Event()
        await self.plugin.pace_gate(event)
        event.result.is_llm_result = lambda: False
        await self.plugin.track_delivery(event)
        await event.send(self.chain)
        await self.plugin.finish_delivery(event)
        state = self.plugin.controller.sessions[event.unified_msg_origin]
        self.assertEqual(state.sent, 0)
        self.assertFalse(state.pending)

    async def test_segmented_sender_counts_once_and_restores(self):
        event = Event()
        original_send = event.send
        await self.plugin.pace_gate(event)
        await self.plugin.track_delivery(event)
        pace_send = event.send
        async def segmented(chain):
            await pace_send(chain)
            self.now.return_value += 2
            await pace_send(chain)
        event.send = segmented
        await event.send(self.chain)
        # Segmented plugin restores the inner sender first, then pace restores.
        event.send = pace_send
        await self.plugin.finish_delivery(event)
        state = self.plugin.controller.sessions[event.unified_msg_origin]
        self.assertEqual(state.sent, 1)
        self.assertEqual(state.last_reply, 1002)
        self.assertEqual(event.send, original_send)

    async def test_stream_success_empty_and_failure_release_reservations(self):
        self.plugin.config["reply_probability"] = 1
        for mode in ("success", "empty", "failure"):
            self.plugin.controller.clear()
            event = Event("普通消息")
            original = event.send_streaming
            await self.plugin.pace_gate(event)
            async def chunks():
                if mode != "empty":
                    yield self.chain
            event.fail_send = mode == "failure"
            self.now.return_value += 60
            await self.plugin.model_completed(event, types.SimpleNamespace(role="assistant"))
            if event.fail_send:
                with self.assertRaises(RuntimeError):
                    await event.send_streaming(chunks())
            else:
                await event.send_streaming(chunks())
            state = self.plugin.controller.sessions[event.unified_msg_origin]
            self.assertFalse(state.pending)
            self.assertEqual(state.sent, int(mode == "success"))
            self.assertEqual(event.send_streaming, original)

    async def test_stream_error_message_does_not_open_window(self):
        self.plugin.config["reply_probability"] = 1
        event = Event("普通消息")
        await self.plugin.pace_gate(event)
        async def chunks():
            yield self.chain
            await self.plugin.model_completed(event, types.SimpleNamespace(role="err"))
        await event.send_streaming(chunks())
        state = self.plugin.controller.sessions[event.unified_msg_origin]
        self.assertEqual(state.sent, 0)
        self.assertFalse(state.pending)
        self.assertIsNone(state.last_random)

    async def test_expired_automatic_reply_is_not_sent(self):
        self.plugin.config.update(reply_probability=1, pending_timeout_seconds=10)
        event = Event("普通消息")
        await self.plugin.pace_gate(event)
        await self.plugin.track_delivery(event)
        self.now.return_value += 11
        await event.send(self.chain)
        await self.plugin.finish_delivery(event)
        self.assertEqual(event.transport_calls, 0)
        self.assertEqual(self.plugin.controller.sessions[event.unified_msg_origin].sent, 0)

    async def test_slow_transport_keeps_random_reservation(self):
        self.plugin.config.update(reply_probability=1, pending_timeout_seconds=10)
        event = Event("普通消息")
        started, resume = asyncio.Event(), asyncio.Event()
        async def slow_send(chain):
            started.set()
            await resume.wait()
        event.send = slow_send
        await self.plugin.pace_gate(event)
        await self.plugin.track_delivery(event)
        self.now.return_value += 9
        task = asyncio.create_task(event.send(self.chain))
        await started.wait()
        self.now.return_value += 2
        other = Event("别的消息", sender="B")
        await self.plugin.pace_gate(other)
        self.assertTrue(other.stopped)
        resume.set()
        await task
        await self.plugin.finish_delivery(event)

    async def test_status_has_empty_records_and_double_probability_note(self):
        self.context.get_config.return_value = {
            "wake_prefix": ["/"],
            "provider_ltm_settings": {"active_reply": {"enable": True, "method": "possibility_reply", "possibility_reply": .1}},
        }
        status = [item async for item in self.plugin.pace_status(Event())][0]
        self.assertIn("暂无记录", status)
        self.assertIn("未配置空前缀", status)
        self.assertIn("先经过该概率筛选", status)


if __name__ == "__main__":
    unittest.main()

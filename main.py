"""Filter local AstrBot LLM requests before the session lock and merge window."""

import math
import random
import time

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star

from .pacing import PaceController, Settings


class GroupPacePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.controller = PaceController()
        self._warnings: set[str] = set()

    def _warn(self, key: str, message: str):
        if key not in self._warnings:
            self._warnings.add(key)
            logger.warning(f"[pace] {message}")

    def _cfg_float(self, key: str, default: float, lo: float, hi: float) -> float:
        try:
            raw = self.config.get(key, default)
            if isinstance(raw, bool):
                raise ValueError
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            self._warn(key, f"配置 {key} 无效，使用默认值 {default}")
            value = default
        return max(lo, min(hi, value))

    def _cfg_bool(self, key: str, default: bool) -> bool:
        value = self.config.get(key, default)
        if isinstance(value, bool):
            return value
        self._warn(key, f"配置 {key} 必须为布尔值，使用默认值 {default}")
        return default

    def _cfg_list(self, key: str) -> list[str]:
        raw = self.config.get(key, [])
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, (list, tuple)):
            self._warn(key, f"配置 {key} 必须为列表，忽略无效值")
            return []
        return [item.strip() for item in raw if isinstance(item, str) and item.strip()]

    def _settings(self) -> Settings:
        return Settings(
            probability=self._cfg_float("reply_probability", 0.2, 0, 1),
            followup_seconds=self._cfg_float("followup_window_seconds", 45, 0, 3600),
            cooldown_seconds=self._cfg_float("cooldown_seconds", 180, 0, 86400),
            max_followups=int(self._cfg_float("max_followup_replies", 3, 0, 100)),
            max_conversation_seconds=self._cfg_float("max_conversation_seconds", 180, 0, 3600),
            pending_seconds=self._cfg_float("pending_timeout_seconds", 300, 10, 3600),
        )

    @staticmethod
    def _mentions_self(event: AstrMessageEvent) -> bool:
        self_id = str(event.get_self_id())
        return any(
            (getattr(component, "type", None) == "At" and str(getattr(component, "qq", "")) == self_id)
            or (getattr(component, "type", None) == "Reply" and str(getattr(component, "sender_id", "")) == self_id)
            for component in event.get_messages()
        )

    def _session_config(self, event):
        return self.context.get_config(umo=event.unified_msg_origin)

    def _is_addressed(self, event: AstrMessageEvent, text: str) -> bool:
        if self._mentions_self(event):
            return True
        # WakingCheck strips prefixes from event.message_str, but leaves the
        # original Plain components intact. Never use is_at_or_wake_command:
        # an empty wake prefix sets that flag for ordinary messages too.
        original = "".join(
            getattr(component, "text", "")
            for component in event.get_messages()
            if getattr(component, "type", None) == "Plain"
        ).lstrip()
        prefixes = self._session_config(event).get("wake_prefix", [])
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        if isinstance(prefixes, (list, tuple)) and any(
            isinstance(prefix, str) and prefix and original.startswith(prefix)
            for prefix in prefixes
        ):
            return True
        mode = self.config.get("address_match_mode", "prefix")
        lowered = text.casefold()
        for word in self._cfg_list("address_words"):
            word = word.casefold()
            if mode == "contains" and word in lowered:
                return True
            if lowered.startswith(word):
                suffix = lowered[len(word):]
                if not suffix or suffix[0].isspace() or suffix[0] in "，,。.!！?？:：、~～":
                    return True
        return False

    @filter.on_waiting_llm_request(priority=100)
    async def pace_gate(self, event: AstrMessageEvent):
        if not self._cfg_bool("enabled", True) or event.is_private_chat():
            return
        if event.get_extra("_group_pace_ticket") is not None:
            return
        ticket, reason = self.controller.decide(
            event.unified_msg_origin, str(event.get_sender_id()),
            self._is_addressed(event, (event.get_message_str() or "").strip()),
            time.monotonic(), self._settings(), random.random,
        )
        logger.debug(f"[pace] {'放行' if ticket else '静默'} {reason}: {event.unified_msg_origin}")
        if ticket is None:
            event.stop_event()
        else:
            event.set_extra("_group_pace_ticket", ticket)
            self._track_stream(event, ticket)

    def _track_stream(self, event, ticket):
        # Streaming bypasses decorating_result and after_message_sent in the
        # host, so observe the transport entry point independently.
        original = event.send_streaming
        event.set_extra("_group_pace_original_stream", original)

        async def tracked_stream(generator, *args, **kwargs):
            if not self.controller.can_send(event.unified_msg_origin, ticket, time.monotonic(), self._settings()):
                close = getattr(generator, "aclose", None)
                if close is not None:
                    await close()
                self.controller.release(event.unified_msg_origin, ticket)
                event.send_streaming = original
                event.set_extra("_group_pace_original_stream", None)
                return
            has_content = False

            async def observed():
                nonlocal has_content
                async for chain in generator:
                    if (getattr(chain, "chain", None)
                            and getattr(chain, "type", None) not in {"reasoning", "tool_call", "tool_call_result", "end"}):
                        has_content = True
                    yield chain

            try:
                response = await original(observed(), *args, **kwargs)
                if has_content and event.get_extra("_group_pace_model_completed", False):
                    self.controller.delivered(
                        event.unified_msg_origin, ticket, time.monotonic(), self._settings(),
                    )
                return response
            finally:
                self.controller.release(event.unified_msg_origin, ticket)
                event.send_streaming = original
                event.set_extra("_group_pace_original_stream", None)

        event.send_streaming = tracked_stream

    @filter.on_llm_response(priority=10000)
    async def model_completed(self, event: AstrMessageEvent, response):
        if event.get_extra("_group_pace_ticket") is not None:
            role = getattr(response, "role", None)
            event.set_extra("_group_pace_model_completed", role == "assistant")
            if role == "err":
                self.controller.release(event.unified_msg_origin, event.get_extra("_group_pace_ticket"))

    @filter.on_decorating_result(priority=10000)
    async def track_delivery(self, event: AstrMessageEvent):
        """Observe successful transport calls, including segmented replies.

        after_message_sent also runs after transport failures in AstrBot, so it
        alone cannot prove delivery. Wrap send only during response delivery;
        await the original sender before recording success.
        """
        ticket = event.get_extra("_group_pace_ticket")
        result = event.get_result()
        if ticket is None or result is None or event.get_extra("_group_pace_send_wrapped", False):
            return
        if not result.is_llm_result():
            self.controller.release(event.unified_msg_origin, ticket)
            return
        original_send = event.send
        event.set_extra("_group_pace_original_send", original_send)
        event.set_extra("_group_pace_send_wrapped", True)

        async def tracked_send(chain, *args, **kwargs):
            if not self.controller.can_send(event.unified_msg_origin, ticket, time.monotonic(), self._settings()):
                self.controller.release(event.unified_msg_origin, ticket)
                return
            try:
                response = await original_send(chain, *args, **kwargs)
            except BaseException:
                self.controller.release(event.unified_msg_origin, ticket)
                raise
            if getattr(chain, "chain", None):
                self.controller.delivered(
                    event.unified_msg_origin, ticket, time.monotonic(), self._settings(),
                )
            return response

        event.send = tracked_send

    @filter.after_message_sent(priority=-20000)
    async def finish_delivery(self, event: AstrMessageEvent):
        original = event.get_extra("_group_pace_original_send")
        if original is not None:
            event.send = original
            event.set_extra("_group_pace_original_send", None)
            event.set_extra("_group_pace_send_wrapped", False)
        original_stream = event.get_extra("_group_pace_original_stream")
        if original_stream is not None:
            event.send_streaming = original_stream
            event.set_extra("_group_pace_original_stream", None)
        ticket = event.get_extra("_group_pace_ticket")
        if ticket is not None:
            self.controller.release(event.unified_msg_origin, ticket)

    @filter.command("pace")
    async def pace_status(self, event: AstrMessageEvent):
        """查看本会话发言节奏、窗口余量与请求统计。"""
        settings = self._settings()
        now = time.monotonic()
        state = self.controller.session(event.unified_msg_origin, now, settings)
        conversation = state.conversations.get(str(event.get_sender_id()))
        remaining = self.controller.remaining(conversation, now, settings) if conversation else 0
        reserved = sum(
            ticket.reason == "followup" and ticket.sender == str(event.get_sender_id())
            and conversation is not None and ticket.generation == conversation.generation
            for ticket in state.pending.values()
        )
        rounds = max(0, settings.max_followups - conversation.followups - reserved) if conversation else 0
        cooldown = max(0, settings.cooldown_seconds - (now - state.last_random)) if state.last_random is not None else 0
        last = f"{now - state.last_reply:.0f}s 前" if state.last_reply is not None else "暂无记录"
        cfg = self._session_config(event)
        prefixes = cfg.get("wake_prefix", [])
        active = cfg.get("provider_ltm_settings", {}).get("active_reply", {})
        wake_note = "空前缀唤醒已配置" if "" in prefixes else "未配置空前缀；普通消息须由框架或其他插件唤醒"
        active_note = ""
        if active.get("enable") and active.get("method") == "possibility_reply":
            active_note = f"\n框架概率主动回复已启用（{active.get('possibility_reply')}）；适用会话会先经过该概率筛选。"
        passed = state.passed
        yield event.plain_result(
            f"发言节奏：{'已启用' if self._cfg_bool('enabled', True) else '已关闭'}\n"
            f"插话概率 {settings.probability:.0%}｜接话窗口 {settings.followup_seconds:.0f}s｜插话冷却 {settings.cooldown_seconds:.0f}s\n"
            f"你的接话窗口剩余 {remaining:.0f}s，可继续 {rounds} 轮；本群插话冷却剩余 {cooldown:.0f}s\n"
            f"最近成功回复：{last}；待处理请求 {len(state.pending)}\n"
            f"本会话请求放行 {sum(passed.values())}（点名 {passed.get('addressed', 0)} / 接话 {passed.get('followup', 0)} / 随机 {passed.get('random', 0)}），"
            f"静默 {sum(state.blocked.values())}，成功回复事件 {state.sent}\n"
            f"{wake_note}{active_note}\n"
            "统计不含被合并插件提前接收的后续消息；流式回复在非空输出发送完成后计数。"
        )

    async def terminate(self):
        self.controller.clear()

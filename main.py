"""群聊发言节奏控制（reply pacing）

背景：
    cmd_config.json 里 wake_prefix 设成 ["/", ""]，空串让群里每条消息都唤醒机器人，
    再由白名单决定哪些会话参与。但“唤醒”在框架里等于“一定回复”，于是群里每句话都被回，
    几个人同时说话就变成逐条刷屏。

本插件在“即将调用 LLM”之前加一道闸门，判定规则：

    1. 被点名（@机器人 / 回复机器人的消息 / 消息里出现叫到它的词 / 斜杠命令）
       → 一定回复，并开启一段“会话窗口”
    2. 会话窗口内（默认 45 秒）的后续消息
       → 视为接着聊，正常回复（避免刚聊两句就被判定为插话而突然不理人）
    3. 窗口外的普通聊天
       → 按概率插话；两次插话之间还要满足冷却（默认 180 秒）

判定为“不回复”时直接 event.stop_event()，框架不会发起 LLM 请求，
既省 token、也不占用会话锁，更不会触发 reply_merge 的“思考”请求。

挂在 on_waiting_llm_request（早于会话锁，也早于 reply_merge 的零频道开窗），
priority=100 让它排在 reply_merge（默认 0）之前。
"""

import random
import time
from typing import Dict, Tuple

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star


class GroupPacePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        # umo -> (最近一次回复时间, 最近一次“主动插话”时间)
        self._state: Dict[str, Tuple[float, float]] = {}
        self._stat_passed = 0
        self._stat_blocked = 0

    # ---------- 配置读取（脏配置不该把插件搞崩） ----------

    def _cfg_float(self, key: str, default: float, lo: float, hi: float) -> float:
        try:
            value = float(self.config.get(key, default))
        except (TypeError, ValueError):
            logger.warning(f"[pace] 配置 {key} 不是数字，回退默认值 {default}")
            value = default
        return max(lo, min(hi, value))

    def _cfg_list(self, key: str) -> list[str]:
        raw = self.config.get(key) or []
        if isinstance(raw, str):
            raw = [raw]
        return [str(item).strip() for item in raw if str(item).strip()]

    # ---------- 是否被点名 ----------

    @staticmethod
    def _mentions_self(event: AstrMessageEvent) -> bool:
        self_id = str(event.get_self_id())
        for component in event.get_messages():
            kind = getattr(component, "type", None)
            if kind == "At" and str(getattr(component, "qq", "")) == self_id:
                return True
            if kind == "Reply" and str(getattr(component, "sender_id", "")) == self_id:
                return True
        return False

    def _is_addressed(self, event: AstrMessageEvent, text: str) -> bool:
        # @机器人 / 引用回复机器人的消息
        if self._mentions_self(event):
            return True
        # 斜杠命令：命令自己会输出，不参与节奏控制
        for prefix in self.context.get_config().get("wake_prefix") or []:
            if prefix and text.startswith(prefix):
                return True
        # 叫到名字
        lowered = text.lower()
        for word in self._cfg_list("address_words"):
            if word.lower() in lowered:
                return True
        return False

    # ---------- 闸门 ----------

    @filter.on_waiting_llm_request(priority=100)
    async def pace_gate(self, event: AstrMessageEvent):
        if not self.config.get("enabled", True):
            return
        # 私聊不限制节奏
        if event.is_private_chat():
            return

        text = (event.get_message_str() or "").strip()
        umo = event.unified_msg_origin
        now = time.monotonic()

        probability = self._cfg_float("reply_probability", 0.2, 0.0, 1.0)
        followup = self._cfg_float("followup_window_seconds", 45, 0, 3600)
        cooldown = self._cfg_float("cooldown_seconds", 180, 0, 86400)

        last_reply, last_random = self._state.get(umo, (0.0, 0.0))

        if self._is_addressed(event, text):
            self._state[umo] = (now, last_random)
            self._stat_passed += 1
            logger.debug(f"[pace] 被点名，放行: {umo}")
            return

        # 刚聊过，属于同一段对话的后续，不必重新掷骰子
        if now - last_reply <= followup:
            self._state[umo] = (now, last_random)
            self._stat_passed += 1
            logger.info(f"[pace] 会话窗口内接话，放行: {umo} | {text[:20]}")
            return

        since_random = now - last_random
        if since_random < cooldown:
            self._stat_blocked += 1
            logger.info(
                f"[pace] 未被点名且插话冷却中（{since_random:.0f}s < {cooldown:.0f}s），静默: "
                f"{umo} | {text[:20]}"
            )
            event.stop_event()
            return

        if random.random() < probability:
            self._state[umo] = (now, now)
            self._stat_passed += 1
            logger.info(f"[pace] 未被点名但命中插话概率 {probability:.0%}，放行: {umo}")
            return

        self._stat_blocked += 1
        logger.info(
            f"[pace] 未被点名且未命中概率 {probability:.0%}，静默: {umo} | {text[:20]}"
        )
        event.stop_event()

    @filter.command("pace")
    async def pace_status(self, event: AstrMessageEvent):
        """查看当前发言节奏设置与统计"""
        probability = self._cfg_float("reply_probability", 0.2, 0.0, 1.0)
        followup = self._cfg_float("followup_window_seconds", 45, 0, 3600)
        cooldown = self._cfg_float("cooldown_seconds", 180, 0, 86400)
        umo = event.unified_msg_origin
        last_reply, last_random = self._state.get(umo, (0.0, 0.0))
        now = time.monotonic()
        yield event.plain_result(
            f"发言节奏：插话概率 {probability:.0%}｜会话窗口 {followup:.0f}s｜插话冷却 {cooldown:.0f}s\n"
            f"本会话：距上次回复 {now - last_reply:.0f}s，距上次主动插话 "
            f"{now - last_random:.0f}s\n"
            f"累计 放行/静默：{self._stat_passed}/{self._stat_blocked}"
        )

    async def terminate(self):
        self._state.clear()

"""Pure pacing state machine. Timestamps use an injected monotonic clock."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    probability: float = 0.2
    followup_seconds: float = 45
    cooldown_seconds: float = 180
    max_followups: int = 3
    max_conversation_seconds: float = 180
    pending_seconds: float = 300


@dataclass
class Conversation:
    started: float
    last_reply: float
    generation: int
    followups: int = 0


@dataclass
class Ticket:
    number: int
    sender: str
    reason: str
    created: float
    generation: int | None = None
    delivered: bool = False
    sending: bool = False


@dataclass
class Session:
    touched: float
    last_reply: float | None = None
    last_random: float | None = None
    conversations: dict[str, Conversation] = field(default_factory=dict)
    pending: dict[int, Ticket] = field(default_factory=dict)
    passed: dict[str, int] = field(default_factory=dict)
    blocked: dict[str, int] = field(default_factory=dict)
    sent: int = 0


class PaceController:
    def __init__(self):
        self.sessions: dict[str, Session] = {}
        self._sequence = 0
        self._last_cleanup = 0.0

    def session(self, key: str, now: float, settings: Settings) -> Session:
        if now - self._last_cleanup >= 60:
            self.sessions = {
                key: state for key, state in self.sessions.items()
                if now - state.touched < 86400
            }
            self._last_cleanup = now
        state = self.sessions.setdefault(key, Session(now))
        state.touched = now
        state.pending = {
            token: ticket for token, ticket in state.pending.items()
            if ticket.sending or now - ticket.created < settings.pending_seconds
        }
        state.conversations = {
            sender: conversation for sender, conversation in state.conversations.items()
            if self.remaining(conversation, now, settings) > 0
            or any(ticket.sender == sender and ticket.generation == conversation.generation
                   for ticket in state.pending.values())
        }
        return state

    @staticmethod
    def remaining(conversation: Conversation, now: float, settings: Settings) -> float:
        return max(0.0, min(
            settings.followup_seconds - (now - conversation.last_reply),
            settings.max_conversation_seconds - (now - conversation.started),
        ))

    def decide(self, key: str, sender: str, addressed: bool, now: float,
               settings: Settings, draw) -> tuple[Ticket | None, str]:
        state = self.session(key, now, settings)
        conversation = state.conversations.get(sender)
        reason = "addressed" if addressed else ""
        generation = None
        if not reason and conversation and self.remaining(conversation, now, settings) > 0:
            reserved = sum(
                ticket.reason == "followup" and ticket.sender == sender
                and ticket.generation == conversation.generation
                for ticket in state.pending.values()
            )
            if conversation.followups + reserved < settings.max_followups:
                reason = "followup"
                generation = conversation.generation
            else:
                return self._block(state, "followup_limit")
        if not reason:
            if any(ticket.sender == sender for ticket in state.pending.values()):
                return self._block(state, "pending")
            if any(ticket.reason == "random" for ticket in state.pending.values()):
                return self._block(state, "pending")
            if state.last_random is not None and now - state.last_random < settings.cooldown_seconds:
                return self._block(state, "cooldown")
            if draw() >= settings.probability:
                return self._block(state, "probability")
            reason = "random"
        self._sequence += 1
        ticket = Ticket(self._sequence, sender, reason, now, generation)
        state.pending[ticket.number] = ticket
        state.passed[reason] = state.passed.get(reason, 0) + 1
        return ticket, reason

    @staticmethod
    def _block(state: Session, reason: str):
        state.blocked[reason] = state.blocked.get(reason, 0) + 1
        return None, reason

    def can_send(self, key: str, ticket: Ticket, now: float, settings: Settings) -> bool:
        state = self.session(key, now, settings)
        # A reply already being delivered may finish its remaining segments.
        # Explicit questions are never discarded due to a slow provider.
        allowed = (ticket.delivered or ticket.reason == "addressed"
                   or state.pending.get(ticket.number) is ticket)
        if allowed:
            ticket.sending = True
        return allowed

    def delivered(self, key: str, ticket: Ticket, now: float, settings: Settings) -> None:
        state = self.session(key, now, settings)
        # Call only after a successful transport; can_send is checked before
        # starting automatic replies so expired reservations cannot resurface.
        first = not ticket.delivered
        ticket.delivered = True
        state.pending.pop(ticket.number, None)
        state.last_reply = now
        if first:
            state.sent += 1
        if ticket.reason == "random":
            state.last_random = now
        conversation = state.conversations.get(ticket.sender)
        if ticket.reason in {"addressed", "random"}:
            if first and (conversation is None or ticket.number > conversation.generation):
                conversation = Conversation(now, now, ticket.number)
                state.conversations[ticket.sender] = conversation
            elif conversation and conversation.generation == ticket.number:
                conversation.last_reply = now
        elif conversation and conversation.generation == ticket.generation:
            if first:
                conversation.followups += 1
            conversation.last_reply = now

    def release(self, key: str, ticket: Ticket) -> None:
        state = self.sessions.get(key)
        if state:
            state.pending.pop(ticket.number, None)

    def clear(self) -> None:
        self.sessions.clear()

"""Deterministic, dependency-free tests for the pacing state machine.

Run from the project root with:
    python3 -m unittest discover -s tests -p 'test_pacing.py' -v
"""

from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

# Import the pure module without loading the plugin's framework-dependent entrypoint.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pacing import PaceController, Settings  # noqa: E402


class PaceControllerTests(unittest.TestCase):
    KEY = "group:one"

    def setUp(self):
        self.controller = PaceController()
        self.settings = Settings(
            probability=0,
            followup_seconds=30,
            cooldown_seconds=100,
            max_followups=3,
            max_conversation_seconds=90,
            pending_seconds=300,
        )

    def decide(self, sender, now, addressed=False, settings=None, draw=None, key=None):
        return self.controller.decide(
            self.KEY if key is None else key,
            sender,
            addressed,
            now,
            self.settings if settings is None else settings,
            (lambda: 1.0) if draw is None else draw,
        )

    def grant(self, sender, now, reason, addressed=False, settings=None, key=None):
        ticket, actual_reason = self.decide(
            sender, now, addressed=addressed, settings=settings, key=key,
            draw=lambda: 0.0,
        )
        self.assertIsNotNone(ticket, actual_reason)
        self.assertEqual(actual_reason, reason)
        self.assertEqual(ticket.reason, reason)
        return ticket

    def deliver(self, ticket, now, settings=None, key=None):
        self.controller.delivered(
            self.KEY if key is None else key,
            ticket,
            now,
            self.settings if settings is None else settings,
        )

    def start(self, sender, now, settings=None, key=None):
        ticket = self.grant(
            sender, now, "addressed", addressed=True, settings=settings, key=key,
        )
        self.deliver(ticket, now, settings=settings, key=key)
        return ticket

    def state(self):
        return self.controller.sessions[self.KEY]

    def assert_blocked(self, sender, now, reason, settings=None, key=None, draw=None):
        ticket, actual_reason = self.decide(
            sender, now, settings=settings, key=key, draw=draw,
        )
        self.assertIsNone(ticket)
        self.assertEqual(actual_reason, reason)

    def test_different_members_do_not_share_followup_windows(self):
        self.start("alice", 10)
        self.start("bob", 25)

        alice_followup = self.grant("alice", 39, "followup")
        self.controller.release(self.KEY, alice_followup)
        self.assert_blocked("alice", 40, "probability")
        self.grant("bob", 40, "followup")
        self.assertNotIn("alice", self.state().conversations)
        self.assertEqual(self.state().conversations["bob"].started, 25)

    def test_reply_to_one_member_does_not_extend_another_members_window(self):
        self.start("alice", 10)
        self.start("bob", 15)
        followup = self.grant("bob", 35, "followup")
        self.deliver(followup, 35)

        self.assert_blocked("alice", 40, "probability")
        self.grant("bob", 40, "followup")

    def test_sessions_do_not_share_conversations_or_random_pending(self):
        settings = Settings(probability=1)
        first = self.grant("alice", 0, "random", settings=settings)
        second = self.grant("alice", 1, "random", settings=settings, key="group:two")
        self.assertNotEqual(first.number, second.number)
        self.deliver(first, 2, settings=settings)
        self.assertNotIn("alice", self.controller.sessions["group:two"].conversations)

    def test_window_starts_at_successful_delivery_not_decision(self):
        ticket = self.grant("alice", 0, "addressed", addressed=True)
        self.assertNotIn("alice", self.state().conversations)

        self.deliver(ticket, 50)
        conversation = self.state().conversations["alice"]
        self.assertEqual((conversation.started, conversation.last_reply), (50, 50))
        followup = self.grant("alice", 79, "followup")
        self.controller.release(self.KEY, followup)
        self.assert_blocked("alice", 80, "probability")

    def test_followup_window_extends_only_after_successful_send(self):
        self.start("alice", 10)
        followup = self.grant("alice", 20, "followup")
        self.assertEqual(self.state().conversations["alice"].last_reply, 10)
        self.deliver(followup, 35)
        self.assertEqual(self.state().conversations["alice"].last_reply, 35)

        next_followup = self.grant("alice", 64, "followup")
        self.controller.release(self.KEY, next_followup)
        self.assert_blocked("alice", 65, "probability")

    def test_max_followups_counts_delivered_and_pending_reservations(self):
        settings = Settings(probability=0, max_followups=2)
        self.start("alice", 10, settings=settings)
        first = self.grant("alice", 11, "followup", settings=settings)
        second = self.grant("alice", 12, "followup", settings=settings)
        self.assertEqual(self.state().conversations["alice"].followups, 0)
        self.assert_blocked("alice", 13, "followup_limit", settings=settings)

        self.deliver(first, 14, settings=settings)
        self.assertEqual(self.state().conversations["alice"].followups, 1)
        self.assert_blocked("alice", 15, "followup_limit", settings=settings)
        self.deliver(second, 16, settings=settings)
        self.assertEqual(self.state().conversations["alice"].followups, 2)
        self.assert_blocked("alice", 17, "followup_limit", settings=settings)

    def test_random_probability_cannot_bypass_exhausted_followup_limit(self):
        settings = Settings(probability=1, max_followups=1)
        self.start("alice", 0, settings=settings)
        followup = self.grant("alice", 1, "followup", settings=settings)
        self.deliver(followup, 2, settings=settings)
        draw = Mock(side_effect=AssertionError("exhausted quota must precede randomness"))

        self.assert_blocked("alice", 3, "followup_limit", settings=settings, draw=draw)
        draw.assert_not_called()
        self.assertEqual(self.state().conversations["alice"].followups, 1)
        self.assertEqual(self.state().pending, {})
        self.grant("alice", 47, "random", settings=settings)

    def test_releasing_followup_reservation_restores_capacity(self):
        settings = Settings(probability=0, max_followups=1)
        self.start("alice", 10, settings=settings)
        failed = self.grant("alice", 11, "followup", settings=settings)
        self.assert_blocked("alice", 12, "followup_limit", settings=settings)
        self.controller.release(self.KEY, failed)

        retry = self.grant("alice", 13, "followup", settings=settings)
        self.assertNotEqual(failed.number, retry.number)
        self.assertEqual(self.state().conversations["alice"].followups, 0)

    def test_maximum_conversation_duration_does_not_slide_with_replies(self):
        settings = Settings(
            probability=0, followup_seconds=40, max_conversation_seconds=60,
        )
        self.start("alice", 10, settings=settings)
        first = self.grant("alice", 45, "followup", settings=settings)
        self.deliver(first, 45, settings=settings)
        second = self.grant("alice", 69, "followup", settings=settings)
        self.deliver(second, 69, settings=settings)

        conversation = self.state().conversations["alice"]
        self.assertEqual(conversation.started, 10)
        self.assertEqual(conversation.last_reply, 69)
        self.assertEqual(self.controller.remaining(conversation, 69, settings), 1)
        self.assert_blocked("alice", 70, "probability", settings=settings)
        self.assertNotIn("alice", self.state().conversations)

    def test_addressed_message_reopens_exhausted_conversation(self):
        settings = Settings(probability=0, max_followups=1)
        original = self.start("alice", 10, settings=settings)
        followup = self.grant("alice", 11, "followup", settings=settings)
        self.deliver(followup, 12, settings=settings)
        self.assert_blocked("alice", 13, "followup_limit", settings=settings)

        reopened = self.grant("alice", 20, "addressed", addressed=True, settings=settings)
        self.assertEqual(self.state().conversations["alice"].generation, original.number)
        self.deliver(reopened, 25, settings=settings)
        conversation = self.state().conversations["alice"]
        self.assertEqual(conversation.generation, reopened.number)
        self.assertEqual((conversation.started, conversation.last_reply), (25, 25))
        self.assertEqual(conversation.followups, 0)
        self.grant("alice", 26, "followup", settings=settings)

    def test_addressed_message_reopens_expired_conversation(self):
        self.start("alice", 10)
        reopened = self.grant("alice", 100, "addressed", addressed=True)
        self.deliver(reopened, 105)
        self.assertEqual(self.state().conversations["alice"].started, 105)
        self.grant("alice", 106, "followup")

    def test_addressed_message_bypasses_random_pending_and_cooldown(self):
        settings = Settings(probability=1)
        random_ticket = self.grant("alice", 0, "random", settings=settings)
        addressed = self.grant("bob", 1, "addressed", addressed=True, settings=settings)
        self.deliver(random_ticket, 2, settings=settings)
        self.deliver(addressed, 3, settings=settings)
        self.grant("carol", 4, "addressed", addressed=True, settings=settings)

    def test_random_pending_blocks_concurrent_random_for_other_members(self):
        settings = Settings(probability=1)
        pending = self.grant("alice", 10, "random", settings=settings)
        draw = Mock(side_effect=AssertionError("pending must be checked before randomness"))
        self.assert_blocked("bob", 11, "pending", settings=settings, draw=draw)
        self.assert_blocked("alice", 12, "pending", settings=settings, draw=draw)
        draw.assert_not_called()
        self.assertEqual(list(self.state().pending), [pending.number])

    def test_failed_random_release_does_not_consume_cooldown(self):
        settings = Settings(probability=1)
        failed = self.grant("alice", 10, "random", settings=settings)
        self.controller.release(self.KEY, failed)
        self.assertIsNone(self.state().last_random)
        self.assertIsNone(self.state().last_reply)
        self.assertEqual(self.state().sent, 0)
        self.assertEqual(self.state().conversations, {})
        self.grant("bob", 11, "random", settings=settings)

    def test_failed_followup_release_does_not_extend_window_or_count(self):
        self.start("alice", 10)
        failed = self.grant("alice", 35, "followup")
        self.controller.release(self.KEY, failed)
        conversation = self.state().conversations["alice"]
        self.assertEqual((conversation.last_reply, conversation.followups), (10, 0))
        self.assertEqual(self.state().sent, 1)
        self.assert_blocked("alice", 40, "probability")

    def test_random_cooldown_starts_at_success_not_reservation(self):
        settings = Settings(probability=1, cooldown_seconds=100)
        ticket = self.grant("alice", 0, "random", settings=settings)
        self.deliver(ticket, 40, settings=settings)
        self.assertEqual(self.state().last_random, 40)
        self.assert_blocked("bob", 139, "cooldown", settings=settings)
        self.grant("bob", 140, "random", settings=settings)

    def test_addressed_and_followup_sends_do_not_reset_random_cooldown(self):
        settings = Settings(probability=1, cooldown_seconds=100)
        random_ticket = self.grant("alice", 0, "random", settings=settings)
        self.deliver(random_ticket, 0, settings=settings)
        self.start("bob", 50, settings=settings)
        followup = self.grant("bob", 60, "followup", settings=settings)
        self.deliver(followup, 60, settings=settings)

        self.assertEqual(self.state().last_random, 0)
        self.assert_blocked("carol", 99, "cooldown", settings=settings)
        self.grant("carol", 100, "random", settings=settings)

    def test_timestamp_zero_is_a_valid_conversation_start(self):
        self.start("alice", 0)
        conversation = self.state().conversations["alice"]
        self.assertEqual((conversation.started, conversation.last_reply), (0, 0))
        self.assertEqual(self.controller.remaining(conversation, 0, self.settings), 30)
        followup = self.grant("alice", 29, "followup")
        self.controller.release(self.KEY, followup)
        self.assert_blocked("alice", 30, "probability")

    def test_timestamp_zero_is_a_valid_random_cooldown_start(self):
        settings = Settings(probability=1, cooldown_seconds=100)
        ticket = self.grant("alice", 0, "random", settings=settings)
        self.deliver(ticket, 0, settings=settings)
        self.assertEqual(self.state().last_random, 0)
        self.assert_blocked("bob", 0, "cooldown", settings=settings)
        self.assert_blocked("bob", 99, "cooldown", settings=settings)
        self.grant("bob", 100, "random", settings=settings)

    def test_duplicate_addressed_delivery_counts_once_and_keeps_original_start(self):
        ticket = self.grant("alice", 0, "addressed", addressed=True)
        self.deliver(ticket, 10)
        self.deliver(ticket, 15)
        conversation = self.state().conversations["alice"]
        self.assertTrue(ticket.delivered)
        self.assertEqual(self.state().sent, 1)
        self.assertEqual(self.state().last_reply, 15)
        self.assertEqual((conversation.started, conversation.last_reply), (10, 15))
        self.assertEqual(conversation.followups, 0)
        self.assertNotIn(ticket.number, self.state().pending)

    def test_duplicate_followup_delivery_counts_once_for_split_response(self):
        self.start("alice", 10)
        ticket = self.grant("alice", 11, "followup")
        self.deliver(ticket, 12)
        self.deliver(ticket, 15)
        conversation = self.state().conversations["alice"]
        self.assertEqual(self.state().sent, 2)
        self.assertEqual(conversation.followups, 1)
        self.assertEqual((conversation.started, conversation.last_reply), (10, 15))
        self.assertEqual(self.state().last_reply, 15)

    def test_duplicate_random_delivery_counts_once_and_tracks_latest_success(self):
        settings = Settings(probability=1, cooldown_seconds=100)
        ticket = self.grant("alice", 0, "random", settings=settings)
        self.deliver(ticket, 10, settings=settings)
        self.deliver(ticket, 15, settings=settings)
        self.assertEqual(self.state().sent, 1)
        self.assertEqual(self.state().last_random, 15)
        self.assertEqual(self.state().conversations["alice"].started, 10)
        self.assert_blocked("bob", 114, "cooldown", settings=settings)
        self.grant("bob", 115, "random", settings=settings)

    def test_expired_random_pending_is_removed_at_exact_deadline(self):
        settings = Settings(probability=1, pending_seconds=10)
        expired = self.grant("alice", 0, "random", settings=settings)
        self.assert_blocked("bob", 9, "pending", settings=settings)
        replacement = self.grant("bob", 10, "random", settings=settings)
        self.assertNotIn(expired.number, self.state().pending)
        self.assertIn(replacement.number, self.state().pending)
        self.assertIsNone(self.state().last_random)
        self.assertEqual(self.state().sent, 0)

    def test_expired_followup_pending_releases_reserved_capacity(self):
        settings = Settings(probability=0, max_followups=1, pending_seconds=10)
        self.start("alice", 0, settings=settings)
        expired = self.grant("alice", 1, "followup", settings=settings)
        self.assert_blocked("alice", 10, "followup_limit", settings=settings)
        replacement = self.grant("alice", 11, "followup", settings=settings)
        self.assertNotIn(expired.number, self.state().pending)
        self.assertIn(replacement.number, self.state().pending)
        self.assertEqual(self.state().conversations["alice"].followups, 0)

    def test_expired_automatic_reply_cannot_start_delivery(self):
        settings = Settings(probability=1, pending_seconds=10)
        ticket = self.grant("alice", 0, "random", settings=settings)
        replacement = self.grant("bob", 10, "random", settings=settings)
        self.assertFalse(self.controller.can_send(self.KEY, ticket, 11, settings))
        self.assertTrue(self.controller.can_send(self.KEY, replacement, 11, settings))
        self.assertEqual(self.state().sent, 0)

    def test_sending_reservation_survives_timeout(self):
        settings = Settings(probability=1, pending_seconds=10)
        ticket = self.grant("alice", 0, "random", settings=settings)
        self.assertTrue(self.controller.can_send(self.KEY, ticket, 9, settings))
        self.assert_blocked("bob", 11, "pending", settings=settings)
        self.deliver(ticket, 12, settings=settings)
        self.assert_blocked("bob", 13, "cooldown", settings=settings)

    def test_delayed_followup_preserves_original_generation_and_count(self):
        self.start("alice", 0)
        ticket = self.grant("alice", 29, "followup")
        self.assertTrue(self.controller.can_send(self.KEY, ticket, 50, self.settings))
        self.deliver(ticket, 50)
        conversation = self.state().conversations["alice"]
        self.assertEqual(conversation.started, 0)
        self.assertEqual(conversation.last_reply, 50)
        self.assertEqual(conversation.followups, 1)
        self.grant("alice", 51, "followup")

    def test_older_addressed_ticket_cannot_overwrite_newer_conversation(self):
        older = self.grant("alice", 0, "addressed", addressed=True)
        newer = self.grant("alice", 1, "addressed", addressed=True)
        self.deliver(newer, 2)
        self.deliver(older, 3)
        conversation = self.state().conversations["alice"]
        self.assertEqual(conversation.generation, newer.number)
        self.assertEqual((conversation.started, conversation.last_reply), (2, 2))
        self.assertEqual(conversation.followups, 0)
        self.assertEqual(self.state().sent, 2)

    def test_older_random_ticket_cannot_overwrite_newer_conversation(self):
        settings = Settings(probability=1)
        older = self.grant("alice", 0, "random", settings=settings)
        newer = self.grant("alice", 1, "addressed", addressed=True, settings=settings)
        self.deliver(newer, 2, settings=settings)
        self.deliver(older, 3, settings=settings)
        conversation = self.state().conversations["alice"]
        self.assertEqual(conversation.generation, newer.number)
        self.assertEqual((conversation.started, conversation.last_reply), (2, 2))
        self.assertEqual(self.state().last_random, 3)

    def test_old_followup_ticket_cannot_update_reopened_conversation(self):
        self.start("alice", 0)
        old_followup = self.grant("alice", 1, "followup")
        newer = self.start("alice", 2)
        self.deliver(old_followup, 3)
        conversation = self.state().conversations["alice"]
        self.assertEqual(conversation.generation, newer.number)
        self.assertEqual((conversation.started, conversation.last_reply), (2, 2))
        self.assertEqual(conversation.followups, 0)
        self.assertEqual(self.state().sent, 3)

    def test_duplicate_old_delivery_cannot_update_reopened_conversation(self):
        older = self.start("alice", 0)
        newer = self.start("alice", 2)
        self.deliver(older, 3)
        conversation = self.state().conversations["alice"]
        self.assertEqual(conversation.generation, newer.number)
        self.assertEqual((conversation.started, conversation.last_reply), (2, 2))
        self.assertEqual(self.state().sent, 2)

    def test_old_generation_pending_does_not_consume_new_followup_quota(self):
        settings = Settings(probability=0, max_followups=1)
        self.start("alice", 0, settings=settings)
        old_pending = self.grant("alice", 1, "followup", settings=settings)
        newer = self.start("alice", 2, settings=settings)
        current_pending = self.grant("alice", 3, "followup", settings=settings)
        self.assertNotEqual(old_pending.generation, current_pending.generation)
        self.assertEqual(current_pending.generation, newer.number)
        self.assert_blocked("alice", 4, "followup_limit", settings=settings)


if __name__ == "__main__":
    unittest.main()

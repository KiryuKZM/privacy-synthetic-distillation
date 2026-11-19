"""Validate a bound producer's progress without comparing two machines' UTC clocks."""
from __future__ import annotations


class ObserverFreshness:
    """Per-consumer monotonic freshness of immutable, identity-bound samples.

    An initial observation is not enough: a later sequence and producer monotonic
    timestamp must both advance. Wall-clock epochs are diagnostic only. This
    class reads no processes/files and never fabricates a producer sample.
    """
    def __init__(self, max_age_seconds=15):
        if max_age_seconds <= 0:
            raise ValueError("Positive freshness interval required")
        self.max_age_seconds = max_age_seconds
        self.reset()

    def reset(self):
        self.sequence = None
        self.producer_monotonic_ns = None
        self.last_advance = None
        self.last_local = None

    def check(self, frame, owner, now_monotonic):
        if self.last_local is not None and now_monotonic < self.last_local:
            return dict(ok=False, reason="consumer_monotonic_rollback")
        self.last_local = now_monotonic
        if any(frame.get(k) != owner.get(k) for k in ("pid", "creation_filetime")):
            return dict(ok=False, reason="producer_identity_mismatch")
        if frame.get("entry_sha256") != owner["source"]["sha256"]:
            return dict(ok=False, reason="producer_source_mismatch")
        seq, stamp = frame.get("sequence"), frame.get("observer_monotonic_ns")
        if type(seq) is not int or type(stamp) is not int or seq < 1 or stamp < 1:
            return dict(ok=False, reason="invalid_producer_counters")
        if self.sequence is None:
            self.sequence, self.producer_monotonic_ns = seq, stamp
            return dict(ok=False, reason="awaiting_actual_advance")
        if seq < self.sequence or stamp < self.producer_monotonic_ns:
            return dict(ok=False, reason="producer_counter_rollback")
        if seq > self.sequence:
            if stamp <= self.producer_monotonic_ns:
                return dict(ok=False, reason="sequence_without_producer_time_advance")
            self.sequence, self.producer_monotonic_ns = seq, stamp
            self.last_advance = now_monotonic
        elif stamp != self.producer_monotonic_ns:
            return dict(ok=False, reason="immutable_sequence_changed")
        if self.last_advance is None:
            return dict(ok=False, reason="awaiting_actual_advance")
        age = now_monotonic-self.last_advance
        return dict(ok=age < self.max_age_seconds,
                    reason="fresh_actual_producer" if age < self.max_age_seconds else "producer_stale",
                    local_age_seconds=age, sequence=seq, producer_monotonic_ns=stamp)

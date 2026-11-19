"""Freshness from observed producer sequence progress, independent of wall clocks."""
class SequenceLiveness:
    def __init__(self, max_stale_seconds=3):
        self.max_age_ns=int(max_stale_seconds*1e9)
        self.seq=None;self.producer_ns=None;self.last_change_ns=None;self.live=False
    def observe(self,seq,producer_ns,now_ns):
        if not all(isinstance(x,int) and x>=0 for x in (seq,producer_ns,now_ns)):
            raise PermissionError('Invalid liveness counters')
        if self.seq is None:
            self.seq=seq;self.producer_ns=producer_ns;self.last_change_ns=now_ns
            return False
        if now_ns<self.last_change_ns:raise PermissionError('Local monotonic clock regression')
        if seq<self.seq or producer_ns<self.producer_ns:raise PermissionError('Producer counter regression')
        if now_ns-self.last_change_ns>self.max_age_ns:
            if seq==self.seq:raise PermissionError('No observed producer progress within3s')
            if producer_ns<=self.producer_ns:raise PermissionError('Producer sequence without clock progress')
            # A delayed consumer cannot date this new frame. Requalify with a
            # second advancing frame instead of treating its own delay as stale.
            self.seq=seq;self.producer_ns=producer_ns;self.last_change_ns=now_ns;self.live=False
            return False
        if seq>self.seq:
            if producer_ns<=self.producer_ns:raise PermissionError('Producer sequence without clock progress')
            self.seq=seq;self.producer_ns=producer_ns;self.last_change_ns=now_ns;self.live=True
        return self.live

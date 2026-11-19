"""Low-overhead, dependency-free stage and batch-one generation measurements.

No model imports, token text, implicit synchronization, or resource mutations.
"""
from contextlib import contextmanager
from time import perf_counter_ns


class StageRecorder:
    def __init__(self, clock_ns=perf_counter_ns):
        self.clock_ns = clock_ns
        self.records = []

    @contextmanager
    def span(self, name):
        start = self.clock_ns()
        status = 'completed'
        try:
            yield
        except BaseException:
            status = 'failed'
            raise
        finally:
            end = self.clock_ns()
            if end < start:
                raise ValueError('Monotonic clock moved backwards')
            self.records.append(dict(stage=name, seconds=(end-start)/1e9,
                                     status=status, start_ns=start, end_ns=end))


class GenerationTimer:
    """A passive Hugging Face batch=1 streamer. HF supplies CPU token tensors.

    Its initial put is the input prompt, subsequent puts are generated IDs.
    The CPU copy performed by HF for streaming is profiling overhead; callers
    must disclose it and must not equate one short cold call with GPU capacity.
    """
    def __init__(self, clock_ns=perf_counter_ns):
        self.clock_ns = clock_ns
        self.start_ns = None
        self.end_ns = None
        self.prompt_tokens = None
        self.token_events = []
        self.ended = False

    def begin(self):
        if self.start_ns is not None:
            raise RuntimeError('Timer cannot be reused')
        self.start_ns = self.clock_ns()

    def put(self, value):
        now = self.clock_ns()
        if self.start_ns is None or self.ended:
            raise RuntimeError('Invalid streamer lifecycle')
        shape = tuple(value.shape)
        if len(shape) == 2 and shape[0] != 1:
            raise ValueError('Only batch=1 is supported')
        count = value.numel()
        if self.prompt_tokens is None:
            self.prompt_tokens = count
            return
        if count != 1:
            raise ValueError('One generated token per event is required')
        self.token_events.append(now)

    def end(self):
        self.ended = True

    def finish(self, actual_output_tokens):
        if self.start_ns is None or self.end_ns is not None:
            raise RuntimeError('Invalid or repeated finish')
        self.end_ns = self.clock_ns()
        if actual_output_tokens != len(self.token_events) or not self.ended:
            raise ValueError('Generated token event count or end marker differs')
        times = [self.start_ns, *self.token_events, self.end_ns]
        if any(b < a for a, b in zip(times, times[1:])):
            raise ValueError('Monotonic clock moved backwards')
        elapsed = (self.end_ns-self.start_ns)/1e9
        decode = ((self.token_events[-1]-self.token_events[0])/1e9
                  if actual_output_tokens >= 2 else None)
        return dict(
            clock='perf_counter_ns', batch_size=1,
            prompt_tokens=self.prompt_tokens,
            output_tokens_including_eos=actual_output_tokens,
            generate_call_seconds=elapsed,
            generate_call_CPU_visible_TTFT_seconds=(
                (self.token_events[0]-self.start_ns)/1e9
                if self.token_events else None),
            decode_after_first_CPU_visible_token_seconds=decode,
            decode_after_first_token_per_second=(
                (actual_output_tokens-1)/decode if decode and decode > 0 else None),
            output_inclusive_tokens_per_second=(
                actual_output_tokens/elapsed if elapsed > 0 else None),
            scope='model.generate only; tokenization and postdecode excluded',
            profiling_overhead='HF streamer adds CPU token copies; not overhead-subtracted',
            missing_rate_reason=('fewer than 2 tokens or zero interval'
                                 if not decode else None))

"""
LLM Tracing - structured observability for LLM calls.

Captures:
- Timing (latency per component)
- Tokens (input/output counts)
- Metadata (model, temperature, etc.)
- Hierarchy (parent/child spans)

Pattern: OpenTelemetry-style spans for LLM operations.

Run with: uv run python -m phase5_production.02_evaluation.llm_tracing
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from contextlib import contextmanager
from typing import Generator
from inspect import cleandoc


class SpanKind(Enum):
    """types of LLM operations"""
    EMBEDDING = "embedding"
    RETRIEVAL = "retrieval"
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    CHAIN = "chain"


@dataclass
class Span:
    """single operation in a trace"""
    name: str
    kind: SpanKind
    trace_id: str
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_id: str | None = None
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: datetime | None = None
    attributes: dict = field(default_factory=dict)
    status: str = "ok"
    error: str | None = None

    @property
    def duration_ms(self) -> int | None:
        if self.end_time is None:
            return None
        delta = self.end_time - self.start_time
        return int(delta.total_seconds() * 1000)

    def set_attribute(self, key: str, value) -> None:
        self.attributes[key] = value

    def end(self, status: str = "ok", error: str | None = None) -> None:
        self.end_time = datetime.now(timezone.utc)
        self.status = status
        self.error = error


@dataclass
class Trace:
    """complete trace of an LLM operation"""
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    spans: list[Span] = field(default_factory=list)
    root_span: Span | None = None

    def add_span(self, span: Span) -> None:
        self.spans.append(span)
        if span.parent_id is None:
            self.root_span = span

    @property
    def total_duration_ms(self) -> int:
        if not self.root_span or not self.root_span.duration_ms:
            return 0
        return self.root_span.duration_ms

    @property
    def total_tokens(self) -> int:
        total = 0
        for span in self.spans:
            total += span.attributes.get('tokens.total', 0)
        return total

    def __str__(self) -> str:
        lines = [f"Trace: {self.trace_id[:8]}"]
        for span in self.spans:
            prefix = "└──" if span == self.spans[-1] else "├──"
            duration = f"{span.duration_ms}ms" if span.duration_ms else "running"
            tokens = span.attributes.get('tokens.total', '')
            tokens_str = f" ({tokens} tokens)" if tokens else ""
            lines.append(f"  {prefix} {span.name}: {duration}{tokens_str}")
        lines.append(f"  Total: {self.total_duration_ms}ms, {self.total_tokens} tokens")
        return "\n".join(lines)


class LLMTracer:
    """
    Tracer for LLM operations.

    Usage:
        tracer = LLMTracer()

        with tracer.trace("rag_query") as trace:
            with tracer.span("embedding", SpanKind.EMBEDDING) as span:
                embedding = embed(query)
                span.set_attribute("tokens.total", 10)

            with tracer.span("retrieval", SpanKind.RETRIEVAL) as span:
                docs = retrieve(embedding)
                span.set_attribute("docs.count", len(docs))

            with tracer.span("llm_call", SpanKind.LLM_CALL) as span:
                response = llm.generate(prompt)
                span.set_attribute("tokens.total", 450)
    """

    def __init__(self):
        self._current_trace: Trace | None = None
        self._current_span: Span | None = None
        self._traces: list[Trace] = []

    @contextmanager
    def trace(self, name: str) -> Generator[Trace, None, None]:
        """start a new trace"""
        trace = Trace()
        self._current_trace = trace

        # create root span
        root = Span(
            name=name,
            kind=SpanKind.CHAIN,
            trace_id=trace.trace_id,
        )
        trace.add_span(root)
        self._current_span = root

        try:
            yield trace
        finally:
            root.end()
            self._traces.append(trace)
            self._current_trace = None
            self._current_span = None

    @contextmanager
    def span(self, name: str, kind: SpanKind) -> Generator[Span, None, None]:
        """create a child span within current trace"""
        if not self._current_trace:
            raise RuntimeError("no active trace - use tracer.trace() first")

        parent_id = self._current_span.span_id if self._current_span else None

        span = Span(
            name=name,
            kind=kind,
            trace_id=self._current_trace.trace_id,
            parent_id=parent_id,
        )
        self._current_trace.add_span(span)

        old_span = self._current_span
        self._current_span = span

        try:
            yield span
        except Exception as e:
            span.end(status="error", error=str(e))
            raise
        finally:
            span.end()
            self._current_span = old_span

    def get_traces(self) -> list[Trace]:
        return self._traces


# region Demo Functions

def demo_llm_tracing() -> None:
    """demonstrate LLM tracing"""
    import time

    print("=" * 60)
    print("  LLM Tracing Demo")
    print("=" * 60)

    tracer = LLMTracer()

    print("\nSimulating RAG query with tracing...\n")

    with tracer.trace("rag_query") as trace:
        # embedding
        with tracer.span("embed_query", SpanKind.EMBEDDING) as span:
            time.sleep(0.045)  # simulate API call
            span.set_attribute("model", "text-embedding-3-small")
            span.set_attribute("tokens.total", 12)

        # retrieval
        with tracer.span("vector_search", SpanKind.RETRIEVAL) as span:
            time.sleep(0.015)  # simulate DB query
            span.set_attribute("docs.count", 5)
            span.set_attribute("similarity.min", 0.82)

        # context assembly
        with tracer.span("assemble_context", SpanKind.CHAIN) as span:
            time.sleep(0.002)
            span.set_attribute("context.length", 1500)

        # LLM generation
        with tracer.span("llm_generate", SpanKind.LLM_CALL) as span:
            time.sleep(0.180)  # simulate LLM call
            span.set_attribute("model", "gpt-4")
            span.set_attribute("tokens.prompt", 350)
            span.set_attribute("tokens.completion", 120)
            span.set_attribute("tokens.total", 470)

    print(trace)

    print("\n" + "-" * 60)
    print("\nTrace attributes by span:\n")
    for span in trace.spans:
        if span.attributes:
            print(f"  {span.name}:")
            for k, v in span.attributes.items():
                print(f"    {k}: {v}")

    print("\n" + "=" * 60)
    print("  Key insight: Trace every LLM operation for debugging")
    print("  Tools: OpenTelemetry, LangSmith, Weights & Biases")
    print("=" * 60)

# endregion


if __name__ == "__main__":
    demo_llm_tracing()

"""
LLM Streaming - Server-Sent Events (SSE) for token-by-token output.

LLM responses take 2-5 seconds. Streaming improves UX by showing
tokens as they're generated instead of waiting for completion.

Run with: uv run python -m phase5_production.01_api_design.llm_streaming
"""

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass


@dataclass
class StreamChunk:
    """single chunk in streaming response"""
    content: str
    done: bool = False
    token_count: int = 0


class MockLLMStreamer:
    """
    Mock LLM streaming for demonstration.

    In production: use OpenAI's stream=True or Anthropic streaming.
    """

    def __init__(self, tokens_per_second: float = 20):
        self.delay = 1.0 / tokens_per_second

    async def stream(self, prompt: str) -> AsyncGenerator[StreamChunk, None]:
        """
        Stream response token by token.

        Real implementation:
            async for chunk in client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            ):
                yield chunk.choices[0].delta.content
        """
        # mock response
        response = f"This is a streaming response to: {prompt[:30]}. " \
                   f"Each word appears one at a time to simulate LLM output."

        words = response.split()

        for i, word in enumerate(words):
            await asyncio.sleep(self.delay)
            is_last = i == len(words) - 1
            yield StreamChunk(
                content=word + " ",
                done=is_last,
                token_count=1,
            )


def format_sse(chunk: StreamChunk) -> str:
    """
    Format chunk as Server-Sent Event.

    SSE format:
        data: {"content": "word", "done": false}

        data: {"content": "", "done": true}
    """
    import json
    data = json.dumps({
        "content": chunk.content,
        "done": chunk.done,
        "token_count": chunk.token_count,
    })
    return f"data: {data}\n\n"


async def stream_to_console(prompt: str) -> None:
    """demo streaming to console"""
    streamer = MockLLMStreamer(tokens_per_second=10)

    print("Streaming: ", end="", flush=True)

    total_tokens = 0
    async for chunk in streamer.stream(prompt):
        print(chunk.content, end="", flush=True)
        total_tokens += chunk.token_count

    print(f"\n\n[Total tokens: {total_tokens}]")


async def demo_sse_format(prompt: str) -> None:
    """demo SSE format output"""
    streamer = MockLLMStreamer(tokens_per_second=10)

    print("SSE Format:\n")

    async for chunk in streamer.stream(prompt):
        sse = format_sse(chunk)
        print(sse, end="")


# region Demo Functions

def demo_fastapi_streaming() -> None:
    """show FastAPI streaming endpoint pattern"""
    from inspect import cleandoc

    code = cleandoc('''
        # FastAPI streaming endpoint pattern

        from fastapi import FastAPI
        from fastapi.responses import StreamingResponse

        app = FastAPI()

        @app.post("/chat/stream")
        async def stream_chat(request: ChatRequest) -> StreamingResponse:
            """stream LLM response as SSE"""

            async def generate():
                async for chunk in llm.stream(request.message):
                    yield format_sse(StreamChunk(
                        content=chunk,
                        done=False,
                    ))
                # final chunk
                yield format_sse(StreamChunk(content="", done=True))

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        # Client-side (JavaScript):
        const eventSource = new EventSource("/chat/stream");
        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.done) {
                eventSource.close();
            } else {
                appendToUI(data.content);
            }
        };
    ''')
    print(code)


async def main() -> None:
    """run streaming demos"""
    print("=" * 60)
    print("  LLM Streaming Demo")
    print("=" * 60)

    prompt = "Explain why streaming matters for LLM APIs"

    print("\n1. Console streaming (simulated token-by-token):\n")
    await stream_to_console(prompt)

    print("\n" + "-" * 60)
    print("\n2. SSE format (what the API returns):\n")
    await demo_sse_format("Short demo")

    print("-" * 60)
    print("\n3. FastAPI endpoint pattern:\n")
    demo_fastapi_streaming()

    print("=" * 60)
    print("  Key insight: Streaming shows progress during slow LLM calls")
    print("=" * 60)

# endregion


if __name__ == "__main__":
    asyncio.run(main())

"""
ChatEngine - Orchestrates chat with memory and generation

Similar to a Service class in Java that coordinates components.
This is the main class you'd use in your application.
"""

import os
from typing import Optional

from chat_memory import ChatMemory
from streaming import stream_openai, stream_anthropic, stream_to_console


class ChatEngine:
    """
    complete chat engine combining memory and generation

    Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                      ChatEngine                             │
    │                                                             │
    │  ┌───────────────┐    ┌───────────────┐    ┌─────────────┐  │
    │  │  ChatMemory   │───►│   Generate    │───►│   Stream    │  │
    │  │   (history)   │    │   (LLM API)   │    │  (output)   │  │
    │  └───────────────┘    └───────────────┘    └─────────────┘  │
    │                                                             │
    │  chat() flow:                                               │
    │  1. Add user message to memory                              │
    │  2. Get messages for API                                    │
    │  3. Call LLM (streaming or not)                             │
    │  4. Add assistant response to memory                        │
    │  5. Return response                                         │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        system_prompt: str = "You are a helpful assistant.",
        provider: str = "openai",  # "openai" or "anthropic"
        model: Optional[str] = None,
        memory_strategy: str = "sliding_window",
        max_messages: int = 20,
        streaming: bool = True,
    ):
        self.provider = provider
        self.model = model or ("gpt-4o-mini" if provider == "openai" else "claude-sonnet-4-20250514")
        self.streaming = streaming

        self.memory = ChatMemory(
            system_prompt=system_prompt,
            strategy=memory_strategy,
            max_messages=max_messages,
        )

    def chat(self, user_message: str) -> str:
        """
        send message and get response

        Flow:
        ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
        │   Add    │───►│   Get    │───►│  Call    │───►│   Add    │
        │  to mem  │    │ messages │    │   LLM    │    │ response │
        └──────────┘    └──────────┘    └──────────┘    └──────────┘
        """
        # add user message to memory
        self.memory.add_user_message(user_message)

        # get messages for API
        messages = self.memory.get_messages()

        # generate response
        if self.streaming:
            response = self._generate_streaming(messages)
        else:
            response = self._generate_sync(messages)

        # add assistant response to memory
        self.memory.add_assistant_message(response)

        return response

    def _generate_streaming(self, messages: list[dict]) -> str:
        """generate with streaming output"""
        if self.provider == "openai":
            stream = stream_openai(messages, model=self.model)
        else:
            stream = stream_anthropic(
                messages,
                model=self.model,
                system=self.memory.system_prompt,
            )

        return stream_to_console(stream)

    def _generate_sync(self, messages: list[dict]) -> str:
        """generate without streaming (wait for complete response)"""
        if self.provider == "openai" and os.environ.get("OPENAI_API_KEY"):
            from openai import OpenAI
            client = OpenAI()
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=500,
            )
            return response.choices[0].message.content

        elif self.provider == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
            from anthropic import Anthropic
            client = Anthropic()
            api_messages = [msg for msg in messages if msg["role"] != "system"]
            response = client.messages.create(
                model=self.model,
                system=self.memory.system_prompt,
                messages=api_messages,
                max_tokens=500,
            )
            return response.content[0].text

        else:
            return "[No API key configured - would generate response here]"

    def reset(self) -> None:
        """clear conversation history"""
        self.memory.clear()

    def get_history(self) -> list[dict]:
        """get conversation history"""
        return self.memory.get_messages()

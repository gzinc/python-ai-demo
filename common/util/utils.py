"""
Shared utilities for Phase 7 framework examples

Provides common decorators and helpers for API key checking and demo management.
"""

import os
from functools import wraps
from typing import Callable

from dotenv import load_dotenv

# load environment variables
load_dotenv()


def check_api_keys() -> tuple[bool, bool]:
    """check which API keys are available"""
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    return bool(openai_key), bool(anthropic_key)


def requires_openai(func: Callable) -> Callable:
    """decorator to skip demo if OPENAI_API_KEY is missing"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        has_openai, _ = check_api_keys()
        if not has_openai:
            print("⚠️  OPENAI_API_KEY not found - skipping demo")
            return
        return func(*args, **kwargs)
    return wrapper


def requires_anthropic(func: Callable) -> Callable:
    """decorator to skip demo if ANTHROPIC_API_KEY is missing"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        _, has_anthropic = check_api_keys()
        if not has_anthropic:
            print("⚠️  ANTHROPIC_API_KEY not found - skipping demo")
            return
        return func(*args, **kwargs)
    return wrapper


def requires_both_keys(func: Callable) -> Callable:
    """decorator to skip demo if both API keys are missing"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        has_openai, has_anthropic = check_api_keys()
        if not (has_openai and has_anthropic):
            print("⚠️  Both OPENAI_API_KEY and ANTHROPIC_API_KEY needed")
            print("Set both keys in .env to run this demo")
            return
        return func(*args, **kwargs)
    return wrapper


def print_section(title: str) -> None:
    """print section header"""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)
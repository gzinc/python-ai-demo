"""tests for demo menu system"""

import pytest
from common.demo_menu import Demo, MenuRunner


def sample_func():
    """test demo function"""
    print("test")


def test_demo_creation():
    """verify demo dataclass initialization"""
    demo = Demo("1", "Test", "description", sample_func)
    assert demo.id == "1"
    assert demo.name == "Test"
    assert not demo.needs_api


def test_demo_display_name_with_api():
    """verify API marker in display name"""
    demo = Demo("1", "Test", "desc", sample_func, needs_api=True)
    assert demo.display_name() == "🔑 [1] Test"
    assert demo.display_name(show_marker=False) == "[1] Test"


def test_demo_display_desc_warning():
    """verify API warning in description"""
    demo = Demo("1", "Test", "desc", sample_func, needs_api=True)
    assert "⚠️ (needs API key)" in demo.display_desc(has_api=False)
    assert "⚠️" not in demo.display_desc(has_api=True)


def test_menu_runner_creates_lookup():
    """verify menu runner creates demo_map"""
    demos = [
        Demo("1", "First", "desc", sample_func),
        Demo("2", "Second", "desc", sample_func),
    ]
    runner = MenuRunner(demos)
    assert "1" in runner.demo_map
    assert runner.demo_map["1"].name == "First"


def test_run_selected_demos_all():
    """verify running all demos"""
    counter = {"count": 0}

    def increment():
        counter["count"] += 1

    demos = [Demo(str(i), f"Demo{i}", "desc", increment) for i in range(3)]
    runner = MenuRunner(demos)
    runner.run_selected_demos("a")
    assert counter["count"] == 3


def test_run_selected_demos_specific():
    """verify running specific demos"""
    executed = []

    def make_func(name):
        def func():
            executed.append(name)
        return func

    demos = [Demo(str(i), f"Demo{i}", "desc", make_func(i)) for i in range(1, 4)]
    runner = MenuRunner(demos)
    runner.run_selected_demos("1,3")
    assert executed == [1, 3]


def test_api_key_filtering():
    """verify demos requiring API are skipped when has_api=False"""
    executed = []

    def track(name):
        executed.append(name)

    demos = [
        Demo("1", "No API", "desc", lambda: track("1"), needs_api=False),
        Demo("2", "Needs API", "desc", lambda: track("2"), needs_api=True),
    ]

    runner = MenuRunner(demos, has_api=False)
    runner.run_selected_demos("a")
    assert executed == ["1"]  # only non-API demo runs


def test_quit_returns_false():
    """verify 'q' selection terminates loop"""
    demos = [Demo("1", "Test", "desc", sample_func)]
    runner = MenuRunner(demos)
    assert runner.run_selected_demos("q") is False
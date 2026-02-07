"""
Demo menu system for interactive examples across all phases.

Eliminates duplication by using Demo dataclass as single source of truth.

Usage:
    from demo_menu import Demo, MenuRunner

    DEMOS = [
        Demo("1", "Example", "description", example_func),
        Demo("2", "Another", "description", another_func, needs_api=True),
    ]

    runner = MenuRunner(DEMOS, title="My Examples")
    runner.run()
"""

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class Demo:
    """single source of truth for demo metadata and function"""
    id: str
    name: str
    description: str
    func: Callable[[], None]
    needs_api: bool = False

    def display_name(self, show_marker: bool = True) -> str:
        """format display name with optional API marker"""
        marker = "🔑 " if show_marker and self.needs_api else ""
        return f"{marker}[{self.id}] {self.name}"

    def display_desc(self, has_api: bool = True) -> str:
        """format description with optional warning"""
        warning = " ⚠️ (needs API key)" if self.needs_api and not has_api else ""
        return f"{self.description}{warning}"


class MenuRunner:
    """handles menu display and demo execution"""

    def __init__(
        self,
        demos: list[Demo],
        title: str = "Demo Menu",
        has_api: bool = True,
        subtitle: Optional[str] = None,
    ):
        self.demos = demos
        self.title = title
        self.subtitle = subtitle
        self.has_api = has_api
        # create lookup map for fast access
        self.demo_map = {demo.id: demo for demo in demos}

    def show_menu(self) -> None:
        """display interactive menu"""
        print("\n" + "=" * 70)
        print(f"  {self.title}")
        if self.subtitle:
            print(f"  {self.subtitle}")
        print("=" * 70)
        print("\n📚 Available Demos:\n")

        for demo in self.demos:
            marker = "🔑" if demo.needs_api else "  "
            status = demo.display_desc(self.has_api)
            print(f"  {marker} [{demo.id}] {demo.name}")
            print(f"      {status}")
            print()

        print("  [a] Run all demos")
        print("  [q] Quit")
        print("\n" + "=" * 70)

        if not self.has_api and any(d.needs_api for d in self.demos):
            print("  ⚠️  Some demos require API keys (marked with 🔑)")
            print("=" * 70)

    def run_selected_demos(self, selections: str) -> bool:
        """run selected demos based on user input"""
        selections = selections.lower().strip()

        if selections == 'q':
            return False

        if selections == 'a':
            # run all demos
            demos_to_run = [d for d in self.demos if self.has_api or not d.needs_api]
        else:
            # parse comma-separated selections
            selected_ids = [s.strip() for s in selections.split(',')]
            demos_to_run = [
                self.demo_map[sid]
                for sid in selected_ids
                if sid in self.demo_map and (self.has_api or not self.demo_map[sid].needs_api)
            ]

        if not demos_to_run:
            print("⚠️  Invalid selection or missing API key")
            return True

        # run each demo
        for demo in demos_to_run:
            try:
                demo.func()
            except Exception as e:
                print(f"\n❌ Error in {demo.name}: {e}")

        return True

    def run(self) -> None:
        """main interactive loop"""
        while True:
            self.show_menu()
            try:
                selection = input("\nSelect demos (comma-separated) or 'a' for all: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n👋 Goodbye!")
                break

            if not selection:
                continue

            if not self.run_selected_demos(selection):
                print("\n👋 Goodbye!")
                break

            # pause before showing menu again
            try:
                input("\n⏸️  Press Enter to continue...")
            except (EOFError, KeyboardInterrupt):
                print("\n\n👋 Goodbye!")
                break
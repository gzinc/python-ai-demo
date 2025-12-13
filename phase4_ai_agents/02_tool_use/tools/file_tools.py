"""
File Tools - Tools for reading, writing, and listing files.

These are REAL tools that actually interact with the filesystem.

Security Note:
    In production, add path validation, permission checks, and sandboxing.
"""

from pathlib import Path

from schemas.tool import ToolDefinition, ToolParameter, ToolResult
from tools.base import BaseTool


class ReadFileTool(BaseTool):
    """
    Tool to read contents of a file.

    Usage by agent:
        ACTION: read_file(path="/tmp/notes.txt")
        OBSERVATION: Contents of the file...
    """

    def __init__(self, allowed_directories: list[str] | None = None):
        """
        Initialize with optional directory restrictions.

        Args:
            allowed_directories: If set, only allow reading from these dirs.
        """
        self._allowed_directories = allowed_directories

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="Read the contents of a text file. Returns the file content as a string.",
            parameters=[
                ToolParameter(
                    name="path",
                    param_type="string",
                    description="Absolute or relative path to the file to read",
                    required=True,
                ),
            ],
        )

    def execute(self, path: str, **kwargs) -> ToolResult:
        """Read and return file contents."""
        file_path = Path(path)

        if not file_path.exists():
            return ToolResult.fail(f"File not found: {path}")

        if not file_path.is_file():
            return ToolResult.fail(f"Not a file: {path}")

        if self._allowed_directories is not None:
            absolute_path = str(file_path.resolve())
            if not any(absolute_path.startswith(directory) for directory in self._allowed_directories):
                return ToolResult.fail(f"Access denied: {path} is not in allowed directories")

        try:
            content = file_path.read_text(encoding="utf-8")
            return ToolResult.ok(content)
        except PermissionError:
            return ToolResult.fail(f"Permission denied: {path}")
        except UnicodeDecodeError:
            return ToolResult.fail(f"Cannot read binary file as text: {path}")
        except Exception as e:
            return ToolResult.fail(f"Error reading file: {str(e)}")


class WriteFileTool(BaseTool):
    """
    Tool to write content to a file.

    Creates the file if it doesn't exist, overwrites if it does.

    Usage by agent:
        ACTION: write_file(path="/tmp/output.txt", content="Hello, World!")
        OBSERVATION: File written successfully: /tmp/output.txt (13 bytes)
    """

    def __init__(self, allowed_directories: list[str] | None = None):
        """Initialize with optional directory restrictions."""
        self._allowed_directories = allowed_directories

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="Write content to a text file. Creates the file if it doesn't exist, overwrites if it does.",
            parameters=[
                ToolParameter(
                    name="path",
                    param_type="string",
                    description="Path where the file should be written",
                    required=True,
                ),
                ToolParameter(
                    name="content",
                    param_type="string",
                    description="Text content to write to the file",
                    required=True,
                ),
            ],
        )

    def execute(self, path: str, content: str, **kwargs) -> ToolResult:
        """Write content to a file."""
        file_path = Path(path)

        if self._allowed_directories is not None:
            absolute_path = str(file_path.resolve())
            if not any(absolute_path.startswith(d) for d in self._allowed_directories):
                return ToolResult.fail(f"Access denied: {path} is not in allowed directories")

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            return ToolResult.fail(f"Cannot create directory for: {path}")

        try:
            file_path.write_text(content, encoding="utf-8")
            bytes_written = len(content.encode("utf-8"))
            return ToolResult.ok(f"File written successfully: {path} ({bytes_written} bytes)")
        except PermissionError:
            return ToolResult.fail(f"Permission denied: {path}")
        except Exception as e:
            return ToolResult.fail(f"Error writing file: {str(e)}")


class ListDirectoryTool(BaseTool):
    """
    Tool to list contents of a directory.

    Usage by agent:
        ACTION: list_directory(path="/tmp")
        OBSERVATION: ["file1.txt", "file2.txt", "subdir/"]
    """

    def __init__(self, allowed_directories: list[str] | None = None):
        """Initialize with optional directory restrictions."""
        self._allowed_directories = allowed_directories

    @property
    def name(self) -> str:
        return "list_directory"

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="List files and subdirectories in a directory. Returns a list of names.",
            parameters=[
                ToolParameter(
                    name="path",
                    param_type="string",
                    description="Path to the directory to list",
                    required=True,
                ),
            ],
        )

    def execute(self, path: str, **kwargs) -> ToolResult:
        """List directory contents."""
        dir_path = Path(path)

        if not dir_path.exists():
            return ToolResult.fail(f"Directory not found: {path}")

        if not dir_path.is_dir():
            return ToolResult.fail(f"Not a directory: {path}")

        if self._allowed_directories is not None:
            absolute_path = str(dir_path.resolve())
            if not any(absolute_path.startswith(d) for d in self._allowed_directories):
                return ToolResult.fail(f"Access denied: {path} is not in allowed directories")

        try:
            items = []
            for item in sorted(dir_path.iterdir()):
                name = item.name + "/" if item.is_dir() else item.name
                items.append(name)
            return ToolResult.ok(items)
        except PermissionError:
            return ToolResult.fail(f"Permission denied: {path}")
        except Exception as e:
            return ToolResult.fail(f"Error listing directory: {str(e)}")

"""A bounded virtual workspace. No host file access or command execution."""
import json
from pathlib import PurePosixPath

class ToolError(ValueError):
    pass

def safe_path(path):
    if not isinstance(path, str) or not path or "\\" in path or ":" in path:
        raise ToolError("invalid workspace path")
    p = PurePosixPath(path)
    if p.is_absolute() or ".." in p.parts:
        raise ToolError("path escapes workspace")
    return str(p)

class Workspace:
    def __init__(self, files):
        self.files = {safe_path(k): v for k, v in files.items()}

    def call(self, action):
        if not isinstance(action, dict) or set(action) != {"tool", "arguments"}:
            raise ToolError("expected tool and arguments")
        name, args = action["tool"], action["arguments"]
        if not isinstance(args, dict):
            raise ToolError("arguments must be object")
        if name == "finish":
            if args:
                raise ToolError("finish takes no arguments")
            return "finished"
        schemas = {"read": {"path"}, "search": {"query"}, "edit": {"path", "old", "new"}}
        if name not in schemas or set(args) != schemas[name]:
            raise ToolError("unknown tool or incorrect arguments")
        if not all(isinstance(v, str) for v in args.values()):
            raise ToolError("arguments must be strings")
        if name == "search":
            if not args["query"]:
                raise ToolError("search query must be nonempty")
            return json.dumps([{ "path": p, "line": i, "text": line}
                for p, data in sorted(self.files.items()) for i, line in enumerate(data.splitlines(), 1)
                if args["query"] in line])
        path = safe_path(args["path"])
        if path not in self.files:
            raise ToolError("file not found")
        if name == "read":
            return self.files[path]
        old = args["old"]
        if not old or self.files[path].count(old) != 1:
            raise ToolError("old text must match exactly once")
        self.files[path] = self.files[path].replace(old, args["new"], 1)
        return "edited"

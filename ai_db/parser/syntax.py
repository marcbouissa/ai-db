import ast
from typing import Optional, Tuple

def validate_python_syntax(content: str, filepath: str) -> Optional[Tuple[int, int, str]]:
    """Validates Python syntax using ast.parse. Returns (line, col, msg) if error, else None."""
    try:
        ast.parse(content, filename=filepath)
        return None
    except SyntaxError as e:
        line = e.lineno or 1
        col = e.offset or 1
        msg = e.msg or "Syntax error"
        return (line, col, msg)
    except Exception as e:
        return (1, 1, str(e))

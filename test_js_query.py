from tree_sitter import Query
from tree_sitter_language_pack import get_language

# Test minimal export
content1 = "(export_statement\n  (function_declaration\n    name: (identifier) @name) @def.function\n)"

try:
    q = Query(get_language("javascript"), content1)
    print("Export function: OK")
except Exception as e:
    print(f"Error: {e}")

# Test full file
with open("ai_db/parser/queries/javascript.scm", "rb") as f:
    content = f.read().decode("utf-8")

try:
    q = Query(get_language("javascript"), content)
    print("Full file OK")
except Exception as e:
    print(f"Full file error: {e}")
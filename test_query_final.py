from tree_sitter import Query
from tree_sitter_language_pack import get_language

# Test the exact content from the file
with open("ai_db/parser/queries/javascript.scm", "rb") as f:
    content = f.read().decode("utf-8")

from tree_sitter import Query
from tree_sitter_language_pack import get_language

try:
    q = Query(get_language("javascript"), content)
    print("Full query OK")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
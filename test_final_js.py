from tree_sitter import Query
from tree_sitter_language_pack import get_language

lang = get_language('javascript')

# Read the file content
with open('ai_db/parser/queries/javascript.scm', 'r', encoding='utf-8') as f:
    content = f.read()

from tree_sitter import Query

try:
    q = Query(lang, content)
    print('Full query OK')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
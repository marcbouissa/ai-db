from tree_sitter import Query
from tree_sitter_language_pack import get_language

lang = get_language('javascript')

content = """(import_statement
  (import_clause
    (named_imports
      (import_specifier
        name: (identifier) @name))*) @ref.import
)"""

from tree_sitter import Query

try:
    q = Query(lang, content)
    print('Import OK')
except Exception as e:
    print(f'Error: {e}')
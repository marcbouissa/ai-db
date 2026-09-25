from tree_sitter import Query
from tree_sitter_language_pack import get_language

# Test with export statements
content = """(export_statement
  (function_declaration
    name: (identifier) @name) @def.function
)

(export_statement
  (class_declaration) @def.class
)

(export_statement
  (lexical_declaration) @ref.export
)

(export_statement
  (variable_declaration) @ref.export
)

(import_statement
  (import_clause
    (named_imports
      (import_specifier
        name: (identifier) @name))*) @ref.import

(call_expression
  function: [
    (identifier) @callee
    (member_expression
      property: (property_identifier) @callee)
  ] @ref.call)

(ERROR) @syntax.error
(MISSING) @syntax.error
"""

from tree_sitter import Query
from tree_sitter_language_pack import get_language

try:
    q = Query(get_language('javascript'), content)
    print('Full query OK')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
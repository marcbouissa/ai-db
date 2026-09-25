from tree_sitter import Query
from tree_sitter_language_pack import get_language

lang = get_language('javascript')

patterns = [
    '''(function_declaration
  name: (identifier) @name
  parameters: (formal_parameters) @params
  body: (statement_block) @body) @def.function''',

    '''(variable_declarator
  name: (identifier) @name
  value: [
    (function_expression
      parameters: (formal_parameters) @params
      body: (statement_block) @body)
    (arrow_function
      parameters: (formal_parameters) @params
      body: [
        (statement_block) @body
        (expression) @body
      ])
  ]) @def.function''',

    '''(method_definition
  name: (property_identifier) @name
  parameters: (formal_parameters) @params
  body: (statement_block) @body) @def.method''',

    '''(class_declaration
  name: (identifier) @name
  (class_heritage
    (identifier) @base)? @def.class)''',

    '''(export_statement
  (function_declaration
    name: (identifier) @name) @def.function
)
''',
    '''(export_statement
  (class_declaration) @def.class
)
''',
    '''(export_statement
  (lexical_declaration) @ref.export
)
''',
    '''(export_statement
  (variable_declaration) @ref.export
)
''',
    '''(import_statement
  (import_clause
    (named_imports
      (import_specifier
        name: (identifier) @name))*) @ref.import
)''',
    '''(call_expression
  function: [
    (identifier) @callee
    (member_expression
      property: (property_identifier) @callee)
  ] @ref.call)
''',
    '''(ERROR) @syntax.error
(MISSING) @syntax.error''',
]

from tree_sitter import Query

for i, part in enumerate(patterns):
    part = part.strip()
    if part and not part.startswith(';'):
        try:
            q = Query(lang, part)
            print(f'Part {i}: OK')
        except Exception as e:
            print(f'Part {i} ERROR: {e}')
            print(f'  Content: {part[:100]}')
            break
else:
    print('All parts OK')
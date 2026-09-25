from tree_sitter import Query
from tree_sitter_language_pack import get_language

lang = get_language('javascript')

content = """; JavaScript tree-sitter queries for symbols, cross-refs, and syntax errors

; Function declarations
(function_declaration
  name: (identifier) @name
  parameters: (formal_parameters) @params
  body: (statement_block) @body) @def.function

; Function expressions (assigned to variables)
(variable_declarator
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
  ]) @def.function

; Method definitions in classes
(method_definition
  name: (property_identifier) @name
  parameters: (formal_parameters) @params
  body: (statement_block) @body) @def.method

; Class declarations
(class_declaration
  name: (identifier) @name
  (class_heritage
    (identifier) @base)? @def.class)

; Export statements
(export_statement
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

; Import statements
(import_statement
  (import_clause
    (named_imports
      (import_specifier
        name: (identifier) @name))*) @ref.import

; Call expressions
(call_expression
  function: [
    (identifier) @callee
    (member_expression
      property: (property_identifier) @callee)
  ] @ref.call)

; Syntax errors
(ERROR) @syntax.error
(MISSING) @syntax.error
"""

from tree_sitter import Query
from tree_sitter_language_pack import get_language

lang = get_language('javascript')

try:
    q = Query(lang, content)
    print('Full query OK')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
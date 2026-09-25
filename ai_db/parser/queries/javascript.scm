; JavaScript tree-sitter queries for symbols, cross-refs, and syntax errors

; Function declarations
(function_declaration
  name: (identifier) @name
  parameters: (formal_parameters) @params
  body: (statement_block) @body) @def.function

; Method definitions in classes
(method_definition
  name: (property_identifier) @name
  parameters: (formal_parameters) @params
  body: (statement_block) @body) @def.method

; Class declarations
(class_declaration
  name: (identifier) @name
  (class_heritage
    (identifier) @base)?) @def.class @ref.inherit

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

; Import statements (ES modules)
(import_statement
  source: (string) @module) @ref.import

; Require calls (CommonJS imports) - capture as import
(call_expression
  function: (identifier) @callee
  (#eq? @callee "require")
  arguments: (arguments
    (string) @module)) @ref.import

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
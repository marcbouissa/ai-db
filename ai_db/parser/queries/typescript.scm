; TypeScript tree-sitter queries for symbols, cross-refs, and syntax errors

; Function declarations with type annotations
(function_declaration
  name: (identifier) @name
  parameters: (formal_parameters) @params
  return_type: (type_annotation)? @return_type
  body: (statement_block) @body) @def.function

; Function expressions with types
(variable_declarator
  name: (identifier) @name
  value: [
    (function_expression
      parameters: (formal_parameters) @params
      return_type: (type_annotation)? @return_type
      body: (statement_block) @body)
    (arrow_function
      parameters: (formal_parameters) @params
      return_type: (type_annotation)? @return_type
      body: [
        (statement_block) @body
        (expression) @body
      ])
  ]) @def.function

; Method definitions in classes
(method_definition
  name: (property_identifier) @name
  parameters: (formal_parameters) @params
  return_type: (type_annotation)? @return_type
  body: (statement_block) @body) @def.method

; Class declarations
(class_declaration
  name: (type_identifier) @name
  (class_heritage
    (extends_clause
      (identifier) @base)
    (implements_clause
      (type_identifier) @impl)*)? @def.class)

; Interface declarations
(interface_declaration
  name: (type_identifier) @name) @def.interface

; Type alias declarations
(type_alias_declaration
  name: (type_identifier) @name) @def.type

; Enum declarations
(enum_declaration
  name: (identifier) @name) @def.type

; Export statements - separate patterns for each declaration type
(export_statement
  (class_declaration) @def.class)
(export_statement
  (interface_declaration) @def.interface)
(export_statement
  (type_alias_declaration) @def.type)
(export_statement
  (enum_declaration) @def.type)
(export_statement
  (lexical_declaration) @ref.export)

; Import statements
(import_statement
  (import_clause
    (named_imports
      (import_specifier
        name: (identifier) @name)))) @ref.import

; Call expressions
(call_expression
  function: [
    (identifier) @callee
    (member_expression
      property: (property_identifier) @callee)
  ] @ref.call)

; New expressions
(new_expression
  constructor: [
    (identifier) @callee
    (member_expression
      property: (property_identifier) @callee)
  ] @ref.call)

; Syntax errors
(ERROR) @syntax.error
(MISSING) @syntax.error
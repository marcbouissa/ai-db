; C tree-sitter queries for symbols, cross-refs, and syntax errors

; Function definitions
(function_definition
  declarator: (function_declarator
    declarator: (identifier) @name
    parameters: (parameter_list) @params)
  body: (compound_statement) @body) @def.function

; Function declarations (without body) - not present in C tree-sitter, using function_definition only

; Struct definitions
(struct_specifier
  name: (type_identifier) @name
  body: (field_declaration_list) @body) @def.struct

; Union definitions
(union_specifier
  name: (type_identifier) @name
  body: (field_declaration_list) @body) @def.type

; Enum definitions
(enum_specifier
  name: (type_identifier) @name
  body: (enumerator_list) @body) @def.type

; Typedef definitions
(type_definition
  declarator: (init_declarator
    declarator: (identifier) @name)) @def.type

; Function calls
(call_expression
  function: [
    (identifier) @callee
    (field_expression
      field: (field_identifier) @callee)
  ] @ref.call)

; Macro calls
(call_expression
  function: (identifier) @callee
  (#match? @callee "^[A-Z_][A-Z0-9_]*$")) @ref.call

; Include statements
(preproc_include
  path: (string_literal) @module) @ref.import

; Syntax errors
(ERROR) @syntax.error
(MISSING) @syntax.error
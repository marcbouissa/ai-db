; Go tree-sitter queries for symbols, cross-refs, and syntax errors

; Function declarations
(function_declaration
  name: (identifier) @name
  parameters: (parameter_list) @params
  result: (parameter_list)? @results
  body: (block) @body) @def.function

; Method declarations (with receiver)
(method_declaration
  name: (field_identifier) @name
  parameters: (parameter_list) @params
  body: (block) @body) @def.method

; Type declarations (structs, interfaces, type aliases)
(type_declaration
  (type_spec
    name: (type_identifier) @name
    type: [
      (struct_type) @def.struct
      (interface_type) @def.interface
      (type_identifier) @def.type
    ])) @def.type

; Function calls
(call_expression
  function: [
    (identifier) @callee
    (selector_expression
      field: (field_identifier) @callee)
  ] @ref.call)

; Method calls
(selector_expression
  field: (field_identifier) @callee) @ref.call

; Imports - using interpreted_string_literal for import path
(import_spec
  (interpreted_string_literal) @module) @ref.import

; Type references in variable declarations
(var_declaration
  (var_spec
    name: (identifier) @name
    type: (type_identifier) @type) @ref.type)

; Type assertions and conversions
(type_assertion_expression
  type: (type_identifier) @callee) @ref.call

; Syntax errors
(ERROR) @syntax.error
(MISSING) @syntax.error
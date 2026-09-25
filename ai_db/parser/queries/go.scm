; Go tree-sitter queries for symbols, cross-refs, and syntax errors

; Function declarations
(function_declaration
  name: (identifier) @name
  parameters: (parameter_list) @params
  result: (_)? @results
  body: (block) @body) @def.function

; Method declarations (receiver is the first parameter)
(method_declaration
  name: (field_identifier) @name
  parameters: (parameter_list) @params
  result: (_)? @results
  body: (block) @body) @def.method

; Type declarations: the name lives on type_spec, not on type_declaration
(type_declaration
  (type_spec
    name: (type_identifier) @name
    type: (struct_type))) @def.struct

(type_declaration
  (type_spec
    name: (type_identifier) @name
    type: (interface_type))) @def.interface

; type Alias = Other
(type_declaration
  (type_spec
    name: (type_identifier) @name
    type: (_) @alias)) @def.type

; Imports - the path is an interpreted_string_literal on import_spec
(import_spec
  path: (interpreted_string_literal) @module) @ref.import

; Calls to a plain function
(call_expression
  function: (identifier) @callee) @ref.call

; Calls to a package-qualified function (fmt.Println)
(call_expression
  function: (selector_expression
    operand: (identifier)
    field: (field_identifier) @callee)) @ref.call

; Method calls (obj.Method())
(selector_expression
  operand: (_)
  field: (field_identifier) @callee) @ref.call

; Type conversions and assertions
(type_assertion_expression
  type: (type_identifier) @callee) @ref.call

; Syntax errors
(ERROR) @syntax.error
(MISSING) @syntax.error

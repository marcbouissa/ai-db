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

; Named non-struct, non-interface types (`type ID string`, `type H func()`); struct and
; interface types are emitted once, by the patterns above
(type_declaration
  (type_spec
    name: (type_identifier) @name
    type: [
      (type_identifier)
      (qualified_type)
      (pointer_type)
      (slice_type)
      (array_type)
      (map_type)
      (channel_type)
      (function_type)
      (generic_type)
    ])) @def.type

; type Alias = Other
(type_declaration
  (type_alias
    name: (type_identifier) @name)) @def.type

; Imports - the path is an interpreted_string_literal on import_spec
(import_spec
  path: (interpreted_string_literal) @module) @ref.import

; Calls to a plain function
(call_expression
  function: (identifier) @callee) @ref.call

; Package-qualified and method calls (fmt.Println(), obj.Method()). A selector alone
; (obj.Field) is a field read, not a call.
(call_expression
  function: (selector_expression
    field: (field_identifier) @callee)) @ref.call

; Type conversions and assertions
(type_assertion_expression
  type: (type_identifier) @callee) @ref.call

; Syntax errors
(ERROR) @syntax.error
(MISSING) @syntax.error

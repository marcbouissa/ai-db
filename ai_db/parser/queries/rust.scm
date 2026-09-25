; Rust tree-sitter queries for symbols, cross-refs, and syntax errors

; Function definitions (includes async functions)
(function_item
  name: (identifier) @name
  parameters: (parameters) @params
  return_type: (_)? @return_type
  body: (block) @body) @def.function

; Method definitions in impl blocks
(impl_item
  (function_item
    name: (identifier) @name
    parameters: (parameters) @params
    return_type: (_)? @return_type
    body: (block) @body)) @def.method

; Struct definitions
(struct_item
  name: (type_identifier) @name
  body: [
    (struct_body) @body
    (tuple_struct_body) @body
  ]) @def.struct

; Enum definitions
(enum_item
  name: (type_identifier) @name
  body: (enum_body) @body) @def.type

; Trait definitions
(trait_item
  name: (type_identifier) @name
  body: (trait_body) @body) @def.interface

; Impl blocks (for trait implementations)
(impl_item
  (type_identifier) @base
  (for
    (type_identifier) @trait)?) @ref.inherit

; Type aliases
(type_item
  name: (type_identifier) @name) @def.type

; Function calls
(call_expression
  function: [
    (identifier) @callee
    (scoped_identifier
      path: (identifier) @mod
      name: (identifier) @callee)
    (field_expression
      field: (field_identifier) @callee)
  ] @ref.call)

; Method calls
(field_expression
  field: (field_identifier) @callee) @ref.call

; Macro calls
(macro_invocation
  macro: (identifier) @callee) @ref.call

; Use statements (imports)
(use_declaration
  (use_tree
    (scoped_identifier
      path: (identifier) @mod
      name: (identifier) @name) @ref.import))

; External crate declarations
(extern_crate_declaration
  name: (identifier) @name) @ref.import

; Syntax errors
(ERROR) @syntax.error
(MISSING) @syntax.error
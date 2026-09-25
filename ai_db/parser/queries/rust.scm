; Rust tree-sitter queries for symbols, cross-refs, and syntax errors

; Function definitions (includes async functions)
(function_item
  name: (identifier) @name
  parameters: (parameters) @params
  return_type: (_)? @return_type
  body: (block) @body) @def.function

; Method definitions in impl blocks (function_item inside impl_item)
(function_item
  name: (identifier) @name
  parameters: (parameters) @params
  return_type: (_)? @return_type
  body: (block) @body) @def.method

; Struct definitions
(struct_item
  name: (type_identifier) @name
  body: (field_declaration_list) @body) @def.struct

; Tuple struct definitions
(struct_item
  name: (type_identifier) @name
  body: (ordered_field_declaration_list) @body) @def.struct

; Unit struct definitions (no body)
(struct_item
  name: (type_identifier) @name) @def.struct

; Enum definitions
(enum_item
  name: (type_identifier) @name
  body: (enum_variant_list) @body) @def.type

; Trait definitions
(trait_item
  name: (type_identifier) @name
  body: (declaration_list) @body) @def.interface

; Impl blocks (for trait implementations) - first type_identifier is the trait
(impl_item
  (type_identifier) @base) @ref.inherit

; Type aliases
(type_item
  name: (type_identifier) @name) @def.type

; Function calls - simple identifier
(call_expression
  function: (identifier) @callee) @ref.call

; Function calls - scoped identifier (mod::func)
(call_expression
  function: (scoped_identifier
    path: (identifier) @mod
    name: (identifier) @callee)) @ref.call

; Method calls - field_expression
(call_expression
  function: (field_expression
    field: (field_identifier) @callee)) @ref.call

; Method calls (standalone field_expression)
(field_expression
  field: (field_identifier) @callee) @ref.call

; Macro calls
(macro_invocation
  macro: (identifier) @callee) @ref.call

; Use statements (imports)
(use_declaration
  (scoped_identifier
    path: (identifier) @mod
    name: (identifier) @name) @ref.import)

; External crate declarations
(extern_crate_declaration
  name: (identifier) @name) @ref.import

; Syntax errors
(ERROR) @syntax.error
(MISSING) @syntax.error
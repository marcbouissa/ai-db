; Rust tree-sitter queries for symbols, cross-refs, and syntax errors

; Function definitions
(function_item
  name: (identifier) @name
  parameters: (parameters) @params
  return_type: (_)? @return_type
  body: (block) @body) @def.function

; Free functions and methods share the function_item node; the chunker's
; qualified-name rules decide the prefix, so both are @def.function here.

; Struct definitions (named-field body)
(struct_item
  name: (type_identifier) @name
  body: (field_declaration_list) @body) @def.struct

; Tuple struct definitions
(struct_item
  name: (type_identifier) @name
  body: (ordered_field_declaration_list) @body) @def.struct

; Unit struct definitions (name only)
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

; `impl Trait for Type` - grammar fields are `trait:` (the trait) and
; `type:` (the self type), in that source order.
(impl_item
  trait: (type_identifier) @base
  type: (type_identifier)? @impl) @ref.inherit

; Type aliases
(type_item
  name: (type_identifier) @name) @def.type

; Trait method signatures
(function_signature_item
  name: (identifier) @name
  parameters: (parameters) @params) @def.method

; Calls: plain function
(call_expression
  function: (identifier) @callee) @ref.call

; Calls: path::function (the last identifier is the function)
(call_expression
  function: (scoped_identifier
    path: (_) @mod
    name: (identifier) @callee)) @ref.call

; Method calls: receiver.method()
(call_expression
  function: (field_expression
    field: (field_identifier) @callee)) @ref.call

; Macro invocation
(macro_invocation
  macro: (identifier) @callee) @ref.call

; Use statements (imports)
(use_declaration
  argument: (scoped_identifier
    path: (_) @mod
    name: (identifier) @name)) @ref.import

; Syntax errors
(ERROR) @syntax.error
(MISSING) @syntax.error

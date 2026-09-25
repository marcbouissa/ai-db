; C++ tree-sitter queries for symbols, cross-refs, and syntax errors

; Function definitions
(function_definition
  declarator: (function_declarator
    declarator: (identifier) @name
    parameters: (parameter_list) @params)
  body: (compound_statement) @body) @def.function

; Function declarations
(declaration
  (function_declarator
    declarator: (identifier) @name
    parameters: (parameter_list) @params)) @def.function

; Method definitions in classes (qualified identifier)
(function_definition
  declarator: (function_declarator
    declarator: (qualified_identifier
      name: (identifier) @name)
    parameters: (parameter_list) @params)
  body: (compound_statement) @body) @def.method

; Class definitions with base classes
(class_specifier
  name: (type_identifier) @name
  body: (field_declaration_list) @body
  (base_class_clause
    (type_identifier) @base)?) @def.class @ref.inherit

; Struct definitions
(struct_specifier
  name: (type_identifier) @name
  body: (field_declaration_list) @body) @def.struct

; Enum definitions
(enum_specifier
  name: (type_identifier) @name
  body: (enumerator_list) @body) @def.type

; Namespace definitions
(namespace_definition
  name: (identifier) @name
  body: (declaration_list) @body) @def.type

; Template declarations
(template_declaration
  (function_definition
    declarator: (function_declarator
      declarator: (identifier) @name)) @def.function)
(template_declaration
  (class_specifier
    name: (type_identifier) @name) @def.class)

; Function calls - simple identifier
(call_expression
  function: (identifier) @callee) @ref.call

; Function calls - field_expression (method calls)
(call_expression
  function: (field_expression
    field: (field_identifier) @callee)) @ref.call

; Function calls - qualified_identifier (namespace::func)
(call_expression
  function: (qualified_identifier
    (namespace_identifier)
    (identifier) @callee)) @ref.call

; Method calls
(field_expression
  field: (field_identifier) @callee) @ref.call

; New expressions
(new_expression
  type: (type_identifier) @callee) @ref.call

; Include statements
(preproc_include
  path: (string_literal) @module) @ref.import

; Using declarations
(using_declaration
  (namespace_identifier) @module) @ref.import

; Syntax errors
(ERROR) @syntax.error
(MISSING) @syntax.error
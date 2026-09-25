; C++ tree-sitter queries for symbols, cross-refs, and syntax errors

; Function definitions
(function_definition
  declarator: (function_declarator
    declarator: [
      (identifier) @name
      (qualified_identifier
        name: (identifier) @name
        scope: (namespace_identifier) @scope)
      (operator_name) @name]
    parameters: (parameter_list) @params)
  body: (compound_statement) @body) @def.function

; Function declarations
(declaration
  (function_declarator
    declarator: [
      (identifier) @name
      (qualified_identifier
        name: (identifier) @name)
      (operator_name) @name]
    parameters: (parameter_list) @params)) @def.function

; Method definitions in classes
(function_definition
  declarator: (function_declarator
    declarator: (qualified_identifier
      name: (identifier) @name
      scope: (namespace_identifier) @scope)
    parameters: (parameter_list) @params)
  body: (compound_statement) @body) @def.method

; Class definitions
(class_specifier
  name: (type_identifier) @name
  body: (field_declaration_list) @body
  base_clause: (base_class_clause
    (base_specifier
      type: (type_identifier) @base)*)? @def.class

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

; Function calls
(call_expression
  function: [
    (identifier) @callee
    (field_expression
      field: (field_identifier) @callee)
    (qualified_identifier
      name: (identifier) @callee
      scope: (namespace_identifier) @scope)
  ] @ref.call)

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
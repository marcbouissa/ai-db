; C++ tree-sitter queries for symbols, cross-refs, and syntax errors

; Function definitions
(function_definition
  declarator: (function_declarator
    declarator: (identifier) @name
    parameters: (parameter_list) @params)
  body: (compound_statement) @body) @def.function

; Function declarations (prototypes, no body)
(declaration
  (function_declarator
    declarator: (identifier) @name
    parameters: (parameter_list) @params)) @def.function

; Method definitions in classes: Invoice::total
(function_definition
  declarator: (function_declarator
    declarator: (qualified_identifier
      name: (identifier) @name
      scope: (namespace_identifier)? @scope)
    parameters: (parameter_list) @params)
  body: (compound_statement) @body) @def.method

; Class definitions. Children must appear in SOURCE ORDER in a tree-sitter
; pattern: name, then the base clause, then the body.
(class_specifier
  name: (type_identifier) @name
  (base_class_clause
    (type_identifier) @base)?
  body: (field_declaration_list) @body) @def.class @ref.inherit

; Struct definitions
(struct_specifier
  name: (type_identifier) @name
  body: (field_declaration_list) @body) @def.struct

; Union definitions
(union_specifier
  name: (type_identifier) @name
  body: (field_declaration_list) @body) @def.struct

; Enum definitions
(enum_specifier
  name: (type_identifier) @name
  body: (enumerator_list) @body) @def.type

; Using declarations act as imports
(using_declaration
  (identifier) @module) @ref.import

; Namespace-qualified using (using std::vector)
(using_declaration
  (qualified_identifier) @module) @ref.import

; Calls: plain function
(call_expression
  function: (identifier) @callee) @ref.call

; Calls: qualified (std::sort)
(call_expression
  function: (qualified_identifier
    name: (identifier) @callee
    scope: (namespace_identifier)? @scope)) @ref.call

; Method calls: obj.method()
(call_expression
  function: (field_expression
    field: (field_identifier) @callee)) @ref.call

; new expressions
(new_expression
  type: (type_identifier) @callee) @ref.call

; Include statements
(preproc_include
  path: (string_literal) @module) @ref.import

; Syntax errors
(ERROR) @syntax.error
(MISSING) @syntax.error

; C tree-sitter queries for symbols, cross-refs, and syntax errors

; Function definitions (also returning pointers: `char *f()`, `int **g()`)
(function_definition
  declarator: (function_declarator
    declarator: (identifier) @name
    parameters: (parameter_list) @params)
  body: (compound_statement) @body) @def.function

(function_definition
  declarator: (pointer_declarator
    declarator: (function_declarator
      declarator: (identifier) @name
      parameters: (parameter_list) @params))
  body: (compound_statement) @body) @def.function

(function_definition
  declarator: (pointer_declarator
    declarator: (pointer_declarator
      declarator: (function_declarator
        declarator: (identifier) @name
        parameters: (parameter_list) @params)))
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

; Typedef definitions: `typedef int t;`, `typedef struct s *sp;`, `typedef int (*cb)(int);`
(type_definition
  declarator: (type_identifier) @name) @def.type

(type_definition
  declarator: (pointer_declarator
    declarator: (type_identifier) @name)) @def.type

(type_definition
  declarator: (function_declarator
    declarator: (parenthesized_declarator
      (pointer_declarator
        declarator: (type_identifier) @name)))) @def.type

; Function and macro calls (`f()`, `MAX()`, `obj->fn()`)
(call_expression
  function: [
    (identifier) @callee
    (field_expression
      field: (field_identifier) @callee)
  ]) @ref.call

; Include statements
(preproc_include
  path: (string_literal) @module) @ref.import

; Syntax errors
(ERROR) @syntax.error
(MISSING) @syntax.error
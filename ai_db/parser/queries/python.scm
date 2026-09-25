; Python tree-sitter queries for symbols, cross-refs, and syntax errors

; Function definitions (includes async functions - they have an 'async' child node)
(function_definition
  name: (identifier) @name
  parameters: (parameters) @params
  body: (block) @body) @def.function

; Class definitions
(class_definition
  name: (identifier) @name
  body: (block) @body
  (argument_list) @bases?) @def.class

; Method definitions inside classes
(function_definition
  name: (identifier) @name
  parameters: (parameters) @params
  body: (block) @body) @def.method

; Import statements
(import_statement
  (dotted_name) @module) @ref.import

(import_from_statement
  (dotted_name) @module
  (dotted_name) @name) @ref.import

; Call expressions - capture the call and the called function name
(call
  function: [
    (identifier) @callee
    (attribute
      attribute: (identifier) @callee)
  ]
) @ref.call

; Inheritance - class bases
(class_definition
  (argument_list
    (identifier) @base) @ref.inherit)

; Assignment to capture type aliases
(assignment
  left: (identifier) @name
  right: [
    (call
      function: (identifier) @type_ctor)
    (subscript
      value: (identifier) @type_ctor)
  ] @def.type)

; Syntax errors - ERROR and MISSING nodes
(ERROR) @syntax.error
(MISSING) @syntax.error
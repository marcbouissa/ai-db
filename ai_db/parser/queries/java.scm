; Java tree-sitter queries for symbols, cross-refs, and syntax errors

; Class declarations
(class_declaration
  name: (identifier) @name
  superclass: (superclass
    (type_identifier) @base)?
  interfaces: (super_interfaces
    (type_identifier) @impl)?
  body: (class_body) @body) @def.class

; Interface declarations
(interface_declaration
  name: (identifier) @name
  extends: (extends_interfaces
    (type_identifier) @base)?
  body: (interface_body) @body) @def.interface

; Enum declarations
(enum_declaration
  name: (identifier) @name
  implements: (super_interfaces
    (type_identifier) @impl)?
  body: (enum_body) @body) @def.type

; Record declarations (Java 14+)
(record_declaration
  name: (identifier) @name
  implements: (super_interfaces
    (type_identifier) @impl)?
  body: (record_body) @body) @def.type

; Method declarations
(method_declaration
  name: (identifier) @name
  parameters: (formal_parameters) @params
  return_type: (type_identifier)? @return_type
  body: (block)? @body) @def.method

; Constructor declarations
(constructor_declaration
  name: (identifier) @name
  parameters: (formal_parameters) @params
  body: (constructor_body) @body) @def.method

; Field declarations
(field_declaration
  (variable_declarator
    name: (identifier) @name)) @def.type

; Method calls
(method_invocation
  name: (identifier) @callee
  arguments: (argument_list) @args) @ref.call

; Qualified method calls (obj.method())
(method_invocation
  object: (expression)
  name: (identifier) @callee
  arguments: (argument_list) @args) @ref.call

; Constructor calls (new Class())
(object_creation_expression
  type: (type_identifier) @callee) @ref.call

; Static method calls (Class.method())
(method_invocation
  name: (identifier) @callee
  object: (type_identifier) @scope) @ref.call

; Import statements
(import_declaration
  name: (scoped_identifier
    path: (identifier) @mod
    name: (identifier) @name) @ref.import)

; Package declaration
(package_declaration
  name: (scoped_identifier) @module) @ref.import

; Syntax errors
(ERROR) @syntax.error
(MISSING) @syntax.error
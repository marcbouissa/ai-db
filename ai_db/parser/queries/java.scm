; Java tree-sitter queries for symbols, cross-refs, and syntax errors

; Class declarations.
; `superclass` and `interfaces` are fields; `super_interfaces` is the node type.
; Children must be listed in SOURCE ORDER: name, superclass, interfaces, body.
(class_declaration
  name: (identifier) @name
  superclass: (superclass
    (type_identifier) @base)?
  interfaces: (super_interfaces
    (type_list
      (type_identifier) @impl)*)?
  body: (class_body) @body) @def.class @ref.inherit

; Interface declarations. `extends_interfaces` is an anonymous child here
; (not a field), and it precedes the body in source order.
(interface_declaration
  name: (identifier) @name
  (extends_interfaces
    (type_list
      (type_identifier) @base)*)?
  body: (interface_body) @body) @def.interface @ref.inherit

; Enum declarations
(enum_declaration
  name: (identifier) @name
  body: (enum_body) @body
  (super_interfaces
    (type_list
      (type_identifier) @impl)*)?) @def.type @ref.inherit

; Record declarations (Java 14+)
(record_declaration
  name: (identifier) @name
  body: (class_body) @body) @def.type

; Method declarations
(method_declaration
  name: (identifier) @name
  parameters: (formal_parameters) @params
  body: (block)? @body) @def.method

; Constructor declarations
(constructor_declaration
  name: (identifier) @name
  parameters: (formal_parameters) @params
  body: (constructor_body) @body) @def.method

; Method invocations
(method_invocation
  name: (identifier) @callee
  arguments: (argument_list) @args) @ref.call

; Constructor invocations: new Foo()
(object_creation_expression
  type: (type_identifier) @callee) @ref.call

; Import statements
(import_declaration
  (scoped_identifier) @module) @ref.import

; Package declaration
(package_declaration
  (identifier) @module) @ref.import

; Syntax errors
(ERROR) @syntax.error
(MISSING) @syntax.error

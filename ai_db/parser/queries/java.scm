; Java tree-sitter queries for symbols, cross-refs, and syntax errors

; Class, interface and enum declarations
(class_declaration
  name: (identifier) @name
  body: (class_body) @body) @def.class

(interface_declaration
  name: (identifier) @name
  body: (interface_body) @body) @def.interface

(enum_declaration
  name: (identifier) @name
  body: (enum_body) @body) @def.type

; Supertypes: one match per supertype (a quantified list captures only the first).
; Generic supertypes (`Base<T>`) are generic_type nodes: capture their type_identifier.
(class_declaration
  superclass: (superclass [
    (type_identifier) @base
    (generic_type (type_identifier) @base)
  ])) @ref.inherit

; `implements` on classes and enums (`super_interfaces`, the `interfaces` field)
(class_declaration
  interfaces: (super_interfaces
    (type_list [
      (type_identifier) @impl
      (generic_type (type_identifier) @impl)
    ]))) @ref.inherit

(enum_declaration
  interfaces: (super_interfaces
    (type_list [
      (type_identifier) @impl
      (generic_type (type_identifier) @impl)
    ]))) @ref.inherit

; Interface `extends` (an anonymous `extends_interfaces` child, not a field)
(interface_declaration
  (extends_interfaces
    (type_list [
      (type_identifier) @base
      (generic_type (type_identifier) @base)
    ]))) @ref.inherit

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

; Syntax errors
(ERROR) @syntax.error
(MISSING) @syntax.error

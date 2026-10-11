; C# tree-sitter queries for symbols, cross-refs, and syntax errors.
; C# does not mark base class vs interface syntactically, so every entry of a
; base_list is an @base. Generic bases (IRepository<T>) record the bare name.

; Classes, structs, interfaces, records
(class_declaration name: (identifier) @name) @def.class
(struct_declaration name: (identifier) @name) @def.struct
(interface_declaration name: (identifier) @name) @def.interface
(record_declaration name: (identifier) @name) @def.type

; Inheritance: one match per base_list entry (a quantified alternation would only
; capture the first one). Generic bases (IRepository<T>) record the bare name.
(class_declaration (base_list (identifier) @base)) @ref.inherit
(class_declaration (base_list (generic_name (identifier) @base))) @ref.inherit
(class_declaration (base_list (qualified_name) @base)) @ref.inherit
(struct_declaration (base_list (identifier) @base)) @ref.inherit
(struct_declaration (base_list (generic_name (identifier) @base))) @ref.inherit
(struct_declaration (base_list (qualified_name) @base)) @ref.inherit
(interface_declaration (base_list (identifier) @base)) @ref.inherit
(interface_declaration (base_list (generic_name (identifier) @base))) @ref.inherit
(interface_declaration (base_list (qualified_name) @base)) @ref.inherit
(record_declaration (base_list (identifier) @base)) @ref.inherit
(record_declaration (base_list (generic_name (identifier) @base))) @ref.inherit
(record_declaration (base_list (qualified_name) @base)) @ref.inherit

; Enums and delegates
(enum_declaration name: (identifier) @name) @def.type
(delegate_declaration name: (identifier) @name) @def.type

; Methods, constructors, destructors, operators, local functions
(method_declaration name: (identifier) @name parameters: (parameter_list) @params) @def.method
(constructor_declaration name: (identifier) @name parameters: (parameter_list) @params) @def.method
(destructor_declaration name: (identifier) @name) @def.method
(local_function_statement name: (identifier) @name) @def.function

; Properties
(property_declaration name: (identifier) @name) @def.property

; Calls: Foo(), obj.Foo(), Foo<T>(), obj.Foo<T>()
(invocation_expression function: (identifier) @callee) @ref.call
(invocation_expression function: (member_access_expression name: (identifier) @callee)) @ref.call
(invocation_expression function: (generic_name (identifier) @callee)) @ref.call
(invocation_expression function: (member_access_expression name: (generic_name (identifier) @callee))) @ref.call

; Constructions: new Foo(), new Foo<T>(), new Ns.Foo()
(object_creation_expression type: (identifier) @callee) @ref.call
(object_creation_expression type: (generic_name (identifier) @callee)) @ref.call
(object_creation_expression type: (qualified_name name: (identifier) @callee)) @ref.call

; using System; using System.Text; (aliases `using A = X.Y;` record X.Y)
(using_directive !name (identifier) @module) @ref.import
(using_directive (qualified_name) @module) @ref.import

; Syntax errors
(ERROR) @syntax.error
(MISSING) @syntax.error

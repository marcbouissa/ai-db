; Apple Pkl tree-sitter queries for symbols, cross-refs, and syntax errors.
; The Pkl grammar has almost no named fields: a declaration's name is its first
; direct `identifier` child (methods: inside `methodHeader`).

; Classes, type aliases
(clazz (identifier) @name) @def.class
(typeAlias (identifier) @name) @def.type

; Methods (in a class) and module-level functions
(classBody (classMethod (methodHeader (identifier) @name)) @def.method)
(module (classMethod (methodHeader (identifier) @name)) @def.function)

; Properties: module-level ones are the config values
(classBody (classProperty (identifier) @name) @def.property)
(module (classProperty (identifier) @name) @def.property)

; Inheritance: class Service extends Base / extends lib.Base
(clazz (classExtendsClause (qualifiedIdentifier (identifier) @base .))) @ref.inherit

; Calls: f(x), obj.f(x)
(unqualifiedAccessExpr (identifier) @callee (argumentList)) @ref.call
(qualifiedAccessExpr (identifier) @callee (argumentList)) @ref.call

; Object construction: new Foo { ... }, new lib.Foo { ... }
(newExpr (declaredType (qualifiedIdentifier (identifier) @callee .))) @ref.call

; Imports and module extension/amendment: import "x.pkl", extends "base.pkl"
(importClause (stringConstant (slStringLiteralPart) @module)) @ref.import
(extendsOrAmendsClause (stringConstant (slStringLiteralPart) @module)) @ref.import

; Syntax errors
(ERROR) @syntax.error
(MISSING) @syntax.error

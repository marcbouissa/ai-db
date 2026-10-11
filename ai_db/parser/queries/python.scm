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

; Method definitions: direct children of a class body (also when decorated).
; The function pattern above matches every function_definition (tree-sitter queries
; cannot express "not inside a class"), so extract_graph drops its class-body matches.
(class_definition
  body: (block
    [
      (function_definition
        name: (identifier) @name
        parameters: (parameters) @params
        body: (block) @body) @def.method
      (decorated_definition
        definition: (function_definition
          name: (identifier) @name
          parameters: (parameters) @params
          body: (block) @body) @def.method)
    ]))

; Import statements (`import a.b`, `import a.b as c`): one ref per imported module
(import_statement
  name: (dotted_name) @module) @ref.import

(import_statement
  name: (aliased_import
    name: (dotted_name) @module)) @ref.import

; `from a.b import c [as d]` and relative `from .x import y` / `from . import y`:
; one ref to the module it imports from
(import_from_statement
  module_name: (dotted_name) @module) @ref.import

(import_from_statement
  module_name: (relative_import) @module) @ref.import

; ... and one per imported name (`callers X` lists the files importing X; the resolver
; links a file to the definitions it imports)
(import_from_statement
  name: [
    (dotted_name) @module
    (aliased_import name: (dotted_name) @module)
  ]) @ref.import

; Call expressions - capture the call and the called function name
(call
  function: [
    (identifier) @callee
    (attribute
      attribute: (identifier) @callee)
  ]
) @ref.call

; Inheritance - class bases (`Base`, and the last component of `abc.ABC`, like calls)
(class_definition
  superclasses: (argument_list
    [
      (identifier) @base
      (attribute attribute: (identifier) @base)
    ]) @ref.inherit)

; Types: annotated assignments with a value at module/class scope (`X: TypeAlias = ...`,
; dataclass fields `deps: list[str] = field(...)`), `type X = ...`, and
; `T = TypeVar(...)` / `NewType` / `NamedTuple` / `TypedDict`. An unannotated
; `app = Flask()` or `x = field()` is not a type; extract_graph drops function-local ones.
(assignment
  left: (identifier) @name
  type: (type)
  right: (_)) @def.type

(type_alias_statement
  left: (type (identifier) @name)) @def.type

(assignment
  left: (identifier) @name
  right: (call
    function: [
      (identifier) @_ctor
      (attribute attribute: (identifier) @_ctor)
    ])
  (#any-of? @_ctor "TypeVar" "NewType" "NamedTuple" "TypedDict" "ParamSpec" "TypeVarTuple")) @def.type

; Syntax errors - ERROR and MISSING nodes
(ERROR) @syntax.error
(MISSING) @syntax.error
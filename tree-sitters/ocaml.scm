; Seeded from tree-sitter-ocaml's tags.scm (definitions only).
; Captures: @definition.<kind> on the declaration node, @name on its identifier.

(
  (comment)? @doc .
  (module_definition
    (module_binding (module_name) @name) @definition.module
  )
  (#strip! @doc "^\\(\\*+\\s*|\\s*\\*+\\)$")
)

(
  (comment)? @doc .
  (module_type_definition (module_type_name) @name) @definition.interface
  (#strip! @doc "^\\(\\*+\\s*|\\s*\\*+\\)$")
)

(
  (comment)? @doc .
  [
    (class_definition
      (class_binding (class_name) @name) @definition.class
    )
    (class_type_definition
      (class_type_binding (class_type_name) @name) @definition.class
    )
  ]
  (#strip! @doc "^\\(\\*+\\s*|\\s*\\*+\\)$")
)

(
  (comment)? @doc .
  (method_definition (method_name) @name) @definition.method
  (#strip! @doc "^\\(\\*+\\s*|\\s*\\*+\\)$")
)

(
  (comment)? @doc .
  (type_definition
    (type_binding
      name: [
        (type_constructor) @name
        (type_constructor_path (type_constructor) @name)
      ]
    ) @definition.type
  )
  (#strip! @doc "^\\(\\*+\\s*|\\s*\\*+\\)$")
)

[
  (constructor_declaration (constructor_name) @name)
  (tag_specification (tag) @name)
] @definition.enum_variant

(field_declaration (field_name) @name) @definition.field

(
  (comment)? @doc .
  (value_definition
    [
      (let_binding pattern: (value_name) @name (parameter))
      (let_binding
        pattern: (value_name) @name
        body: [(fun_expression) (function_expression)]
      )
    ] @definition.function
  )
  (#strip! @doc "^\\(\\*+\\s*|\\s*\\*+\\)$")
)

(
  (comment)? @doc .
  (external (value_name) @name) @definition.function
  (#strip! @doc "^\\(\\*+\\s*|\\s*\\*+\\)$")
)

(
  (comment)? @doc .
  (value_definition
    [
      (let_binding pattern: (parenthesized_operator (_) @name) (parameter))
      (let_binding
        pattern: (parenthesized_operator (_) @name)
        body: [(fun_expression) (function_expression)]
      )
    ] @definition.operator
  )
  (#strip! @doc "^\\(\\*+\\s*|\\s*\\*+\\)$")
)

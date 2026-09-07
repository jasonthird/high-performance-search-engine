; Seeded from tree-sitter-python's tags.scm (definitions only).
; Captures: @definition.<kind> on the declaration node, @name on its identifier.

(module (expression_statement (assignment left: (identifier) @name) @definition.constant))

(class_definition
  name: (identifier) @name) @definition.class

(function_definition
  name: (identifier) @name) @definition.function

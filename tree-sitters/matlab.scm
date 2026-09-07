; Seeded from tree-sitter-matlab's tags.scm (definitions only).
; Captures: @definition.<kind> on the declaration node, @name on its identifier.

 (class_definition
   name: (identifier) @name) @definition.class

(function_definition
  name: (identifier) @name) @definition.function

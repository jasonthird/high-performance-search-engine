; Seeded from tree-sitter-java's tags.scm (definitions only).
; Captures: @definition.<kind> on the declaration node, @name on its identifier.

(class_declaration
  name: (identifier) @name) @definition.class

(method_declaration
  name: (identifier) @name) @definition.method

(interface_declaration
  name: (identifier) @name) @definition.interface

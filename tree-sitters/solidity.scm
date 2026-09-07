; Seeded from tree-sitter-solidity's tags.scm (definitions only).
; Captures: @definition.<kind> on the declaration node, @name on its identifier.

(contract_declaration (_
    (function_definition
        name: (identifier) @name) @definition.method))

(source_file
    (function_definition
        name: (identifier) @name) @definition.function)

(contract_declaration
  name: (identifier) @name) @definition.class

(interface_declaration
  name: (identifier) @name) @definition.interface

(library_declaration
  name: (identifier) @name) @definition.interface

(struct_declaration name: (identifier) @name) @definition.class

(enum_declaration name: (identifier) @name) @definition.class

(event_definition name: (identifier) @name) @definition.class

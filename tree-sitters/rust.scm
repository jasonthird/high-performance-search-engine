; Seeded from tree-sitter-rust's tags.scm (definitions only).
; Captures: @definition.<kind> on the declaration node, @name on its identifier.

(struct_item
    name: (type_identifier) @name) @definition.class

(enum_item
    name: (type_identifier) @name) @definition.class

(union_item
    name: (type_identifier) @name) @definition.class

(type_item
    name: (type_identifier) @name) @definition.class

(declaration_list
    (function_item
        name: (identifier) @name) @definition.method)

(function_item
    name: (identifier) @name) @definition.function

(trait_item
    name: (type_identifier) @name) @definition.interface

(mod_item
    name: (identifier) @name) @definition.module

(macro_definition
    name: (identifier) @name) @definition.macro

; An impl block is a unit: its header names the type, its methods nest inside.
(impl_item type: (_) @name) @definition.implementation

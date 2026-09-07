; Seeded from tree-sitter-elm's tags.scm (definitions only).
; Captures: @definition.<kind> on the declaration node, @name on its identifier.

(value_declaration (function_declaration_left (lower_case_identifier) @name)) @definition.function

(type_declaration ((upper_case_identifier) @name) ) @definition.type

(type_declaration (union_variant (upper_case_identifier) @name)) @definition.union

(module_declaration 
    (upper_case_qid (upper_case_identifier)) @name
) @definition.module

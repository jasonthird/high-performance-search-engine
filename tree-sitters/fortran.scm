; Seeded from tree-sitter-fortran's tags.scm (definitions only).
; Captures: @definition.<kind> on the declaration node, @name on its identifier.

(derived_type_statement
  (type_name) @name) @definition.class

(program_statement
  (name) @name) @definition.module

(module_statement
  (name) @name) @definition.module

(submodule_statement
  (module_name) (name) @name) @definition.module

(interface
 (function
   (function_statement
    (name) @name) @definition.interface))

(interface
 (subroutine
   (subroutine_statement
    (name) @name) @definition.interface))

(function_statement
  (name) @name) @definition.function

(subroutine_statement
  (name) @name) @definition.function

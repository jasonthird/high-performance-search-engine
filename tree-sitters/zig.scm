(function_declaration name: (identifier) @name) @definition.function
(variable_declaration (identifier) @name (struct_declaration)) @definition.class
(variable_declaration (identifier) @name (enum_declaration)) @definition.class
(variable_declaration (identifier) @name (union_declaration)) @definition.class
(test_declaration (string) @name) @definition.test

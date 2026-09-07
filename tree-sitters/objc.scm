(function_definition declarator: (function_declarator declarator: (identifier) @name)) @definition.function
(function_definition declarator: (pointer_declarator declarator: (function_declarator declarator: (identifier) @name))) @definition.function
(method_definition (method_identifier) @name) @definition.method
(method_definition (identifier) @name) @definition.method
(class_interface (identifier) @name) @definition.class
(class_implementation (identifier) @name) @definition.class
(protocol_declaration (identifier) @name) @definition.interface
(struct_specifier name: (type_identifier) @name body: (_)) @definition.class
(enum_specifier name: (type_identifier) @name body: (_)) @definition.class
(preproc_function_def name: (identifier) @name) @definition.macro

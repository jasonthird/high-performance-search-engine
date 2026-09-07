; Seeded from tree-sitter-dart's tags.scm (definitions only).
; Captures: @definition.<kind> on the declaration node, @name on its identifier.

(class_declaration
  name: (identifier) @name) @definition.class

(mixin_declaration
  (identifier) @name) @definition.class

(extension_declaration
  name: (identifier) @name) @definition.class

(extension_type_declaration
  name: (extension_type_name
    (identifier) @name)) @definition.class

(enum_declaration
  name: (identifier) @name) @definition.class

(function_declaration
  signature: (function_signature
    name: (identifier) @name)) @definition.function

(getter_declaration
  signature: (getter_signature
    name: (identifier) @name)) @definition.function

(setter_declaration
  signature: (setter_signature
    name: (identifier) @name)) @definition.function

(external_function_declaration
  signature: (function_signature
    name: (identifier) @name)) @definition.function

(external_getter_declaration
  signature: (getter_signature
    name: (identifier) @name)) @definition.function

(external_setter_declaration
  signature: (setter_signature
    name: (identifier) @name)) @definition.function

(top_level_variable_declaration
  (static_final_declaration_list
    (static_final_declaration
      name: (identifier) @name))) @definition.variable

(top_level_variable_declaration
  (initialized_identifier_list
    (initialized_identifier
      name: (identifier) @name))) @definition.variable

(external_variable_declaration
  (identifier_list
    (identifier) @name)) @definition.variable

(method_signature
  (function_signature
    name: (identifier) @name)) @definition.method

(method_signature
  (getter_signature
    name: (identifier) @name)) @definition.method

(method_signature
  (setter_signature
    name: (identifier) @name)) @definition.method

(method_signature
  (operator_signature)) @definition.method

(constructor_signature
  name: (identifier) @name) @definition.method

(constant_constructor_signature
  (identifier) @name) @definition.method

(factory_constructor_signature
  (identifier) @name) @definition.method

(redirecting_factory_constructor_signature
  (identifier) @name) @definition.method

(type_alias
  (type_identifier) @name) @definition.type

(enum_constant
  name: (identifier) @name) @definition.constant

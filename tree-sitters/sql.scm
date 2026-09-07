(create_table (object_reference name: (identifier) @name)) @definition.table
(create_view (object_reference name: (identifier) @name)) @definition.view
(create_materialized_view (object_reference name: (identifier) @name)) @definition.view
(create_function (object_reference name: (identifier) @name)) @definition.function
; The grammar files the index name under `column:`.
(create_index column: (identifier) @name) @definition.index
(create_type name: (identifier) @name) @definition.type

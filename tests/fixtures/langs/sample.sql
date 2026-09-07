CREATE TABLE widgets (
    id INTEGER PRIMARY KEY,
    width INTEGER NOT NULL
);

CREATE VIEW wide_widgets AS SELECT * FROM widgets WHERE width > 10;

CREATE INDEX widgets_width ON widgets (width);

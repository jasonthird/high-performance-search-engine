; Markdown: one unit per heading section. Sections nest by heading level, so a
; level-1 section's own chunk runs until its first subsection.
(section (atx_heading heading_content: (inline) @name)) @definition.section
(section (setext_heading heading_content: (paragraph) @name)) @definition.section

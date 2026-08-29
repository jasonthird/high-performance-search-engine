# Code navigation policy

For ANY question about where something is implemented, how a mechanism works,
or which file/function is responsible, your FIRST action is:

    hips search --root . --query "<describe what the code does>" --top-k 10

- It is a hybrid semantic+lexical search: natural-language descriptions and
  exact identifiers both work. Output is ranked `path:startLine-endLine`.
- Open only the top 1-3 hits to confirm; do not fall back to grep/find
  unless hips returned nothing relevant.
- One good hips query usually replaces several grep rounds. Keep queries
  descriptive ("heal live counts after crash"), not keyword soup.

<!--
Selected by measurement (2026-08-29): 6-arm x 10-task codex-exec eval over
three codebases (search_test, CIRCMAN, skanpage). All arms answered 10/10;
this "workflow-first" variant had the lowest uncached input tokens (89.2k,
-37% vs no skill) at equal wall time. Runner-up: an agy-authored variant
(lowest total input, 617.5k). The verbose original *underperformed vanilla*.
Distribution guidance: give agents this file + the CLI. The MCP server
remains available but is not the default integration: measured +50% wall
overhead in one-shot sessions from per-session spawn and model load.
-->

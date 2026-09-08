# Code navigation policy

For code-navigation questions, use hips before grep or file listing:

- Sandboxed shell: prefer MCP `search_code` with
  `{"query":"<what the code does>","top_k":10}`.
- No shell sandbox: prefer
  `hips search --root . --query "<what the code does>" --top-k 10`.

If MCP is unavailable or serves another repository, use the CLI with an
explicit `--root`. MCP is pinned to its reported root; use `expected_root`
to reject mismatches and `path_glob` to filter within that repository.

Open the top 1–3 ranked `path:startLine-endLine` locations. Use grep for known
exact literals or when hips returns no relevant results. For debugging, use
MCP `verbose: true` or CLI `-v`. Report failed hybrid execution and label any
explicit lexical recovery.

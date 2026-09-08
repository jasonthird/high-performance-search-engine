#!/bin/sh
# Routing hint for code-navigation prompts. Keyword-gated (as Morph's hook
# is) so it costs nothing on prompts that are not about finding code.
# Subagents always get it: they do not reliably inherit CLAUDE.md.
event="$1"
input=$(cat)
if [ "$event" = "UserPromptSubmit" ]; then
  printf '%s' "$input" | grep -qiE '\b(where|find|how does|how is|which (file|function)|implement|locate|explain)\b' || exit 0
fi
command -v hips >/dev/null 2>&1 || exit 0
hint='[hips] For code navigation, use hips before grep or file listing. If the shell is sandboxed, prefer MCP search_code; otherwise prefer `hips search --root . --query "<what the code does>"`. If MCP is unavailable or serves another repository, use the CLI with an explicit --root. Grep is for known exact literals.'
printf '{"hookSpecificOutput":{"hookEventName":"%s","additionalContext":"%s"}}\n' "$event" "$(printf '%s' "$hint" | sed 's/"/\\"/g')"

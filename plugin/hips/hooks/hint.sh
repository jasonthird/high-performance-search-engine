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
hint='[hips] For where-is-it / how-does-it-work questions, run `hips search --root . --query "<what the code does>"` BEFORE grep, find, or directory listing. Grep only for exact literals you already know.'
printf '{"hookSpecificOutput":{"hookEventName":"%s","additionalContext":"%s"}}\n' "$event" "$(printf '%s' "$hint" | sed 's/"/\\"/g')"

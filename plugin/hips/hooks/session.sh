#!/bin/sh
# SessionStart / SessionEnd: lease the repository's background index watcher
# for this Claude Code session. `hips session start` reads the hook payload
# (session_id, cwd) from stdin, takes a lease, starts the watcher if none is
# running, and answers with a one-line context note; `end` releases the
# lease, and the watcher exits after the last one. $PPID here is the claude
# process (verified), so a killed session releases its lease too.
#
# Without `hips` on PATH this is a silent no-op: the hook must never fail.
event="$1"
if ! command -v hips >/dev/null 2>&1; then
  [ -x "$HOME/.cargo/bin/hips" ] && PATH="$HOME/.cargo/bin:$PATH" || exit 0
fi
case "$event" in
  SessionStart) exec hips session start --hook --pid "$PPID" ;;
  SessionEnd)   exec hips session end --hook ;;
esac
exit 0

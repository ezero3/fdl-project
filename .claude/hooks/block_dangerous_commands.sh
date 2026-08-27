#!/usr/bin/env bash
input=$(cat)

if command -v jq >/dev/null 2>&1; then
  command=$(jq -r '.tool_input.command // ""' <<< "$input")
else
  # Safe fallback: scan the full hook payload if jq is not available.
  command=$input
fi

block() { echo "$1" >&2; exit 2; }

grep -qiE '(^|[[:space:];&|()])\\?rm([[:space:]]|$|[;&|)])' <<< "$command" && block "rm commands are not allowed"
grep -qiE ':\(\)\s*\{'                              <<< "$command" && block "Fork bomb detected"
grep -qiE '(curl|wget)\s+.+\|\s*(bash|sh|zsh)'     <<< "$command" && block "Piping remote content directly to shell is not allowed"
grep -qiE 'dd\s+if=.+of=/dev/(sd|nvme|disk)'       <<< "$command" && block "Direct disk write via dd is not allowed"
grep -qiE 'git\s+push\s+.*--force.*(main|master)'  <<< "$command" && block "Force push to main/master is not allowed"
grep -qiE 'git\s+push\s+.*(main|master).*--force'  <<< "$command" && block "Force push to main/master is not allowed"
grep -qiE 'git\s+reset\s+--hard'                   <<< "$command" && block "Destructive git reset --hard is not allowed"
grep -qiE 'git\s+clean\s+-[a-z]*f'                 <<< "$command" && block "Destructive git clean -f is not allowed"
grep -qiE 'chmod\s+-R\s+777\s+/'                   <<< "$command" && block "Setting world-writable permissions on root is not allowed"
grep -qiE '>\s*/etc/'                               <<< "$command" && block "Overwriting system files in /etc is not allowed"
grep -qiE 'mkfs\.'                                  <<< "$command" && block "Formatting a filesystem is not allowed"

exit 0

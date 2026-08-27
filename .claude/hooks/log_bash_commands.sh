#!/usr/bin/env bash
input=$(cat)

if command -v jq >/dev/null 2>&1; then
  command=$(jq -r '.tool_input.command // ""' <<< "$input")
  status=$(
    jq -r '
      if ((.tool_response? | type) == "object") then
        if (.tool_response.interrupted // false) then
          "interrupted"
        elif (.tool_response.isError // .tool_response.is_error // false) then
          "error"
        elif ((.tool_response.exit_code? // .tool_response.exitCode? // null) != null) then
          "exit_" + ((.tool_response.exit_code? // .tool_response.exitCode?) | tostring)
        else
          "ok"
        end
      else
        "ok"
      end
    ' <<< "$input" 2>/dev/null
  )
  [ -n "$status" ] || status="ok"
else
  command=$input
  status="unknown"
fi

json_string() {
  if command -v jq >/dev/null 2>&1; then
    jq -Rn --arg value "$1" '$value'
  else
    printf '"%s"' "$(printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g')"
  fi
}

ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

printf '%s\n' "{\"ts\":\"$ts\",\"cmd\":$(json_string "$command"),\"status\":\"$status\"}" \
  >> "$(dirname "$0")/../audit.jsonl"

exit 0

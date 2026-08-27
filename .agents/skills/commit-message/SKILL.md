---
name: commit-message
description: Create a commit message by analyzing git diffs
allowed-tools: Bash(git status:*), Bash(git diff --staged), Bash(git commit:*)
---

## Context:

- Current git status: !`git status`
- Current git diff: !`git diff --staged`

Analyze above staged git changes and create a commit message. Use present tense and explain "why" something has changed, not just "what" has changed.

## Types of commits:
- `feat` - New feature
- `fix` - Bug fix
- `refactor` - Refactoring code
- `docs` - Documentation
- `style` - Styling/formatting
- `test` - Tests
- `perf` - Performance

## Format:
Use the following format for making the commit message:
```
<type>: <concise_description>
<optional_body_explaining_why>
```
`<concise_description>`: the description should consist of 1-2  precise sentences that describe commit most accurate


## Output:

1. Show summary of changes currently staged
2. Propose commit message with appropriate emoji
3. Ask for confirmation before committing

DO NOT auto-commit - wait for user approval, and only commit if the user says so.
DO NOT add any `Co-Authored-By ...`
---
name: openspec-apply-change
description: "Implement tasks from an OpenSpec change by reading task definitions, generating code changes, marking tasks complete, and tracking progress. Use when the user wants to start or resume implementation, work through spec tasks, check the task list, or apply the next pending change."
license: MIT
compatibility: Requires openspec CLI.
metadata:
  author: openspec
  version: "1.0"
  generatedBy: "1.2.0"
---

Implement tasks from an OpenSpec change — read task definitions, generate focused code changes, update task status, and report progress.

**Input**: Optionally specify a change name (e.g., `apply add-auth`). If omitted, infer from conversation context. If vague or ambiguous, MUST prompt for available changes.

**Steps**

1. **Select the change**

   If a name is provided, use it. Otherwise:
   - Infer from conversation context if the user mentioned a change
   - Auto-select if only one active change exists
   - If ambiguous, run `openspec list --json` to get available changes and use the **AskUserQuestion tool** to let the user select

   Always announce: "Using change: <name>" and how to override (e.g., `/opsx:apply <other>`).

2. **Check status to understand the schema**
   ```bash
   openspec status --change "<name>" --json
   ```
   Parse the JSON to understand:
   - `schemaName`: The workflow being used (e.g., "spec-driven")
   - Which artifact contains the tasks (typically "tasks" for spec-driven, check status for others)

3. **Get apply instructions**

   ```bash
   openspec instructions apply --change "<name>" --json
   ```

   This returns context file paths, progress (total/complete/remaining), task list with status, and a dynamic instruction.

   **Handle states:**
   - `state: "blocked"` (missing artifacts): show message, suggest openspec-continue-change
   - `state: "all_done"`: congratulate, suggest archive
   - Otherwise: proceed to implementation

4. **Read context files**

   Read all files listed in `contextFiles` from the apply instructions output. File sets vary by schema (e.g., spec-driven uses proposal/specs/design/tasks).

5. **Show current progress**

   Display:
   - Schema being used
   - Progress: "N/M tasks complete"
   - Remaining tasks overview
   - Dynamic instruction from CLI

6. **Implement tasks (loop until done or blocked)**

   For each pending task:
   - Show which task is being worked on
   - Make the code changes required
   - Keep changes minimal and focused
   - Mark task complete in the tasks file: `- [ ]` → `- [x]`
   - Continue to next task

   **Pause if:**
   - Task is unclear → ask for clarification
   - Implementation reveals a design issue → suggest updating artifacts
   - Error or blocker encountered → report and wait for guidance
   - User interrupts

7. **On completion or pause, show status**

   Display:
   - Tasks completed this session
   - Overall progress: "N/M tasks complete"
   - If all done: suggest archive
   - If paused: explain why and wait for guidance

**Output format**: Always show `## Implementing: <change-name> (schema: <schema-name>)` as a header, then for each task: `Working on task N/M: <description>` followed by `✓ Task complete`. On completion, show final progress (`N/N tasks complete ✓`) and suggest archiving. On pause, show the issue encountered with numbered options for the user.

**Guardrails**
- Always read context files (from apply instructions output) before starting — use `contextFiles` from CLI output, don't assume file names
- Keep code changes minimal and scoped to each task
- Update task checkbox (`- [ ]` → `- [x]`) immediately after completing each task
- Pause and ask before proceeding when: task is ambiguous, implementation reveals a design issue (suggest artifact updates), error or blocker encountered, or user interrupts — never guess
- Keep going through tasks until done or blocked

**Fluid Workflow Integration**

This skill can be invoked anytime — before all artifacts are done (if tasks exist), after partial implementation, or interleaved with other actions. If implementation reveals design issues, suggest updating artifacts rather than blocking.

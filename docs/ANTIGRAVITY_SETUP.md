# Antigravity setup for this project

One-time preparation so the coding agents (Gemini 3.1 Pro / Claude Opus 4.6)
work smoothly on `C:\Users\pmqua\PycharmProjects\OpenRadioss_Python`.
Adapted from the proven vtk_to_d3plot_converter setup (researched 2026-08-02).

## Which app

| App | Version | Use |
|---|---|---|
| Antigravity 2.0 (`...\Programs\Antigravity\Antigravity.exe`) | 2.4.3 | **The user's choice.** Standalone agent manager. |
| Antigravity IDE (`...\Programs\Antigravity IDE\Antigravity IDE.exe`) | 2.1.1 | VS Code fork; alternative with line-by-line diff review. |

Both share the same agent harness, rules, skills and MCP config (`~/.gemini/`),
so this repo's `AGENTS.md`, `.agents/rules/`, `.agents/skills/` and
`.agents/workflows/` work identically in either app. Notes for 2.0:

- One milestone = one conversation, ONE agent at a time. The roadmap
  milestones (docs/STATE.md) are mostly independent, but do not fan out
  parallel agents on the same working tree.
- Point tasks at the main project folder; skip worktrees (simpler, and the
  .venv is in the workspace root).
- Code review: 2.0 reviews via artifacts (Implementation Plan / diffs /
  Walkthrough). For line-by-line reading, open the folder in PyCharm and
  review the git diff after each milestone before approving the next.

## What this repo already provides (do not recreate)

- `AGENTS.md` — agent rules: env, commands, terminal discipline, domain rules.
  Officially read by Antigravity.
- `docs/STATE.md` — status + the milestone roadmap. `AGENTS.md` points to it
  with `@docs/STATE.md`.
- `.agents/rules/env.md` — Always-On rule pinning the Python interpreter
  (workaround for the documented "agent ignores the venv" bug).
- `.agents/workflows/milestone.md` — the `/milestone N` review-gated flow.
- `.agents/skills/` — run-reference-openradioss, validation-compare,
  lspp-check.
- `.venv` — Python 3.12.3 with all dependencies (numpy, scipy, numba, pytest;
  pins in `requirements-lock.txt`).
- `tests/data/rd_decks/` — vendored official corpus decks for the guarded
  corpus tests.
- Git repository. Commit after every green sub-step — it is the only undo,
  and quota pauses mid-milestone are normal.

## Terminal-hang mitigations (Windows)

Known Windows bugs on the shared harness: commands stuck on "Running…"
forever (EOF never signalled), and the `c:\` vs `C:\` path-comparison hang.
Mitigations:

- Primary (already enforced by `AGENTS.md` and `.agents/rules/env.md`): one
  flat command per call, no outer quotes around whole commands, no
  `&&`/pipes/redirection, absolute `.venv\Scripts\*.exe` paths, no
  long-running processes (never launch `pyradioss-gui`; the full 25-min
  suite only as a deliberate final gate).
- If a 2.0 agent terminal wedges on a command: cancel the step and have the
  agent re-issue a flatter/simpler form — do not let it retry identically.
- IDE only (VS Code fork), if you ever switch: user settings JSON —
  `"terminal.integrated.defaultProfile.windows": "PowerShell"` (pwsh 7, NOT
  "Windows PowerShell") and `"terminal.integrated.windowsEnableConpty": false`.

## Agent permission settings

There is **no sandbox on Windows** — the allow/deny lists are the only
guardrail.

- Terminal auto-execution: **Auto** (2.0) — never Turbo. (IDE equivalent:
  Request Review.)
- Allow list (token-prefix matched):
  `.venv\Scripts\python.exe`, `.venv\Scripts\pip.exe`, `git status`,
  `git diff`, `git log`, `git add`, `git commit`,
  `powershell -File tools\run_reference_or.ps1`
- Deny list:
  `Remove-Item`, `rm`, `del`, `rmdir`, `git push`, `git reset --hard`,
  `Invoke-WebRequest`, `curl`, `wget`, `pip install`
- **Outside-of-workspace file access: Ask** — this is the line that keeps
  agents from writing into `E:\openradioss_run`, `E:\foxcore_data`,
  `C:\OpenRadioss` and `C:\OpenRadioss_old` (all READ-ONLY reference data;
  reads are fine and needed — the Fortran source citations live there).
- Respect `.gitignore`: on (keeps `.venv` out of the context index).
- Browser URL allowlist: `github.com`, `raw.githubusercontent.com`,
  `pypi.org`, `openradioss.org`, `openradioss.atlassian.net`,
  `help.altair.com` — for upstream source, docs and the Radioss reference
  manuals.

## Model choice and quota

- Recommendation: **Claude Opus 4.6 (thinking)** for physics/numerics
  milestones (Fortran-fidelity work rewards thinking models); Gemini 3.1 Pro
  (High) is fine for parsing/reader/tooling milestones and spreads quota.
- Quota: Pro refreshes every ~5 h up to a weekly cap; limits are dynamic.
  This is WHY milestones are sliced small and why commits happen at every
  green sub-step — a milestone must be resumable after a hard quota stop.
- Decide the "AI Credit Overages" setting deliberately (Never = hard stop).

## How to run the work

1. Open Antigravity 2.0 and point the workspace at the **project folder**
   (`C:\Users\pmqua\PycharmProjects\OpenRadioss_Python`), main folder, no
   worktree.
2. One milestone = one conversation ("context rot" in long sessions is the
   documented failure mode); docs/STATE.md + git history carry state.
3. Kickoff prompt for each milestone: see `docs/ANTIGRAVITY_PROMPT.md`
   (or simply `/milestone 42` if the workflow command is picked up).
4. Review the Implementation Plan artifact, approve, let it work.
5. It must finish by running the fast tier and showing the output, then
   produce a Walkthrough artifact.
6. Verify yourself (do not trust the walkthrough alone):
   ```powershell
   .venv\Scripts\python.exe -m pytest -q -m "not slow"
   ```
7. Review the git diff (PyCharm) — pay attention to any modified
   pre-existing test: demand the tightening-vs-weakening justification.
8. Tick the milestone checkbox in `docs/STATE.md` if the agent forgot.
9. Start a **new conversation** for the next milestone.

Optional: `/grill-me` on the first milestone makes the agent ask its
clarifying questions first. Avoid `/goal` (runs to completion without
review) until the loop has earned trust.

## MCP (optional)

Config is shared across IDE/2.0/CLI at `~/.gemini/config/mcp_config.json`.
Not needed for this project. If you ever add servers: the key is
`serverUrl`, not `url` — configs copied from Cursor/VS Code fail silently.

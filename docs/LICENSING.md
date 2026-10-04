# Licensing — the MIT / GPL-3.0 / AGPL-3.0 contradiction

This document is the licensing gate for pyradioss (the Python port of
OpenRadioss). It records the contradiction found in Phase 0 / Task 0.0, the
four lawful ways out of it, each with the licence text that decides it, and
the decision status.

**Status: no maintainer decision recorded.** Task 0.0 does not choose; see
[Decision](#decision).

All upstream citations name a path relative to
`$HOME/Projects/OpenRadioss` (so the read-only upstream checkout is
`OpenCourant/...`, never written to). Line ranges are the ones actually read
while writing this file; `git log` on this document's first commit is the
arbiter if a range is ever disputed.

## The contradiction

| Artefact | Declared licence |
| --- | --- |
| `LICENSE:1-3` | `MIT License` / `Copyright (c) 2026 Minh Quang Pham` |
| `README.md:300-302` | `GPL-3.0 (see [LICENSE](LICENSE)) — inherited from files derived from OpenRadioss, © Altair Engineering Inc.` |
| `pyproject.toml:10` | `license = { file = "LICENSE" }` — points at the MIT file |
| upstream `OpenCourant/LICENSE.md:1-3` | `GNU AFFERO GENERAL PUBLIC LICENSE` / `Version 3, 19 November 2007` |
| upstream source headers, e.g. `OpenCourant/engine/source/engine/resol.F:1-7` | `GNU Affero General Public License … either version 3 of the License, or (at your option) any later version.` |

Three artefacts, three different answers, and none of them matches the
licence of the code that was transcribed from them.

The upstream licence is **AGPL-3.0-or-later**. `OpenCourant/LICENSE.md:1-3`
is the licence text itself:

> `### GNU AFFERO GENERAL PUBLIC LICENSE`
> `Version 3, 19 November 2007`

The per-file header confirms the `-or-later` grant and the copyright holder.
`OpenCourant/engine/source/engine/resol.F:1-7`:

> `Copyright>        OpenRadioss`
> `Copyright>        Copyright (C) 2026 Siemens`
> `Copyright>`
> `Copyright>        This program is free software: you can redistribute it and/or modify`
> `Copyright>        it under the terms of the GNU Affero General Public License as published by`
> `Copyright>        the Free Software Foundation, either version 3 of the License, or`
> `Copyright>        (at your option) any later version.`

Copyright is **Siemens**, not Altair: the same header's commercial
alternative paragraph, `OpenCourant/engine/source/engine/resol.F:18-23`,
points at Simcenter Radioss:

> `Copyright>        Commercial Alternative: Simcenter Radioss Software`
> …
> `Copyright>        software under a commercial license.  Contact Siemens to discuss further if the`
> `Copyright>        commercial version may interest you:`
> `Copyright>        https://www.siemens.com/en-us/products/simcenter/mechanical-simulation/radioss/.`

So `README.md:300-302` misattributes the upstream copyright holder as well as
naming the wrong licence.

This repository is a **literal port**: it transcribes upstream Fortran
formulas into Python file by file. Under AGPL §5 that makes the ported files
derivative works of AGPL-covered code, and §5(c) is decisive —
`OpenCourant/LICENSE.md:213-219`:

> `c) You must license the entire work, as a whole, under this`
> `   License to anyone who comes into possession of a copy. This`
> `   License will therefore apply, along with any applicable section 7`
> `   additional terms, to the whole of the work, and all its parts,`
> `   regardless of how they are packaged. This License gives no`
> `   permission to license the work in any other way, but it does not`
> `   invalidate such permission if you have separately received it.`

Two further terms matter for a port:

- §5(a)-(b), `OpenCourant/LICENSE.md:207-212`: the work must carry prominent
  modified notices, the relevant date, and a notice that it is released under
  the AGPL.
- §13, `OpenCourant/LICENSE.md:536-547`: a modified version that users
  interact with **through a network** must offer those users the
  Corresponding Source. pyradioss ships an optional tkinter GUI, so option 1
  below brings §13 into scope for anyone serving the port over a network.

`OpenCourant/LICENSE.md:625-628` is the FSF's own recommended per-file notice
("attach them to the start of each source file", lines 625-626), and
`OpenCourant/LICENSE.md:630-644` is the notice itself as a template — the
pattern option 1 would follow for the ported modules.

## The four lawful resolutions

### Option 1 — Relicense the port to AGPL-3.0-or-later

Replace the MIT `LICENSE` with the AGPL text, set
`pyproject.toml: license = "AGPL-3.0-or-later"`, and state
AGPL-3.0-or-later in `README.md`. Consistent with the upstream source it
derives from; removes the conflict outright. Cost: the port can no longer be
distributed under permissive terms, and every downstream file that adds
upstream-derived code must carry the AGPL header.

### Option 2 — Keep the existing files' derived status and relicense only the new program

Not lawful for a *literal* port. The new files are AGPL-covered derivative
works of AGPL-covered code, so AGPL §5(c)
(`OpenCourant/LICENSE.md:213-219`) pulls the whole work — including the
repository as a whole — under the AGPL: the licence "will therefore apply …
to the whole of the work, and all its parts, regardless of how they are
packaged", and it "gives no permission to license the work in any other way".
Listed because it is the option most often *assumed* correct, and it is not.

### Option 3 — Replace the copied expressions

Re-derive every ported formula from the published physics literature rather
than transcribing it from the Fortran, citing the upstream files as
references only. This is the **only** route that keeps a permissive licence
(MIT), because without expression-level copying there is no derivative work
to license. Cost: it converts a transcription task into a re-derivation task —
a large, honest cost increase — and it requires the maintainer to accept that
"line-for-line port" and "permissive licence" are mutually exclusive.

### Option 4 — Obtain a commercial or dual-licence grant from Siemens

Ask Siemens for a commercial or dual-licence grant covering the derivative
work. The grant path exists upstream: every source header advertises it,
`OpenCourant/engine/source/engine/resol.F:18-23`:

> `Copyright>        Commercial Alternative: Simcenter Radioss Software`
> …
> `Copyright>        software under a commercial license.  Contact Siemens to discuss further if the`
> `Copyright>        commercial version may interest you:`

Note that AGPL §5(c) itself contemplates this route
(`OpenCourant/LICENSE.md:217-219`): the AGPL "gives no permission to license
the work in any other way, but it does not invalidate such permission if you
have separately received it". Cost: it is a negotiation, it is not free, and
nothing can be assumed about the outcome — the repository cannot depend on it.

## Decision

**No decision recorded, and none may be recorded by an agent.** The licence
choice is a MAINTAINER decision (`plan/00_ORCHESTRATION.md` §1.3) and no such
decision exists in this repository. Per the Task 0.0 brief this task ends at
*recommendation*: `pyproject.toml:10` and `README.md:300-302` are deliberately
left untouched, so neither artefact silently adopts any option on the
maintainer's behalf.

**Recommendation: Option 1 — relicense the port to AGPL-3.0-or-later**,
because the program is a literal transcription of AGPL source: AGPL §5(c)
(`OpenCourant/LICENSE.md:213-219`) makes the whole work AGPL, Option 2 is
therefore unlawful, Option 3 contradicts the project's stated goal (a faithful
port), and Option 4 is not guaranteed and cannot gate the roadmap.

*A recommendation pending confirmation is not a decision.* An earlier revision
of this file carried a "**Ruling:** … as a controller ruling pending
maintainer confirmation" line directly under the recommendation; it claimed a
decision that no maintainer made, so it contradicted the two sentences above
it and has been removed. The only status this file records is the one in this
section: none.

### Consequences of the unrecorded decision

- `tests/test_p0_licensing.py::test_declared_licence_is_consistent` is marked
  `@pytest.mark.xfail(strict=True)` with the contradiction as its reason. It
  stays red by design; the day the artefacts are reconciled it XPASSes and
  FAILS the suite, forcing the marker to be removed deliberately.
- Every downstream phase whose entry criterion is "Task 0.0 decided" is
  **unsatisfied**. This is the intended blocking behaviour: no further phase
  may add upstream-derived code until a maintainer records one of the four
  options above here.
- Nothing in this repository's current `LICENSE`, `README.md` or
  `pyproject.toml` should be read as evidence of which option was chosen.

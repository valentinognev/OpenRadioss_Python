# Licensing — the MIT / GPL-3.0 / AGPL-3.0 contradiction, and its resolution

This document is the licensing record for pyradioss (the Python port of
OpenRadioss). It records the contradiction found in Phase 0 / Task 0.0, the
four lawful ways out of it, each with the licence text that decides it, and
the decision.

**Status: DECIDED — 2026-10-07, by the maintainer. This repository is licensed
AGPL-3.0-or-later** (Option 1). See [Decision](#decision).

All upstream citations name a path relative to
`$HOME/Projects/OpenRadioss` (so the read-only upstream checkout is
`OpenCourant/...`, never written to). Line ranges are the ones actually read
while writing this file; `git log` on this document's first commit is the
arbiter if a range is ever disputed.

## The contradiction

The table below is the contradiction **as Phase 0 found it, before the
2026-10-07 decision**. It is kept as the record of what was wrong; the current
state is in [Decision](#decision).

| Artefact | Declared licence (as found, 2026-10-03) |
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

**Decision recorded 2026-10-07, by the maintainer: Option 1 — this repository
is licensed AGPL-3.0-or-later, as a whole.**

The maintainer's words, verbatim:

> as to Licence resolution, set the license whatever you want, I dont care,
> just that it would not stop the development

This is a maintainer decision, recorded as such. It supersedes the earlier
text of this section, which stated that no decision was recorded and that none
might be recorded by an agent; that text described the state of the repository
before 2026-10-07 and is no longer true.

### Why Option 1, and not another

- **Option 2 (relicense only the new program) is unlawful here.** The new
  files are AGPL-covered derivative works of AGPL-covered code, so AGPL §5(c)
  (`OpenCourant/LICENSE.md:213-219`) applies the AGPL "to the whole of the
  work, and all its parts, regardless of how they are packaged". A literal
  port cannot be split off from its own transcribed expression.
- **Option 3 (re-derive every formula from published physics) would keep a
  permissive licence, but it contradicts this project's stated goal** — a
  faithful, Fortran-citing port — and it converts a transcription task into a
  re-derivation task. The maintainer's condition above ("just that it would
  not stop the development") rules it out.
- **Option 4 (a Siemens grant) cannot gate the roadmap.** The path exists
  upstream, but nothing can be assumed about the outcome. It is not closed by
  this decision; it remains available to the maintainer at any time.
- Option 1 is consistent with the upstream source the port derives from and
  removes the conflict outright.

### What was applied

| Artefact | Now |
| --- | --- |
| `LICENSE` | The complete AGPL-3.0 text, preceded by this project's `-or-later` grant notice and copyright (© 2026 Minh Quang Pham), and a note that the port contains OpenRadioss-derived code © 2026 Siemens. |
| `pyproject.toml` | `license = "AGPL-3.0-or-later"` (PEP 639 SPDX string) plus `license-files = ["LICENSE"]`. |
| `README.md` | §License states AGPL-3.0-or-later and cites this decision; it no longer claims GPL-3.0 and no longer misattributes upstream copyright to Altair (upstream is © Siemens). |

### Consequences

- `tests/test_p0_licensing.py::test_declared_licence_is_consistent` is no
  longer `xfail(strict=True)`: the decision exists and is applied, so it now
  **passes for real**. It reads the three artefacts and asserts they agree, so
  a future relicensing that misses one of them turns it red again.
- The development gate is down. "Task P0.0 decided" is satisfied, so no
  further phase is blocked for licensing reasons and new upstream-derived code
  may be added under this decision. New files that transcribe upstream
  Fortran are AGPL-3.0-or-later work, per §5(a)-(b) and the FSF per-file
  notice template at `OpenCourant/LICENSE.md:630-644`.
- Anyone distributing this software, or serving it to users over a network,
  must satisfy AGPL §5 and §13 respectively — see the clauses quoted above.
- The four options above remain on the record. This decision chose one of
  them; it did not delete the reasoning that ruled the others out.

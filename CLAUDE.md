# Umbrella workspace — guidance for Claude

This is the umbrella directory for LISA Analysis Tools. It is transient
dev scaffolding — multiple git repos (`lisa-on-gpu`, `LISAanalysistools`,
`GPUBackendTools`, `BBHx`, `FastEMRIWaveforms`, `GBGPU`, `Eryn`) sit under
here, checked out side by side for convenience during development. The
canonical Claude-facing guidance for this work now lives in
**LISAanalysistools (LAT)**, which is the project's central, published
LISA-physics library and the durable home for cross-repo documentation.

Start here:

- **LISA Analysis Tools–wide rules** (apply to every sub-repo) →
  [`LISAanalysistools/docs/conventions.md`](LISAanalysistools/docs/conventions.md)
- **Cross-repo architecture** ("where does X live", dependency graph,
  backend-wheel model) →
  [`LISAanalysistools/docs/architecture-map.md`](LISAanalysistools/docs/architecture-map.md)
- **Per-repo maps** → `<repo>/docs/codebase-map.md`
- **Global-fit production work** (where everything in the pipeline lives,
  how to move through it without reading the whole tree) →
  [`LISAanalysistools/docs/global-fit-navigation.md`](LISAanalysistools/docs/global-fit-navigation.md)
- **Examining a production run** (the deep-dive routine, the health
  checklist, how the gating is read, known failure signatures, rulings) →
  [`LISAanalysistools/docs/global-fit-deep-dives.md`](LISAanalysistools/docs/global-fit-deep-dives.md)
- **LAT-specific guidance** (build/test/lint, backend architecture, stock
  global fits, ...) → [`LISAanalysistools/CLAUDE.md`](LISAanalysistools/CLAUDE.md)

When a sub-repo has its own `CLAUDE.md`, it points back at
`LISAanalysistools/docs/conventions.md` so the rules stay visible
when that repo is opened standalone.

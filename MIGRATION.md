# Migration playbook — `mikekatz04/*` → `lisa-analysis-tools/*` (DONE)

This document records the GitHub-side and code-side steps for moving
the sprint codebase into the new **LISA Analysis Tools** organization
(slug: `lisa-analysis-tools`). Transfers were executed on 2026-06-08.

Final mapping (note: LAT was renamed during transfer to claim the
org-name URL, and the sprint-root umbrella was NOT transferred — its
useful contents were folded into the LAT repo instead):

| Old URL | New URL | Branch |
|---|---|---|
| `mikekatz04/LISAanalysistools` | `lisa-analysis-tools/lisa-analysis-tools` *(renamed)* | `dev` |
| `mikekatz04/GPUBackendTools`   | `lisa-analysis-tools/GPUBackendTools`   | `spline` |
| `mikekatz04/BBHx`              | `lisa-analysis-tools/BBHx`              | `dev` |
| `mikekatz04/GBGPU`             | `lisa-analysis-tools/GBGPU`             | `dev` |
| `mikekatz04/Eryn`              | `lisa-analysis-tools/Eryn`              | `dev` |
| `mikekatz04/LATW`              | `lisa-analysis-tools/LATW`              | (check) |
| `mikekatz04/lisa_sprint_2026`  | *not transferred* — contents (install.sh, DEVELOPMENT.md, constraints/, tools/) merged into the LAT repo. | `main` |

Repos that **stay put**:

- `mikekatz04/lisa-on-gpu` — retiring, see Phase 3L.7n in `CLAUDE.md`.
- `BlackHolePerturbationToolkit/FastEMRIWaveforms` — community-owned.
- `asantini29/phentax` — external collaborator.

---

## §1 — Pre-flight decisions

Two things to lock in before clicking anything in the GitHub UI.

### 1a. Umbrella repo name

The current `lisa_sprint_2026` is the umbrella. Once it's in the new
org, the cleanest name for the umbrella is `lisa-analysis-tools`
(matching the org slug → URL becomes
`github.com/lisa-analysis-tools/lisa-analysis-tools`). This is the
GitHub convention used by e.g. `numpy/numpy`, `scipy/scipy`,
`jaxlib/jaxlib`. **Rename during transfer** — GitHub's transfer form
lets you change the name in the same step.

Alternative: keep the snapshot name `lisa_sprint_2026` (less clean but
preserves the git-log identity of the snapshot commits).

### 1b. Contact email in CONTRIBUTING.md

Every sub-repo's `CONTRIBUTING.md` has:

> ...reported by contacting the project team via email at
> **`mikekatz04@gmail.com`**.

For an org repo, this should probably be a project list address
(`contact@lisa-analysis-tools.org`?) rather than a personal email. This
is an open decision — not blocking the transfer — but worth deciding so
we can rewrite the line org-wide in one pass.

`pyproject.toml` `author = { email = ... }` fields are author
attribution, not project contact — leave those alone.

---

## §2 — Create the new GitHub organization

This is a **manual step you take in the GitHub UI**. I cannot do it
for you.

1. <https://github.com/organizations/new>
2. **Organization name (slug):** `lisa-analysis-tools`
3. **Display name:** `LISA Analysis Tools` (shown on the org page)
4. **Contact email:** your work or project list address
5. **Plan:** Free (org-level features can be upgraded later)
6. Skip the "invite members" step for now; you'll add collaborators
   after repos land.

Once created, set:

- **Organization profile**: upload an icon if you have one, write a
  one-paragraph description ("Open-source toolkit for LISA data
  analysis...").
- **Optional** — create a special repo named `.github` under the org and
  put a `profile/README.md` in it. Its contents are shown on the org's
  landing page (`github.com/lisa-analysis-tools`). The `DEVELOPMENT.md`
  in this umbrella is a good starting point for that profile README.

---

## §3 — Transfer the 7 repos (GitHub UI)

For **each** of the 7 repos listed above, you go through:

1. Open `https://github.com/mikekatz04/<repo>/settings`
2. Scroll to **Danger Zone → Transfer ownership**.
3. **New owner**: `lisa-analysis-tools`
4. **Repository name in new owner** *(only if renaming)*: e.g.
   `lisa-analysis-tools` for the umbrella, otherwise keep the existing
   name.
5. Type the repo name to confirm. GitHub will:
   - Move all branches, tags, releases, issues, PRs, stars, watchers.
   - Set up a redirect from `mikekatz04/<repo>` → `lisa-analysis-tools/<repo>`
     for clones and web traffic.
   - Move the GitHub Pages site (URL pattern changes — see §6).

Recommended transfer order (low-risk first):

1. `LATW` — pure-Python tutorials, no consumers depend on its URL at
   build time.
2. `Eryn` — standalone sampler.
3. `GPUBackendTools` — used as a build-time include by everyone, but
   the include is via filesystem path, not URL, so the transfer doesn't
   break compiles.
4. `LISAanalysistools` — same.
5. `BBHx`, `GBGPU` — same.
6. `lisa_sprint_2026` (umbrella, rename to `lisa-analysis-tools` if
   chosen in §1a).

Verify after each transfer:

```bash
git ls-remote https://github.com/lisa-analysis-tools/<repo>.git | head -1
# should print the same SHA as the old repo's HEAD
```

---

## §4 — Update local git remotes

After the transfers complete, the old URLs still work (via redirect),
but it's cleanest to update them. From the umbrella directory:

```bash
cd /Users/mkatz/Research/lisa_sprint_2026

git remote set-url origin https://github.com/lisa-analysis-tools/lisa-analysis-tools.git

for d in LISAanalysistools GPUBackendTools BBHx GBGPU Eryn; do
    git -C "$d" remote set-url origin https://github.com/lisa-analysis-tools/${d}.git
done

# Verify
for d in . LISAanalysistools GPUBackendTools BBHx GBGPU Eryn; do
    echo "=== $d ==="; git -C "$d" remote -v
done
```

(LATW is not under the sprint tree right now — update its remote
wherever it lives locally.)

---

## §5 — Apply URL edits across the codebase

GitHub's redirect handles clones, but in-tree URL strings (README
badges, doc links, Sphinx config, the `detector.py` orbit-file
download URL, the `publish.yml` repository-name guard) should be
updated to point at the canonical new URLs.

Edits the migration applies (one commit per sub-repo, all to `dev` /
branch HEAD):

| File pattern | Old → New |
|---|---|
| `README.md` clone URLs | `mikekatz04/<repo>` → `lisa-analysis-tools/<repo>` |
| `README.md` doc badges & links | `mikekatz04.github.io/<repo>` → `lisa-analysis-tools.github.io/<repo>` |
| `docs/source/README.rst` | same as README.md |
| `setup.py` `url=...` | new canonical URL |
| `.github/workflows/publish.yml` `repository.full_name` guard | `mikekatz04/<repo>` → `lisa-analysis-tools/<repo>` |
| `src/<pkg>/utils/citation.py` `title = "mikekatz04/<repo>: ..."` | `lisa-analysis-tools/<repo>: ...` |
| LAT `src/lisatools/detector.py` GitHub raw-file URL (orbit files) | new canonical URL |
| LAT `src/lisatools/diagnostic.py` Eryn doc link | new canonical URL |
| LAT `src/lisatools/sources/defaultresponse.py` lisa-on-gpu doc link | **leave** (lisa-on-gpu is not moving) |
| Eryn `src/eryn/ensemble.py` Sphinx-link rST | new canonical URL |
| All `mikekatz04@gmail.com` in `CONTRIBUTING.md` "report to" lines | **decision needed** (§1b) |
| All `mikekatz04@gmail.com` in `pyproject.toml` `author=` blocks | **leave** (author attribution) |
| All `https://github.com/mikekatz04` in `CONTRIBUTORS.md` author links | **leave** (author attribution) |

The umbrella `README.md` (currently `mikekatz04/sprint_2026`) gets
replaced by the new org-level intro (see `DEVELOPMENT.md`).

**I have NOT applied these yet** — they go in once the transfers
complete. The exact edit list is staged below in `MIGRATION_EDITS.md`
(produced in the next step) so you can review before push.

---

## §6 — GitHub Pages

Each sub-repo currently serves docs from
`https://mikekatz04.github.io/<Repo>`. After transfer, the Pages URL
becomes `https://lisa-analysis-tools.github.io/<Repo>` — and GitHub does
**not** auto-redirect the old Pages URL across an org change.

Per repo, in the new org:

1. Settings → Pages → confirm the source branch (usually `gh-pages` or
   `main /docs`) carried over.
2. If the docs build pushes to `gh-pages` from CI, verify the workflow
   ran cleanly post-transfer (the `repository.full_name` guard in
   `publish.yml` is one of the things §5 updates).
3. Re-publish on `gh-pages` if the build was last triggered from the
   old URL guard.

Optional: configure a custom CNAME on the org (e.g.
`docs.lisa-analysis-tools.org`) so users have a stable URL independent
of GitHub's naming.

---

## §7 — Outside references that still point at the old URLs

These do not block the migration but you should track them:

- **`lisa-on-gpu` README** has `github.com/mikekatz04/LISAanalysistools`
  in its install note. Since `lisa-on-gpu` is staying at
  `mikekatz04/lisa-on-gpu` AND it's being retired (`Phase 3L.7n`), this
  is a low-priority cleanup — fix when its final deprecation notice
  goes out.
- **`asantini29/phentax`** — not yours to change, but if it documents
  installing LISAanalysistools alongside, ping Alessandro to update.
- **Zenodo entries** for previously-released versions reference the old
  GitHub URL. These freeze in place by design (DOIs are immutable);
  future releases under the new org get new Zenodo records.

---

## §8 — Post-migration verification

```bash
# fresh-clone smoke test on a clean machine / new directory:
mkdir /tmp/lat-fresh && cd /tmp/lat-fresh
git clone https://github.com/lisa-analysis-tools/lisa-analysis-tools.git
cd lisa-analysis-tools
./install.sh
python -c "import lisatools, eryn, bbhx, gbgpu, gpubackendtools; \
           print('all import OK')"
```

GH Pages spot-checks:

- <https://lisa-analysis-tools.github.io/lisa-analysis-tools/>
- <https://lisa-analysis-tools.github.io/Eryn/>
- <https://lisa-analysis-tools.github.io/BBHx/>
- <https://lisa-analysis-tools.github.io/GBGPU/>
- <https://lisa-analysis-tools.github.io/GPUBackendTools/>

---

## What I'm doing for you (machine-side)

Already done in this session, ready for you to review:

- ✅ `DEVELOPMENT.md` — the org-level development information page.
- ✅ `install.sh` — rewritten to point at the new org, includes
  BBHx + GBGPU + LATW, pins pybind11 via `constraints/sprint.txt`,
  installs in dependency order. Re-runnable; reuses existing clones.
- ✅ `MIGRATION.md` — this document.

Staged for after you create the org and run the transfers:

- ⏳ Apply the §5 URL edits across all 6 sub-repos (one commit per
  repo, `dev` / branch HEAD).
- ⏳ Update local git remotes via the §4 block.

When you've created the org and finished the transfers, tell me
"transfers done" and I'll run §4 + §5 in one pass.

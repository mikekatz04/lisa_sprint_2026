"""Render a schematic of the LISA sprint 2026 directory.

Produces two artifacts:

1. ``sprint_architecture.png`` -- a matplotlib schematic of the sprint
   sub-repos and their architectural relationships (LAT as the central
   LISA-physics library, GBT as the shared C++/CUDA base, GBGPU/BBHx
   composing on top, FEW + Eryn standalone, lisa-on-gpu retired to a
   pure-Python husk).

2. ``sprint_filetree.dot`` -- a Graphviz DOT file with the full filtered
   recursion of the sprint tree. If the ``dot`` binary is on PATH, it
   also renders ``sprint_filetree.svg``.

Run from anywhere; paths are resolved relative to this file's location.

    python sprint_structure_diagram.py
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SPRINT_ROOT = Path(__file__).resolve().parent

# Directory / file names that add no signal to the tree diagram.
SKIP_DIRS = {
    ".git",
    ".github",
    ".claude",
    ".idea",
    ".vscode",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    "__pycache__",
    "build",
    "_build",
    "dist",
    "node_modules",
    ".tox",
    ".venv",
    "venv",
    "env",
    ".cache",
}
SKIP_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".so",
    ".o",
    ".a",
    ".dylib",
    ".egg-info",
    ".DS_Store",
}
SKIP_NAMES = {".DS_Store"}


def keep(path: Path) -> bool:
    name = path.name
    if name in SKIP_NAMES:
        return False
    if path.is_dir():
        if name in SKIP_DIRS:
            return False
        if name.endswith(".egg-info"):
            return False
        return True
    return path.suffix not in SKIP_SUFFIXES


# ---------------------------------------------------------------------------
# 1. Architectural schematic (matplotlib) — multi-layer detailed view
# ---------------------------------------------------------------------------


# Per-repo (fill, edge) palette. Used by both cells and the legend.
REPO_COLORS: dict[str, tuple[str, str]] = {
    "GBT":   ("#fce6c0", "#a86a1a"),
    "LAT":   ("#cfe5cf", "#2a6a2a"),
    "GBGPU": ("#d6e0f5", "#274472"),
    "BBHx":  ("#c4d6f0", "#1a3a72"),
    "FEW":   ("#e8d6f0", "#5b2a72"),
    "Eryn":  ("#f5d6d6", "#722a2a"),
}

# Column (x_left, x_right) per repo.
COLS: dict[str, tuple[float, float]] = {
    "GBT":   (0.30, 3.50),
    "LAT":   (3.80, 11.20),
    "GBGPU": (11.50, 14.80),
    "BBHx":  (15.10, 18.40),
    "FEW":   (18.70, 21.20),
    "Eryn":  (21.50, 23.30),
}

# Layers: (label, y_baseline, height). Ordered bottom-up.
LAYERS: list[tuple[str, float, float]] = [
    ("Pip wheels (backend matrix)",            0.40, 0.90),
    ("Shared C++/CUDA primitives (GBT base)",  1.60, 1.40),
    ("C++/CUDA core (cutils/)",                3.20, 3.70),
    ("Pybind11 / nanobind wrappers",           7.10, 2.10),
    ("JAX implementation",                     9.40, 2.00),
    ("Python frontend",                       11.60, 2.20),
    ("Application / dev scripts",             14.00, 0.90),
]


@dataclass
class Cell:
    repo: str          # key into COLS / REPO_COLORS
    layer: int         # index into LAYERS
    lines: list[str]   # body text, one per line
    title: str = ""    # optional override; default uses repo name


def render_architecture(out_path: Path) -> None:
    """Layered architectural schematic — post Phase 3L.7n / 3M / dedup.

    Reads (left → right) as the six in-tree repos; (bottom → top) as the
    build stack from pip wheels up through application scripts. Each
    cell names the actual files / classes that live there. Cross-cell
    arrows highlight the non-obvious dependency edges that span repos:
    cuda_complex de-duplication, the `lat_chunked_het_kernels.hh`
    #include used by GBGPU + BBHx, and the LAT-side `*Backend`
    composition that lets downstreams reach LAT-owned wrappers.
    """

    cells: list[Cell] = [
        # ---- Application / dev scripts (layer 6) ----
        Cell("LAT", 6, [
            "scripts/gb_chunked_het/   scripts/sobbh/   scripts/wdm/",
            "scripts/benchmark/        scripts/validation/   scripts/diagnostics/",
        ]),
        Cell("Eryn", 6, [
            "global-fit driver",
            "PE test scripts",
        ]),

        # ---- Python frontend (layer 5) ----
        Cell("LAT", 5, [
            "lisatools.response   directresponse, tdionfly, tdiconfig, parallelbase",
            "lisatools.analysiscontainer   AC + DomainBase + sens auto-instantiation",
            "lisatools.sensitivity / domains   FDDomain, WDMDomain (Python views)",
            "lisatools.sources   DRA shim, defaultresponse",
            "lisatools.detector   pycppdetector frontends",
        ]),
        Cell("GBGPU", 5, [
            "gbgpu.ucb   compute group + UCB physics",
            "gbgpu.sources   GB chunked-het / FD",
            "gbgpu.gbcomps   GB|SOBBH WDMComputations, GBFDComputations",
            "  ← moved lisa-on-gpu → GBGPU (Phase 3L.7i)",
        ]),
        Cell("BBHx", 5, [
            "bbhx.sobbh   SOBBH sources",
            "bbhx.phenomhm   MBH waveforms (PhenomHM)",
            "phentax integration",
        ]),
        Cell("FEW", 5, [
            "fastemriwaveforms.amps / trajectory",
            "EMRI template generator",
        ]),
        Cell("Eryn", 5, [
            "eryn.moves, eryn.ensemble",
            "eryn.nuts",
        ]),

        # ---- JAX implementation (layer 4) ----
        Cell("LAT", 4, [
            "lisatools.jax.response   base, projection, tdi_config,",
            "                          amp_phase_extract            ← Phase 3D",
            "lisatools.jax.wdm   wdm_settings, wdm_domain,",
            "                     wavelet_lookup, fast_inner       ← Phase 3D",
        ]),
        Cell("GBGPU", 4, [
            "gbgpu.jax.sources   ucb",
            "gbgpu.jax.wdm   kernels, heterodyne, fast_inner_het  ← 3F",
            "gbgpu.jax.tdi_on_the_fly + wrappers   ← 3L.7j",
            "gbgpu.jax.wdm.computation_group",
        ]),
        Cell("BBHx", 4, [
            "bbhx.jax.sources   sobbh   ← Phase 3G",
        ]),
        Cell("FEW", 4, [
            "(JAX path: standalone)",
        ]),

        # ---- pybind11 / nanobind wrappers (layer 3) ----
        Cell("GBT", 3, [
            "gpubackendtools_backend_{cpu,cuda12x}",
            "binding.cxx → cspline module",
            "+ get_include() / get_cmake_module_path()",
        ]),
        Cell("LAT", 3, [
            "binding.cxx        → lisatools_backend_*.pycppdetector",
            "binding_flr.cxx    → registers shared Wraps  ★ L2 OWNER",
            "Wraps owned by LAT: Orbits, TDIConfig, WDMSettings,",
            "                    WDMDomain, FDDomain, LISAResponse,",
            "                    {FD,TD}SplineTDIWaveform, LISATDIonTheFly",
            "+ OrbitsView POD  (L3 ABI, sizeof + 15 offsetof asserts)",
        ]),
        Cell("GBGPU", 3, [
            "binding_*.cxx → gbgpu_backend_*.cgbgpu",
            "GBTDIonTheFlyWrap         ← Phase 3L.7g",
            "GBComputationGroupWrap    ← Phase 3L.7h",
        ]),
        Cell("BBHx", 3, [
            "binding_*.cxx → bbhx_backend_*.cbbhx",
            "SOBBHTDIonTheFlyWrap        ← Phase 3L.8",
            "SOBBHComputationGroupWrap   ← Phase 3L.8",
        ]),
        Cell("FEW", 3, [
            "own pybind11 binding",
            "(stays on pybind11; not nanobind-migrated)",
        ]),

        # ---- C++/CUDA core (cutils/, layer 2) ----
        Cell("LAT", 2, [
            "Detector.{hpp,cu}    →  Orbits (CPU/GPU alias-distinct)",
            "LISAResponse.{hh,cu}                        ← 3E (from lisa-on-gpu)",
            "WDMSettings.hh   ← 3L.2     │   WDMDomain.hh   ← 3L.4",
            "FDDomain.hh      ← 3L.1     │   LISATDIonTheFly.{hh,cu}  ← 3L.5",
            "FDSplineTDIWaveform.{hh,cu}    ← 3L.6",
            "TDSplineTDIWaveform.{hh,cu}    ← 3L.6",
            "",
            "★ lat_chunked_het_kernels.hh   ← 3L.7a (#include shared)",
            "★ lat_wdm_fft.hh               ← 3L.7a slice 2",
            "  templated wdm_het_*_kernel + _impl<SourceT>  ← 3L.7a slice 3",
            "",
            "lisatools_header_abi.hpp   LISATOOLS_IS_WRAPPER_OWNER  (L2)",
            "orbits_view.hpp            POD layout asserts          (L3)",
        ]),
        Cell("GBGPU", 2, [
            "GBTDIonTheFly.{hh,cu}     ← 3L.7 carve",
            "GBComputationGroup.{hh,cu}  ← 3L.7h",
            "(GB-specific UCB physics)",
            "",
            "wdm_het_*_impl<GBTDIonTheFly>  via",
            "  #include <lat_chunked_het_kernels.hh>",
            "",
            "cuda_complex.hpp  DELETED (3.dedup) →",
            "  resolves via ${GBT_CUTILS}",
        ]),
        Cell("BBHx", 2, [
            "SOBBHTDIonTheFly.{hh,cu}     ← 3L.8",
            "SOBBHComputationGroup.{hh,cu}  ← 3L.8",
            "PhenomHM CUDA kernels (MBH)",
            "",
            "wdm_het_*_impl<SOBBHTDIonTheFly>  via",
            "  #include <lat_chunked_het_kernels.hh>",
            "",
            "Interpolate.hh    → 2-line shim over",
            "                    gpubackendtools (3L.7c/dedup)",
            "global.h          → wraps gbt_global.h",
        ]),
        Cell("FEW", 2, [
            "EMRI template CUDA kernels",
            "",
            "OWN cuda_complex.hpp",
            "(intentionally diverged: std:: qualified;",
            " not part of sprint-wide GBT dedup)",
        ]),

        # ---- Shared C++/CUDA primitives (GBT base, layer 1) ----
        Cell("GBT", 1, [
            "cuda_complex.hpp   sprint-wide single source",
            "  (LAT / GBGPU / BBHx all dedup'd to here; FEW intentional fork)",
            "Interpolate.cu + InterpolateDevice.hh   cubic spline (tile-x adapter)",
            "FFT helpers   |   gbt_global.h   |   CMake config",
            "Python helpers: gpubackendtools.get_include() / get_cmake_module_path()",
        ]),

        # ---- Pip wheels (backend matrix, layer 0) ----
        Cell("GBT", 0, ["gpubackendtools-cpu  •  -cuda12x"]),
        Cell("LAT", 0, ["lisaanalysistools-cpu  •  -cuda12x"]),
        Cell("GBGPU", 0, ["gbgpu-cpu  •  -cuda12x"]),
        Cell("BBHx", 0, ["bbhx-cpu  •  -cuda12x"]),
        Cell("FEW", 0, ["fastemriwaveforms-cpu  •  -cuda12x"]),
        Cell("Eryn", 0, ["eryn  (pure Python)"]),
    ]

    # Cross-cell arrows: (src repo, src layer, dst repo, dst layer, label, color, rad).
    # All annotate non-obvious cross-repo edges that the layered grid alone hides.
    cross: list[tuple[str, int, str, int, str, str, float]] = [
        # GBT base flows up into every C++/CUDA core.
        ("GBT", 1, "LAT",   2, "headers + CMake", "#a86a1a", 0.0),
        ("GBT", 1, "GBGPU", 2, "cuda_complex (dedup'd via ${GBT_CUTILS})", "#a86a1a", 0.0),
        ("GBT", 1, "BBHx",  2, "cuda_complex + Interpolate", "#a86a1a", 0.0),
        ("GBT", 1, "FEW",   2, "(CMake config only;\nFEW has own cuda_complex)", "#a86a1a", 0.10),

        # LAT-owned shared chunked-het kernels are #include'd by downstreams.
        ("LAT", 2, "GBGPU", 2, "#include <lat_chunked_het_kernels.hh>", "#2a6a2a", -0.25),
        ("LAT", 2, "BBHx",  2, "#include <lat_chunked_het_kernels.hh>", "#2a6a2a", -0.30),

        # Wrapper layer: LAT registers shared Wraps; downstreams reach them via *Backend.
        ("LAT", 3, "GBGPU", 3, "LAT Wraps via LISAToolsBackend\n(no re-registration: L2)", "#2a6a2a", -0.18),
        ("LAT", 3, "BBHx",  3, "LAT Wraps via LISAToolsBackend", "#2a6a2a", -0.22),

        # Python frontend composition.
        ("LAT", 5, "GBGPU", 5, "AnalysisContainer + DomainBase", "#444444", -0.12),
        ("LAT", 5, "BBHx",  5, "", "#444444", -0.18),
        ("LAT", 5, "FEW",   5, "AnalysisContainer", "#444444", -0.22),
        ("LAT", 5, "Eryn",  5, "likelihoods", "#444444", -0.26),
    ]

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig_w = 26.0
    fig_h = 17.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(-0.5, 24.5)
    ax.set_ylim(-1.0, 17.4)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title block (above legend, above column headers).
    ax.text(
        12.0, 16.95,
        "LISA Sprint 2026 — detailed architectural schematic",
        ha="center", va="center", fontsize=18, fontweight="bold",
    )
    ax.text(
        12.0, 16.45,
        "columns = repos under ~/Research/lisa_sprint_2026/   •   "
        "layers (bottom→top) = build stack from pip wheels up through application scripts   •   "
        "★ = sprint-wide invariant   •   ← Phase NN.M = when the artefact moved into its current home",
        ha="center", va="center", fontsize=10, color="#444",
    )

    # Layer band shading + labels (left margin).
    band_colors = ["#f9f9f9", "#f3f3f3"]
    for i, (label, y0, h) in enumerate(LAYERS):
        ax.add_patch(mpatches.Rectangle(
            (-0.4, y0 - 0.05), 23.9, h + 0.10,
            facecolor=band_colors[i % 2], edgecolor="none", zorder=0,
        ))
        ax.text(
            -0.35, y0 + h / 2, label,
            ha="left", va="center", fontsize=9, color="#666", style="italic", zorder=1,
        )

    # Column header strip (repo names) above the application layer.
    for repo, (x0, x1) in COLS.items():
        fill, edge = REPO_COLORS[repo]
        ax.add_patch(FancyBboxPatch(
            (x0, 15.0), x1 - x0, 0.35,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=fill, edgecolor=edge, linewidth=1.4, zorder=2,
        ))
        ax.text(
            (x0 + x1) / 2, 15.18, repo,
            ha="center", va="center", fontsize=11.5, fontweight="bold", color=edge, zorder=3,
        )

    # ------------------------------------------------------------------
    # Cell rendering
    # ------------------------------------------------------------------
    cell_centers: dict[tuple[str, int], tuple[float, float, float, float]] = {}
    for cell in cells:
        x0, x1 = COLS[cell.repo]
        _, y0, h = LAYERS[cell.layer]
        fill, edge = REPO_COLORS[cell.repo]
        rect = FancyBboxPatch(
            (x0 + 0.05, y0 + 0.05), (x1 - x0) - 0.10, h - 0.10,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=fill, edgecolor=edge, linewidth=1.2, zorder=2,
        )
        ax.add_patch(rect)

        # Body: equally-spaced lines, monospace for files/code-ish content.
        body_top = y0 + h - 0.18
        body_bot = y0 + 0.18
        n = max(len(cell.lines), 1)
        if n == 1:
            ys = [(body_top + body_bot) / 2]
        else:
            step = (body_top - body_bot) / max(n - 1, 1)
            ys = [body_top - i * step for i in range(n)]
        for line, ly in zip(cell.lines, ys):
            ax.text(
                x0 + 0.20, ly, line,
                ha="left", va="center",
                fontsize=7.6 if cell.layer >= 2 else 8.0,
                family="DejaVu Sans Mono", color="#1a1a1a", zorder=3,
            )

        cell_centers[(cell.repo, cell.layer)] = (
            x0, x1, y0, y0 + h,
        )

    # ------------------------------------------------------------------
    # Cross-cell arrows
    # ------------------------------------------------------------------
    def edge_pt(box: tuple[float, float, float, float], tx: float, ty: float) -> tuple[float, float]:
        x0, x1, y0, y1 = box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        dx, dy = tx - cx, ty - cy
        if dx == 0 and dy == 0:
            return cx, cy
        scale_x = ((x1 - x0) / 2) / abs(dx) if dx else float("inf")
        scale_y = ((y1 - y0) / 2) / abs(dy) if dy else float("inf")
        s = min(scale_x, scale_y)
        return cx + dx * s, cy + dy * s

    for (sr, sl, dr, dl, label, color, rad) in cross:
        src = cell_centers.get((sr, sl))
        dst = cell_centers.get((dr, dl))
        if src is None or dst is None:
            continue
        scx, scy = (src[0] + src[1]) / 2, (src[2] + src[3]) / 2
        dcx, dcy = (dst[0] + dst[1]) / 2, (dst[2] + dst[3]) / 2
        sx, sy = edge_pt(src, dcx, dcy)
        dx, dy = edge_pt(dst, scx, scy)
        arrow = FancyArrowPatch(
            (sx, sy), (dx, dy),
            arrowstyle="-|>", mutation_scale=12,
            color=color, linewidth=1.3,
            connectionstyle=f"arc3,rad={rad}",
            zorder=4,
        )
        ax.add_patch(arrow)
        if label:
            mx = (sx + dx) / 2 + rad * 0.5
            my = (sy + dy) / 2 + 0.18
            ax.text(
                mx, my, label,
                ha="center", va="bottom", fontsize=7.0, color=color,
                bbox={"boxstyle": "round,pad=0.18", "facecolor": "white",
                      "edgecolor": "none", "alpha": 0.88},
                zorder=5,
            )

    # ------------------------------------------------------------------
    # Side panel: sprint-wide invariants + retired/tooling notes
    # ------------------------------------------------------------------
    side_x = 0.0
    side_y = -0.40
    side_w = 23.5
    side_h = 0.95
    ax.add_patch(FancyBboxPatch(
        (side_x, side_y), side_w, side_h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor="#fff5cc", edgecolor="#806600", linewidth=1.2, zorder=2,
    ))
    ax.text(
        side_x + 0.20, side_y + side_h - 0.22,
        "Sprint-wide invariants",
        ha="left", va="center", fontsize=10, fontweight="bold", color="#5a4400",
    )
    ax.text(
        side_x + 0.20, side_y + side_h - 0.50,
        "★ L1 — constraints/sprint.txt pins pybind11==3.0.4 + nanobind==2.12.0; export PIP_CONSTRAINT before any pip install (cross-wheel type-registry agreement).",
        ha="left", va="center", fontsize=8.0, color="#333",
    )
    ax.text(
        side_x + 0.20, side_y + side_h - 0.72,
        "★ L2 — LISATOOLS_IS_WRAPPER_OWNER macro + per-TU static_assert + tools/check_single_registrant.sh grep gate: only LAT may nb::class_<> a shared Wrap.",
        ha="left", va="center", fontsize=8.0, color="#333",
    )
    ax.text(
        side_x + 0.20, side_y + side_h - 0.92,
        "★ L3 — orbits_view.hpp POD layout asserted at every LAT build (sizeof + 15 offsetofs). CPU/GPU class aliasing required for every class shipped in both wheels.",
        ha="left", va="center", fontsize=8.0, color="#333",
    )

    # Retired + tooling annotation along the very bottom.
    ax.text(
        0.0, -0.85,
        "lisa-on-gpu = 6.4 KB pure-Python husk (DeprecationWarning shim) after Phase 3L.7n; "
        "directory deletion deferred ~1 release cycle.   •   "
        "tools/check_single_registrant.sh + constraints/sprint.txt live at the sprint root.",
        ha="left", va="center", fontsize=8.0, color="#555", style="italic",
    )

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------
    legend_handles = [
        mpatches.Patch(facecolor=REPO_COLORS["GBT"][0],   edgecolor=REPO_COLORS["GBT"][1],   label="GBT — C++/CUDA base"),
        mpatches.Patch(facecolor=REPO_COLORS["LAT"][0],   edgecolor=REPO_COLORS["LAT"][1],   label="LAT — central LISA library (L2 wrapper owner)"),
        mpatches.Patch(facecolor=REPO_COLORS["GBGPU"][0], edgecolor=REPO_COLORS["GBGPU"][1], label="GBGPU — GB sources"),
        mpatches.Patch(facecolor=REPO_COLORS["BBHx"][0],  edgecolor=REPO_COLORS["BBHx"][1],  label="BBHx — SOBBH + MBH sources"),
        mpatches.Patch(facecolor=REPO_COLORS["FEW"][0],   edgecolor=REPO_COLORS["FEW"][1],   label="FEW — EMRI (standalone, own cuda_complex)"),
        mpatches.Patch(facecolor=REPO_COLORS["Eryn"][0],  edgecolor=REPO_COLORS["Eryn"][1],  label="Eryn — sampler"),
    ]
    # Place the legend in a dedicated band between the subtitle and the
    # column-header strip (y in data coords ≈ 15.8). Axes range is
    # (-1.0, 17.4) → axes-fraction y = (15.8 - (-1.0)) / (17.4 - (-1.0)) ≈ 0.913.
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.913),
        fontsize=8.5,
        frameon=True,
        ncol=6,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Full filtered file tree (Graphviz DOT)
# ---------------------------------------------------------------------------


@dataclass
class Node:
    nid: str
    label: str
    is_dir: bool
    children: list["Node"] = field(default_factory=list)


def build_tree(root: Path) -> Node:
    counter = {"n": 0}

    def make_id() -> str:
        counter["n"] += 1
        return f"n{counter['n']}"

    def walk(p: Path) -> Node:
        node = Node(nid=make_id(), label=p.name or str(p), is_dir=p.is_dir())
        if p.is_dir():
            try:
                entries = sorted(p.iterdir(), key=lambda c: (not c.is_dir(), c.name.lower()))
            except PermissionError:
                return node
            for child in entries:
                if not keep(child):
                    continue
                node.children.append(walk(child))
        return node

    return walk(root)


def render_dot(root: Node, out_path: Path) -> None:
    lines: list[str] = []
    lines.append("digraph sprint {")
    lines.append('  graph [rankdir=LR, fontname="Helvetica", nodesep=0.18, ranksep=0.55, concentrate=false];')
    lines.append('  node  [fontname="Helvetica", fontsize=9, shape=box, style="rounded,filled", margin="0.08,0.04"];')
    lines.append('  edge  [color="#888", arrowsize=0.6];')

    def emit(node: Node) -> None:
        if node.is_dir:
            color = "#cfe5cf" if node is root else "#dfe9f5"
            shape = "folder"
        else:
            ext = Path(node.label).suffix.lower()
            shape = "note"
            color = {
                ".py": "#fff2cc",
                ".cpp": "#ffe0d0",
                ".cu": "#ffe0d0",
                ".hh": "#ffe0d0",
                ".hpp": "#ffe0d0",
                ".h": "#ffe0d0",
                ".cxx": "#ffe0d0",
                ".md": "#e8e8e8",
                ".txt": "#f4f4f4",
                ".cmake": "#f0e0ff",
                ".sh": "#e0f0ff",
                ".toml": "#fdf6e3",
                ".yml": "#fdf6e3",
                ".yaml": "#fdf6e3",
            }.get(ext, "#ffffff")
        label = node.label.replace('"', '\\"')
        lines.append(f'  {node.nid} [label="{label}", shape={shape}, fillcolor="{color}"];')
        for child in node.children:
            emit(child)
            lines.append(f"  {node.nid} -> {child.nid};")

    emit(root)
    lines.append("}")
    out_path.write_text("\n".join(lines) + "\n")


def try_render_svg(dot_path: Path) -> Path | None:
    if shutil.which("dot") is None:
        return None
    svg_path = dot_path.with_suffix(".svg")
    subprocess.run(
        ["dot", "-Tsvg", str(dot_path), "-o", str(svg_path)],
        check=True,
    )
    return svg_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print(f"sprint root: {SPRINT_ROOT}")

    arch_png = SPRINT_ROOT / "sprint_architecture.png"
    render_architecture(arch_png)
    print(f"wrote {arch_png.relative_to(SPRINT_ROOT)}")

    tree = build_tree(SPRINT_ROOT)
    dot_path = SPRINT_ROOT / "sprint_filetree.dot"
    render_dot(tree, dot_path)
    print(f"wrote {dot_path.relative_to(SPRINT_ROOT)}")

    svg = try_render_svg(dot_path)
    if svg is None:
        print("graphviz `dot` not on PATH — skipped SVG render "
              "(install with: brew install graphviz)")
    else:
        print(f"wrote {svg.relative_to(SPRINT_ROOT)}")


if __name__ == "__main__":
    main()

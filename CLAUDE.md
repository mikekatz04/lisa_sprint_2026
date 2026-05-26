# Sprint root — guidance for Claude

This is the umbrella directory for the LISA sprint 2026 work. Multiple git
repos sit under here (`lisa-on-gpu`, `LISAanalysistools`, `GPUBackendTools`,
`BBHx`, `FastEMRIWaveforms`, `GBGPU`, `Eryn`). The rule below applies to
all of them — when a sub-repo has its own `CLAUDE.md`, it should also
state this rule so it remains visible when that repo is opened
standalone.

## Backend implementation hierarchy

When implementing or modifying an algorithm that exists across multiple
backends (GPU C++ / CPU C++ / JAX), follow this hierarchy:

1. **GPU C++ (CUDA) leads.** This is the canonical performance target
   and reference implementation. New algorithms and optimizations are
   designed for the GPU first; CPU and JAX paths follow. When the GPU
   C++ changes, the other backends should be updated to match.

2. **CPU C++ mirrors GPU C++ as closely as possible.** Same kernel
   structure, same algorithm, same data flow — use `#ifdef __CUDACC__`
   or shared compile-time macros (e.g. `CUDA_SHARED`, `THREAD_START`,
   `BLOCK_INCR`) to bridge differences. The CPU path exists primarily
   for testing, validation, and use in CPU-only environments. It must
   not diverge in algorithm or output beyond floating-point order of
   operations.

3. **CPU C++ must reproduce the overall lisatools computation.** When
   validated against the lisatools reference (e.g. `FDSignal.transform`,
   `TDSignal.transform`, `XYZ2SensitivityMatrix`), the CPU C++ output
   should match to machine precision (≤ 1e-15 mismatch) in direct
   modes. Cache/approximation modes have explicit error budgets
   documented per-feature.

4. **JAX may diverge internally** — design it to be as JAX-efficient as
   possible. JAX-CPU and JAX-GPU compilation targets are allowed to
   differ if it improves efficiency on each. JAX should NOT
   mechanically translate the C++ kernel structure (shared memory,
   block tiling, register caches); instead it should use JAX-native
   idioms:
   - `jax.lax.scan` for outer loops, `jax.vmap` for inner batched work
   - `jax.lax.dynamic_slice` / `dynamic_update_slice` with **static
     shapes** (mask out variable regions instead of slicing them)
   - Functional accumulator carries rather than in-place buffers
   - Host-precompute anything that needs `scipy.special` / iterative
     solvers JAX lacks

5. **JAX must match C++ inner-product outputs.** The end-to-end
   likelihood quantities (`<d|h>`, `<h|h>`, the 5 swap_ll terms) must
   match the CPU/GPU C++ to floating-point precision (reldiff ≲ 1e-12)
   on representative test cases. Intermediate quantities (raw
   templates, per-chunk WDM coefficients) may differ at FP precision
   due to summation order — this is acceptable, validate at the
   inner-product level.

**Workflow for a new feature.** Implement on GPU C++ first → mirror to
CPU C++ via `#ifdef` → port to JAX with JAX-native idioms → validate
each backend against the others at the inner-product level before
merging.


## No backend strings as function kwargs (sprint-wide rule)

When a class or function needs a compute backend (CPU C++, CUDA C++,
JAX, ...), the backend choice MUST be made at object instantiation,
not as a keyword argument on individual methods. Concretely:

- **Allowed:** ``MyClass(force_backend="cpu")`` /
  ``MyClass(force_backend="cuda12x")`` / ``MyClass(force_backend="jax")``.
  Construct subclasses of :class:`FastLISAResponseParallelModule` /
  :class:`LISAToolsParallelModule` etc. and use ``self.backend`` /
  ``self.backend.xp`` / ``self.backend.name`` to dispatch internally.
- **Forbidden in method signatures:** ``backend="jax"`` / ``backend="cpp"`` /
  ``use_cpp=True`` / ``use_jax=True``. Method names MAY carry a backend
  suffix (e.g. ``get_ll_grad_jax``) when the implementation is
  intrinsically tied to that backend, but the choice of which method
  to call belongs to the caller -- not a runtime kwarg.

Rationale: one instance = one backend, so its arrays / kernels /
dispatchers stay consistent. Mixing backends per call leads to
ambiguous ownership of ``xp`` arrays, surprise host↔device copies, and
brittle dispatch logic.

This rule applies to **this folder and all sub-repos** (lisa-on-gpu,
LISAanalysistools, GPUBackendTools, BBHx, FastEMRIWaveforms, GBGPU,
Eryn, plus any repo-root scripts in the sprint tree).


## Narrowband mismatches mm2 / mm5 (chunked-het / WDM validation)

When verifying a chunked-heterodyne or other narrowband WDM template
against a lisatools reference signal, the canonical narrowband
mismatches are:

- **`mm5`** -- "5-layer" mismatch over a 5-m-layer band around the
  carrier `f0`. The band is defined by frequency bounds
  `[f0 - 3*layer_df, f0 + 2*layer_df]` (slightly asymmetric to cover
  the spectral tails on the side where the WDM transform spreads). Use
  this as the **primary** chunked-het accuracy metric -- it captures
  the dominant carrier + first-neighbour m-layers.

- **`mm2`** -- "2-layer" mismatch over just `m_floor` and `m_floor + 1`
  (the two layers that hold the bulk of a near-monochromatic GB
  signal). Band bounds: `[(m_floor - 0.5)*layer_df,
  (m_floor + 1 + 0.5)*layer_df]`. Use this as a tighter check
  isolating the carrier itself; it strips away spectral-tail
  contributions.

Both are **`1 - normalized overlap`**:

```python
mm = 1 - <d|h> / sqrt(<d|d> <h|h>)
```

via `AnalysisContainer.template_inner_product(..., normalize=True)`,
after slicing both `data` and `template` to the same narrow band by
building a per-binary `WDMSettings(min_freq=..., max_freq=...)` and
reusing the parent grid for layer-index alignment.

The canonical implementation lives in
`gb_chunked_prior_draws.py:283-340` (the `mm5` and `mm2` blocks).
SOBBH and other source-class versions should mirror the same band
definition for direct cross-source comparison.

Acceptance thresholds (current chunked-het with N_cp_sig=48,
N_cp_orbit=32, half-day wavelets, full angular prior):
- median mm5 ~ 1e-9, 90% < 8e-9, 99% < 3e-7
- low-frequency (m_floor < 100) sources occasionally show mm5 ~ 1e-7
  due to spectral-tail extension below ind_min_f -- documented
  systematic, not a bug.


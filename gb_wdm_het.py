"""Python wrapper / fallback for the chunked-heterodyne WDM GB kernels.

This module gives the user a single Python entry point for the
heterodyne WDM workflow we built and validated in
``check_shortened_wdm.py``:

* :func:`compute_chunk_geometry` -- pre-builds the per-chunk
  ``starts / keep_lo / keep_hi / n_global_lo`` arrays handed to the
  kernels (works for both aligned and partial-slide tilings).
* :func:`compute_wdm_window` -- phitilde samples expected by
  ``gb_chunk_fd_to_wdm`` on the host. Hands a Nt_sub-long array off to
  the C++ kernel.
* :func:`group_binaries_by_layer` -- clusters binaries by their carrier
  WDM layer so each kernel launch's shared PSD/data slab amortises over
  binaries in the same band.
* :class:`GBWDMHeterodyne` -- main user-facing class. Constructor takes
  the WDM grid + obs setup; methods ``fill_global``, ``get_ll``,
  ``swap_ll`` follow the same call signatures as the existing
  ``GBComputationGroup`` API.

Until the pybind binding for the new C++ kernels is wired
(``lisa-on-gpu/src/fastlisaresponse/cutils/TDIonthefly.cu``:
``gb_wdm_het_fill_global_kernel`` / ``gb_wdm_het_get_ll_kernel`` /
``gb_wdm_het_swap_ll_kernel``), the wrapper uses a pure-Python fallback
backed by ``check_shortened_wdm._stitched_wdm_from_heterodyne`` /
``chunked_get_ll_python_reference``. Switching to the native path
should be a single ``backend.gb_wdm_het_*_wrap(...)`` call replacement
inside each method.
"""
from __future__ import annotations

import numpy as np

# Reuse the validated chunked-heterodyne primitives from the test script.
from check_shortened_wdm import (
    CachedHeterodyneGenerator,
    USE_RECOMMENDED_TUKEY,
    chunked_get_ll_python_reference,
    group_binaries_by_layer,
    recommended_tukey_alpha,
)
from lisatools.detector import EqualArmlengthOrbits
from lisatools.domains import FDSettings, FDSignal, WDMSettings
from fastlisaresponse.tdiconfig import TDIConfig


# ---------------------------------------------------------------------------
# Host helpers (mirrored on the C++ side at kernel-launch time)
# ---------------------------------------------------------------------------

def compute_chunk_geometry(Nt, Nt_sub, n_pad):
    """Pre-compute per-chunk stitching geometry for the heterodyne kernels.

    Returns a dict of numpy arrays:

      ``starts``           (n_chunks,) int   -- chunk start in WDM-pixel units
      ``keep_lo``          (n_chunks,) int   -- chunk-local lo to keep
      ``keep_hi``          (n_chunks,) int   -- chunk-local hi (exclusive)
      ``n_global_lo``      (n_chunks,) int   -- global WDM pixel where the
                                                first kept sample lands

    Handles the partial-slide case (when ``(Nt - Nt_sub) % step != 0``):
    appends one extra chunk at ``Nt - Nt_sub`` and adjusts the previous
    chunk's ``keep_hi`` so the overlap region isn't double-counted.

    Mirrors the Python implementation in
    ``check_shortened_wdm._stitched_wdm_partial_slide``.
    """
    step = int(Nt_sub) - 2 * int(n_pad)
    assert step > 0 and step % 2 == 0, (Nt_sub, n_pad, step)
    n_full = (int(Nt) - int(Nt_sub)) // step + 1
    starts = [j * step for j in range(n_full)]
    last_full_end = starts[-1] + Nt_sub

    keep_lo = []
    keep_hi = []
    n_global_lo = []
    for j, n0 in enumerate(starts):
        klo = 0 if j == 0 else n_pad
        if j == n_full - 1 and last_full_end == Nt:
            khi = Nt_sub
        else:
            khi = Nt_sub - n_pad
        keep_lo.append(klo)
        keep_hi.append(khi)
        n_global_lo.append(n0 + klo)

    if last_full_end < Nt:
        n0_partial = Nt - Nt_sub
        new_lo_global = starts[-1] + (Nt_sub - n_pad)
        new_lo_local = new_lo_global - n0_partial
        starts.append(n0_partial)
        keep_lo.append(new_lo_local)
        keep_hi.append(Nt_sub)
        n_global_lo.append(new_lo_global)

    return dict(
        starts      = np.asarray(starts,      dtype=np.int64),
        keep_lo     = np.asarray(keep_lo,     dtype=np.int32),
        keep_hi     = np.asarray(keep_hi,     dtype=np.int32),
        n_global_lo = np.asarray(n_global_lo, dtype=np.int32),
    )


def compute_wdm_window(Nf, Nt_sub, dt, backend="cpu"):
    """Return the Nt_sub-long phitilde sampled at the chunk's grid.

    Equivalent to ``WDMSettings(Nf, Nt_sub, dt).window`` -- exposed here
    as a free function so the kernel host code can call it without
    constructing a full settings object.
    """
    return np.asarray(
        WDMSettings(Nf=int(Nf), Nt=int(Nt_sub), dt=float(dt),
                    force_backend=backend).window,
        dtype=float,
    )


# ---------------------------------------------------------------------------
# GPU grid-sizing for the chunked-heterodyne kernels
# ---------------------------------------------------------------------------

# Per-arch resource limits relevant for occupancy on the chunked-het kernels.
#   max_shared_per_sm_bytes : usable shared-mem budget per SM (the "shared"
#                             carveout of the unified L1+shared block; opt-in
#                             via cudaFuncSetAttribute when > 48 KB).
#   max_threads_per_sm      : max concurrent threads per SM.
#   max_blocks_per_sm       : hard cap on concurrent blocks per SM.
#   num_sms                 : SM count on a representative variant of the
#                             card (SXM5 for A100/H100; PCIe variants have
#                             fewer SMs).
#   max_shared_per_block    : opt-in upper bound for static-or-dynamic shared
#                             per block (used to raise an error if the kernel
#                             genuinely won't fit at all).
_GPU_ARCH_LIMITS = {
    "A100": dict(
        max_shared_per_sm_bytes=164 * 1024,
        max_threads_per_sm=2048,
        max_blocks_per_sm=32,
        num_sms=108,
        max_shared_per_block=163 * 1024,
    ),
    "H100": dict(
        max_shared_per_sm_bytes=228 * 1024,
        max_threads_per_sm=2048,
        max_blocks_per_sm=32,
        num_sms=132,
        max_shared_per_block=227 * 1024,
    ),
}


def chunked_het_grid_dim(
    gpu_arch,
    n_chunks,
    shared_per_block_bytes,
    threads_per_block,
    latency_hide_factor=2,
):
    """Pick (grid_dim_x, resident_blocks_per_sm) for the chunked-het kernels.

    Maximizes per-SM occupancy under the joint shared-mem + threads-per-SM
    constraints, then sizes the grid so the runtime always has at least
    ``latency_hide_factor`` blocks queued ahead of each SM. Caps the grid at
    ``n_chunks`` so different blocks process different chunks (no double
    work on fill_global's per-pixel write).

    Args:
        gpu_arch: ``"A100"`` or ``"H100"``. Raises ``ValueError`` otherwise.
        n_chunks: number of chunks the kernel will iterate over.
        shared_per_block_bytes: static + dynamic shared-mem per block, in bytes.
        threads_per_block: NUM_THREADS used by the kernel launch.
        latency_hide_factor: launch this many times the resident block count
            per SM so the scheduler always has a ready block to start when one
            retires. Default 2 -- conservative; 1 leaves no slack, 4+ wastes
            per-block launch overhead.

    Returns:
        dict with keys
          ``grid_dim``                 (int) -- the launch grid size,
          ``resident_blocks_per_sm``   (int) -- occupancy-limited blocks/SM,
          ``max_blocks_per_sm_shared`` (int) -- shared-mem-limited blocks/SM,
          ``max_blocks_per_sm_threads``(int) -- threads-limited blocks/SM,
          ``num_sms``                  (int),
          ``threads_per_sm_resident``  (int) -- threads * resident blocks,
          ``theoretical_occupancy``    (float in [0,1]) -- threads_per_sm_resident
                                                          / max_threads_per_sm,
          ``shared_headroom_bytes``    (int) -- spare shared-mem at the chosen
                                                resident count; negative if the
                                                kernel doesn't fit on this arch.

    Raises:
        ValueError: unknown ``gpu_arch`` or kernel doesn't fit at all.
    """
    arch = str(gpu_arch).upper()
    if arch not in _GPU_ARCH_LIMITS:
        raise ValueError(
            f"Unsupported gpu_arch={gpu_arch!r}; supported: "
            f"{sorted(_GPU_ARCH_LIMITS)}"
        )
    L = _GPU_ARCH_LIMITS[arch]

    s_per_block = int(shared_per_block_bytes)
    t_per_block = int(threads_per_block)
    if s_per_block > L["max_shared_per_block"]:
        raise ValueError(
            f"Kernel needs {s_per_block / 1024:.1f} KB shared per block; "
            f"{arch} max per block is {L['max_shared_per_block'] / 1024:.1f} KB. "
            f"Reduce N_sparse or move more buffers to heap."
        )
    if t_per_block > 1024:
        raise ValueError(
            f"threads_per_block={t_per_block} exceeds CUDA hard cap of 1024."
        )

    max_blocks_shared = (
        L["max_shared_per_sm_bytes"] // s_per_block if s_per_block > 0 else L["max_blocks_per_sm"]
    )
    max_blocks_threads = L["max_threads_per_sm"] // t_per_block

    resident = min(max_blocks_shared, max_blocks_threads, L["max_blocks_per_sm"])
    if resident < 1:
        raise ValueError(
            f"Kernel does not fit one resident block on {arch} "
            f"(shared={s_per_block/1024:.1f} KB, threads={t_per_block})."
        )

    target_grid = resident * L["num_sms"] * int(latency_hide_factor)
    grid_dim = min(int(n_chunks), target_grid)
    if grid_dim < 1:
        grid_dim = 1

    threads_per_sm_resident = resident * t_per_block
    headroom = L["max_shared_per_sm_bytes"] - resident * s_per_block

    return dict(
        grid_dim=int(grid_dim),
        resident_blocks_per_sm=int(resident),
        max_blocks_per_sm_shared=int(max_blocks_shared),
        max_blocks_per_sm_threads=int(max_blocks_threads),
        num_sms=int(L["num_sms"]),
        threads_per_sm_resident=int(threads_per_sm_resident),
        theoretical_occupancy=float(threads_per_sm_resident) / float(L["max_threads_per_sm"]),
        shared_headroom_bytes=int(headroom),
    )


# ---------------------------------------------------------------------------
# Main wrapper class
# ---------------------------------------------------------------------------

class GBWDMHeterodyne:
    """High-level Python entry point for the chunked-heterodyne WDM GB kernels.

    Constructor sets up the WDM grid, chunk geometry, phitilde window,
    and a cached :class:`CachedHeterodyneGenerator`. Methods
    ``fill_global``, ``get_ll``, ``swap_ll`` mirror the existing
    ``GBComputationGroup`` API but route through the heterodyne path.

    Args:
        Nf, Nt: WDM grid dimensions of the global template.
        dt: TD sample step (seconds).
        T_full, t_ref_full: full observation duration + source-phase
            reference time.
        Nt_sub: per-chunk WDM time pixels.
        n_pad: WDM pixels discarded at each chunk edge during stitching.
        N_sparse: heterodyne FFT length per chunk; must be <= 256
            (FAST_WDM_N_SPARSE_MAX in the C++ kernel).
        tukey_alpha: explicit Tukey alpha, or ``USE_RECOMMENDED_TUKEY``
            (default) to auto-pick.
        use_tukey: if False, force alpha = 0.
        nchannels: 3 for XYZ.
        backend: ``"cpu"``, ``"cuda12x"``, etc. The C++ backend is
            selected automatically; ``"cpu"`` falls back to the pure
            Python implementation here.
    """
    def __init__(self, Nf, Nt, dt, T_full, t_ref_full,
                 Nt_sub=256, n_pad=32, N_sparse=256,
                 tukey_alpha=USE_RECOMMENDED_TUKEY, use_tukey=True,
                 nchannels=3, backend="cpu",
                 tdi_gen="2nd generation",
                 orbits=None,
                 t_obs_start=0.0):
        self.Nf       = int(Nf)
        self.Nt       = int(Nt)
        self.dt       = float(dt)
        self.T_full   = float(T_full)
        self.t_ref_full = float(t_ref_full)
        # Absolute time at which the observation begins. WDM pixel 0
        # corresponds to absolute time ``t_obs_start``; chunk j starts at
        # ``t_obs_start + n0 * layer_dt``. The CachedHeterodyneGenerator
        # evaluates GBTDIonTheFly at these absolute times, so this MUST
        # match the t_arr range the injection was built on (otherwise
        # the orbits / source phase land at the wrong absolute time and
        # the template comes out uncorrelated with the injection).
        self.t_obs_start = float(t_obs_start)
        self.Nt_sub   = int(Nt_sub)
        self.n_pad    = int(n_pad)
        self.N_sparse = int(N_sparse)
        self.tukey_alpha = float(tukey_alpha)
        self.use_tukey   = bool(use_tukey)
        self.nchannels = int(nchannels)
        self.backend   = backend

        self.T_chunk    = self.Nf * self.Nt_sub * self.dt
        self.chunk_df   = 1.0 / self.T_chunk
        self.layer_df   = 1.0 / (2.0 * self.Nf * self.dt)
        self.n_rfft_chunk = self.Nf * self.Nt_sub // 2 + 1
        self.log2_N_sparse = int(np.log2(self.N_sparse))
        self.log2_Nt_sub   = int(np.log2(self.Nt_sub))
        assert 2 ** self.log2_N_sparse == self.N_sparse
        assert 2 ** self.log2_Nt_sub   == self.Nt_sub

        self.geometry = compute_chunk_geometry(self.Nt, self.Nt_sub, self.n_pad)
        self.wdm_window = compute_wdm_window(self.Nf, self.Nt_sub, self.dt,
                                             backend=backend)

        # Cached heterodyne generator, reused across all binaries and chunks.
        # If the caller didn't supply orbits, default to EqualArmlength
        # (cheap and good enough for most tests); for prior-draw scripts
        # the injection's orbit model (e.g. ESAOrbits) MUST be passed
        # here too or the template won't match.
        if orbits is None:
            orbits = EqualArmlengthOrbits(force_backend=backend)
        self._gb_kwargs = dict(
            tdi_config=TDIConfig(tdi_gen), orbits=orbits,
            tdi_chan="XYZ", force_backend=backend,
        )
        self._gen = CachedHeterodyneGenerator(
            T_window=self.T_chunk, t_ref_source=self.t_ref_full,
            N_sparse=self.N_sparse, dt=self.dt, nchannels=self.nchannels,
            gb_kwargs=self._gb_kwargs,
        )

        # FD/WDM settings reused per chunk.
        self._chunk_fd_set  = FDSettings(self.n_rfft_chunk, self.chunk_df,
                                         force_backend=backend)
        self._chunk_wdm_set = WDMSettings(Nf=self.Nf, Nt=self.Nt_sub,
                                          dt=self.dt, force_backend=backend)

    # ------------------------------------------------------------------
    # Pure-Python fallback implementations
    # ------------------------------------------------------------------

    def _stitched_wdm(self, source_params):
        """Build the stitched WDM template for one binary (host fallback)."""
        Nf, Nt = self.Nf, self.Nt
        layer_dt = Nf * self.dt
        stitched = np.zeros((self.nchannels, Nf, Nt), dtype=float)
        g = self.geometry
        for j, n0 in enumerate(g["starts"]):
            chunk_t_start = self.t_obs_start + float(n0) * layer_dt
            chunk_fd, _ = self._gen.chunk_fd(
                source_params, chunk_t_start, Nf * self.Nt_sub,
                tukey_alpha=self.tukey_alpha, use_tukey=self.use_tukey,
            )
            chunk_wdm = FDSignal(chunk_fd, self._chunk_fd_set).transform(
                self._chunk_wdm_set
            )
            w_chunk = np.asarray(chunk_wdm.arr)
            klo, khi = int(g["keep_lo"][j]), int(g["keep_hi"][j])
            n_lo = int(g["n_global_lo"][j])
            n_hi = n_lo + (khi - klo)
            stitched[:, :, n_lo:n_hi] = w_chunk[:, :, klo:khi]
        return stitched

    def fill_global(self, template_out, params_list, factors=None):
        """Accumulate per-binary stitched WDM templates into ``template_out``.

        Args:
            template_out: ``(nchannels, Nf, Nt)`` float array, the
                accumulator (caller pre-zeros).
            params_list: iterable of 9-tuples (one per binary).
            factors: per-binary multiplicative factors (default = +1).
        """
        if factors is None:
            factors = np.ones(len(params_list), dtype=float)
        for src, f in zip(params_list, factors):
            template_out += float(f) * self._stitched_wdm(src)
        return template_out

    def get_ll(self, data_d, invC, params_list):
        """Per-binary ``<d|h>`` / ``<h|h>`` via chunked heterodyne.

        Args:
            data_d: ``(nchannels, Nf, Nt)`` WDM data.
            invC: ``(nchannels, Nf, Nt)`` inverse-PSD weighting on the
                same grid.
            params_list: iterable of 9-tuples.

        Returns:
            (d_h, h_h) -- numpy arrays of length ``len(params_list)``.
        """
        d_h, h_h = chunked_get_ll_python_reference(
            td_arr=None,    # unused in the fallback (only orbits/params matter)
            full_wdm_arr=np.asarray(data_d),
            wdm_set=None, sens_mat_inv=np.asarray(invC),
            dt=self.dt, Nf=self.Nf, Nt=self.Nt,
            source_params_list=list(params_list),
            t_ref_full=self.t_ref_full, backend=self.backend,
            Nt_sub=self.Nt_sub, n_pad=self.n_pad, N_sparse=self.N_sparse,
            tukey_alpha=self.tukey_alpha, use_tukey=self.use_tukey,
        )
        return d_h, h_h

    def swap_ll(self, data_d, invC, params_add_list, params_remove_list):
        """5-way swap-ll accumulator.

        Returns ``(d_h_add, d_h_remove, add_add, remove_remove,
        add_remove)`` -- each length ``num_bin``.
        """
        num_bin = len(params_add_list)
        d_h_add = np.zeros(num_bin); d_h_rem = np.zeros(num_bin)
        aa = np.zeros(num_bin); rr = np.zeros(num_bin); ar = np.zeros(num_bin)
        for i in range(num_bin):
            w_add = self._stitched_wdm(params_add_list[i])
            w_rem = self._stitched_wdm(params_remove_list[i])
            d_h_add[i] = float(np.sum(data_d * w_add * invC))
            d_h_rem[i] = float(np.sum(data_d * w_rem * invC))
            aa[i]      = float(np.sum(w_add  * w_add  * invC))
            rr[i]      = float(np.sum(w_rem  * w_rem  * invC))
            ar[i]      = float(np.sum(w_add  * w_rem  * invC))
        return d_h_add, d_h_rem, aa, rr, ar


class SOBBHWDMHeterodyne(GBWDMHeterodyne):
    """Stellar-origin BBH variant of :class:`GBWDMHeterodyne`.

    Mirrors the GB pipeline but plugs ``SOBBHTDIonTheFly`` in as the
    source class and uses ``f_low`` (param index 5) as the heterodyne
    carrier. All other knobs (Nt_sub, N_sparse, n_pad, Tukey, chunk
    geometry, partial-slide handling) are unchanged -- the C++ kernel
    bodies in ``TDIonthefly.cu`` take ``GBTDIonTheFly *gb`` and treat
    ``gb->f0_index`` as opaque, so the SOBBH C++ specialisation only
    needs the analogous ``SOBBHTDIonTheFly`` pointer and ``f0_index =
    5`` instead.

    Param order on ``params_list`` (matches ``SOBBHTDIonTheFly``):
        (m1, m2, s1, s2, distance, f_low, phi_c, inc, psi, lam, beta)
    """
    def __init__(self, *args, **kwargs):
        # Override the source class + param index before the parent's
        # CachedHeterodyneGenerator is constructed. We do this by
        # rebuilding self._gen at the end of __init__.
        super().__init__(*args, **kwargs)
        from fastlisaresponse.tdionfly import SOBBHTDIonTheFly
        from check_shortened_wdm import CachedHeterodyneGenerator
        self._gen = CachedHeterodyneGenerator(
            T_window=self.T_chunk, t_ref_source=self.t_ref_full,
            N_sparse=self.N_sparse, dt=self.dt, nchannels=self.nchannels,
            gb_kwargs=self._gb_kwargs,
            source_class=SOBBHTDIonTheFly, n_params=11,
            f0_param_index=5,                                  # f_low
        )


__all__ = [
    "compute_chunk_geometry",
    "compute_wdm_window",
    "group_binaries_by_layer",
    "recommended_tukey_alpha",
    "GBWDMHeterodyne",
    "SOBBHWDMHeterodyne",
]

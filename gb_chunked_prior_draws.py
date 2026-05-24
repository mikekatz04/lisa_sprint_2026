#!/usr/bin/env python
"""Prior-draw mismatch test of the WDM GB CHUNKED-HETERODYNE template.

Drop-in counterpart of ``gb_lookup_prior_draws.py`` that swaps the
WDM-lookup-table template step for the new chunked-heterodyne pipeline
in :mod:`gb_wdm_het` (validated to floating-point precision against the
lisatools WDM transform in ``check_shortened_wdm.py:test_K`` and
``test_L``).

For each draw:

  1. Draw an 8-D GB parameter from the same prior as
     ``gb_lookup_prior_draws.py``; transform to the 9-param physical
     vector.
  2. Build an accurate dense-TD waveform via ``GBTDIonTheFly`` and
     transform it to the WDM domain (injection).
  3. Build the WDM template via
     :meth:`gb_wdm_het.GBWDMHeterodyne.fill_global` (chunked-heterodyne
     + Tukey-auto). NO lookup table.
  4. Compute ``log_like``, full-band, 5-layer, and 2-layer
     mismatches via ``AnalysisContainer.template_likelihood`` /
     ``template_inner_product`` -- identical to the lookup script so
     numbers are directly comparable.
  5. Save NPZ + diagnostic plots with the same column names as the
     lookup script.

Env-var knobs match the lookup script unless explicitly noted.

Run:
    python gb_chunked_prior_draws.py
"""
from __future__ import annotations

import os
import time

import numpy as np
import matplotlib
if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = None

from lisatools.detector import ESAOrbits
from lisatools.utils.constants import YRSID_SI
from fastlisaresponse.tdiconfig import TDIConfig
from fastlisaresponse.tdionfly import GBTDIonTheFly

from lisatools.datacontainer import DataResidualArray
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.domains import (
    TDSettings, TDSignal, FDSettings, WDMSettings, WDMSignal,
)

from gb_lookup_prior_draws import build_gb_prior
from gb_wdm_het import GBWDMHeterodyne


def _make_wdm_signal_slice(full_wdm_arr, parent_set, min_freq, max_freq,
                           min_time, max_time, backend):
    """Slice ``full_wdm_arr`` onto a band/time-cropped ``WDMSettings``."""
    band_set = WDMSettings(
        parent_set.Nf, parent_set.Nt, parent_set.data_dt,
        min_freq=min_freq, max_freq=max_freq,
        min_time=min_time, max_time=max_time,
        force_backend=backend,
    )
    arr_band = full_wdm_arr[
        :,
        band_set.ind_min_f - parent_set.ind_min_f
        : band_set.ind_max_f - parent_set.ind_min_f + 1,
    ]
    return WDMSignal(arr_band, band_set), band_set


def main():
    backend = os.environ.get("CHUNKED_BACKEND", "cpu")
    xp = np if backend == "cpu" else cp

    # --- config (matches gb_lookup_prior_draws.py defaults) ----------------
    N_DRAWS = int(os.environ.get("N_DRAWS", 50))
    SNR_MIN = float(os.environ.get("SNR_MIN", 5.0))
    SNR_MAX = float(os.environ.get("SNR_MAX", 1100.0))
    N_INJ = int(os.environ.get("N_INJ", 16384))
    MAX_REJECT = int(os.environ.get("MAX_REJECT", 500))
    SEED = int(os.environ.get("SEED", 12345))
    OUTPUT_PREFIX = os.environ.get("OUTPUT_PREFIX", "gb_prior_chunked_test")
    PROGRESS_EVERY = int(os.environ.get("PROGRESS_EVERY", 1))

    # --- chunked-heterodyne knobs ------------------------------------------
    Nt_sub = int(os.environ.get("NT_SUB",  256))
    N_sparse = int(os.environ.get("N_SPARSE", 256))
    # n_pad: per-chunk edge discard (in WDM pixels). Recommended fraction is
    # 1/8 of Nt_sub. Phase 1's joint sweep showed mm5/mm2 < 1e-9 at this
    # default with Tukey-auto.
    n_pad = int(os.environ.get("N_PAD", Nt_sub // 8))

    np.random.seed(SEED)

    # --- detector / WDM grid (same as the lookup script) ------------------
    orbits = ESAOrbits(force_backend=backend)
    dt = 10.0
    Nf = int(os.environ.get("NF", 1460))
    Nt = int(os.environ.get("NT", 256 * 10 * (1460 // Nf if Nf <= 1460 else 1)))
    print(f"[run] Nf={Nf} Nt={Nt} dt={dt}  (Tobs={Nf*Nt*dt:.3e}s, "
          f"Nt_sub={Nt_sub}, N_sparse={N_sparse}, n_pad={n_pad})", flush=True)
    wavelet_duration = Nf * dt
    Tobs = Nt * wavelet_duration
    Nobs = Nf * Nt

    tdi_config = TDIConfig("2nd generation")
    t_start = int(0.5 * YRSID_SI / dt) * dt              # 6 months in
    t_arr = np.arange(Nobs) * dt + t_start
    t_ref = t_start

    gb_tdi_kwargs = dict(
        tdi_config=tdi_config, orbits=orbits,
        tdi_chan="XYZ", force_backend=backend,
    )
    t_tdi_inj = xp.linspace(t_arr[0], t_arr[-1], N_INJ)
    gb_gen_inj = GBTDIonTheFly(
        t_tdi_inj, Tobs, t_ref, 1.0 / dt, 1, **gb_tdi_kwargs,
    )

    N = t_arr.shape[-1]
    td_set = TDSettings(N, dt, force_backend=backend)
    freqs = np.fft.rfftfreq(N, dt)
    df = freqs[1] - freqs[0]
    N_fd = len(freqs)

    # WDM active band. We need ind_min_f = 0 to support priors down to
    # f0 ~ 1 mHz (mm5 band runs from m_floor-3 -> needs layer 0 at m_floor=2).
    min_freq = float(os.environ.get("MIN_FREQ_HZ", 0.0))
    max_freq = float(os.environ.get("MAX_FREQ_HZ", 35.0e-3))
    _ = FDSettings(N_fd, df, min_freq=min_freq, max_freq=max_freq,
                   force_backend=backend)
    min_time = 20 * wavelet_duration
    max_time = (Nt - 20) * wavelet_duration

    wdm_set = WDMSettings(
        Nf, Nt, dt,
        min_freq=min_freq, max_freq=max_freq,
        min_time=min_time, max_time=max_time,
        force_backend=backend,
    )

    # --- chunked-heterodyne generator -------------------------------------
    chunked = GBWDMHeterodyne(
        Nf=Nf, Nt=Nt, dt=dt, T_full=Tobs, t_ref_full=t_ref,
        Nt_sub=Nt_sub, n_pad=n_pad, N_sparse=N_sparse,
        backend=backend, tdi_gen="2nd generation",
        orbits=orbits,                                  # MUST match injection
        t_obs_start=float(t_start),                     # MUST match t_arr[0]
    )
    print(f"[run] chunked: n_chunks={len(chunked.geometry['starts'])}, "
          f"T_chunk={chunked.T_chunk:.3e}s, alpha={chunked.tukey_alpha}, "
          f"use_tukey={chunked.use_tukey}", flush=True)

    # --- prior (same as the lookup script) --------------------------------
    layer_df = wdm_set.layer_df
    buffer_layers = 7                                    # ~5 main + 2 margin
    f0_lo_default = (wdm_set.ind_min_f + buffer_layers) * layer_df
    f0_hi_default = (wdm_set.ind_max_f - buffer_layers) * layer_df
    f0_lo_hz = float(os.environ.get("F0_LO_HZ", f0_lo_default))
    f0_hi_hz = float(os.environ.get("F0_HI_HZ", f0_hi_default))
    fdot_max = float(os.environ.get("FDOT_MAX", 1e-15))
    A_lims = (float(os.environ.get("A_LO", 1e-23)),
              float(os.environ.get("A_HI", 1e-20)))

    beta_env = os.environ.get("BETA_LIMS", "")
    if beta_env:
        beta_lims = tuple(float(s) for s in beta_env.split(","))
    else:
        beta_lims = None

    prior, tc = build_gb_prior(
        A_lims=A_lims, f0_lims_hz=(f0_lo_hz, f0_hi_hz),
        fdot_lims=(-fdot_max, fdot_max), beta_lims=beta_lims,
    )

    print(f"[run] N_DRAWS={N_DRAWS} SNR window=[{SNR_MIN}, {SNR_MAX}]", flush=True)
    print(f"[run] f0 range = [{f0_lo_hz*1e3:.4f}, {f0_hi_hz*1e3:.4f}] mHz "
          f"(layer_df = {layer_df:.3e} Hz)", flush=True)

    # --- per-draw loop ----------------------------------------------------
    sens_mat = None
    snr_list, log_like_list, mismatch_list = [], [], []
    log_like_5_layers_list, mismatch_5_layers_list = [], []
    log_like_2_layers_list, mismatch_2_layers_list = [], []
    params_list = []
    attempt_total = 0
    t_loop_start = time.perf_counter()

    for i in range(N_DRAWS):
        chosen = None
        for _ in range(MAX_REJECT):
            attempt_total += 1
            x_sampled = prior.rvs(size=1)
            params_i = tc.both_transforms(x_sampled.copy())[0]

            inj_spline = gb_gen_inj(
                *params_i.reshape(9, 1),
                convert_to_ra_dec=False, return_spline=True,
            )
            td_inj = np.asarray(inj_spline.eval_tdi(t_arr))[0]
            wdm_inj_sig = TDSignal(td_inj, settings=td_set).transform(
                wdm_set, window=None,
            )
            injection = DataResidualArray(wdm_inj_sig)

            if sens_mat is None:
                sens_mat = XYZ2SensitivityMatrix(
                    injection.data_res_arr.settings, model="scirdv1",
                )

            analysis = AnalysisContainer(injection, sens_mat)
            d_d = float(np.real(analysis.inner_product()))
            snr = float(analysis.snr())
            if SNR_MIN <= snr <= SNR_MAX:
                chosen = (params_i, wdm_inj_sig, analysis, d_d, snr)
                break

        if chosen is None:
            print(f"[warn] draw {i}: exhausted {MAX_REJECT} attempts; "
                  f"keeping last (snr={snr:.2f})", flush=True)
            chosen = (params_i, wdm_inj_sig, analysis, d_d, snr)
        params_i, wdm_inj_sig, analysis, d_d, snr = chosen

        # --- CHUNKED-HETERODYNE TEMPLATE (Phase 4 swap-in) ---------------
        # injection is on the full WDM grid (3, Nf_active, Nt_active).
        # The chunked generator builds the template on the same FULL grid
        # (3, Nf, Nt). For the full-band mismatch we have to mask the
        # template down to the active band to match the injection.
        template_full = np.zeros((3, Nf, Nt), dtype=float)
        chunked.fill_global(
            template_full, [tuple(params_i.tolist())], factors=None,
        )
        # crop to the active band that matches the injection's WDMSettings
        tpl_active = template_full[
            :, wdm_set.ind_min_f: wdm_set.ind_max_f + 1, :
        ]
        # Carry over the active-time slicing from wdm_set if any
        if wdm_set.Nt_active != wdm_set.Nt:
            tpl_active = tpl_active[:, :, wdm_set.active_slice_t]
        tpl_wdm = WDMSignal(tpl_active, wdm_set)

        # --- full-band log_like + mismatch (same as lookup script) -------
        log_like = analysis.template_likelihood(tpl_wdm)
        mismatch = analysis.template_inner_product(tpl_wdm, normalize=True)
        snr_list.append(snr)
        log_like_list.append(log_like)
        mismatch_list.append(mismatch)
        params_list.append(params_i)

        # --- 5-layer mismatch -------------------------------------------
        f0 = float(params_i[1])
        m_floor = int(f0 / layer_df)
        new_wdm_set = WDMSettings(
            wdm_set.Nf, wdm_set.Nt, wdm_set.data_dt,
            min_time=wdm_set.min_time, max_time=wdm_set.max_time,
            min_freq=f0 - 3 * wdm_set.layer_df,
            max_freq=f0 + 2 * wdm_set.layer_df,
            force_backend=backend,
        )
        wdm_inj_arr = np.asarray(wdm_inj_sig.arr)
        inj_band = WDMSignal(
            wdm_inj_arr[:,
                new_wdm_set.ind_min_f - wdm_set.ind_min_f
                : new_wdm_set.ind_max_f - wdm_set.ind_min_f + 1],
            new_wdm_set,
        )
        tpl_band = WDMSignal(
            tpl_active[:,
                new_wdm_set.ind_min_f - wdm_set.ind_min_f
                : new_wdm_set.ind_max_f - wdm_set.ind_min_f + 1],
            new_wdm_set,
        )
        analysis_5 = AnalysisContainer(
            DataResidualArray(inj_band),
            XYZ2SensitivityMatrix(new_wdm_set, model="scirdv1"),
        )
        log_like_5 = analysis_5.template_likelihood(DataResidualArray(tpl_band))
        mm5 = 1.0 - analysis_5.template_inner_product(
            DataResidualArray(tpl_band), normalize=True,
        )
        log_like_5_layers_list.append(log_like_5)
        mismatch_5_layers_list.append(mm5)

        # --- 2-layer mismatch (m_floor, m_floor+1) -----------------------
        new_wdm_set_2 = WDMSettings(
            wdm_set.Nf, wdm_set.Nt, wdm_set.data_dt,
            min_time=wdm_set.min_time, max_time=wdm_set.max_time,
            min_freq=(m_floor - 0.5) * wdm_set.layer_df,
            max_freq=(m_floor + 1 + 0.5) * wdm_set.layer_df,
            force_backend=backend,
        )
        inj_2 = WDMSignal(
            wdm_inj_arr[:,
                new_wdm_set_2.ind_min_f - wdm_set.ind_min_f
                : new_wdm_set_2.ind_max_f - wdm_set.ind_min_f + 1],
            new_wdm_set_2,
        )
        tpl_2 = WDMSignal(
            tpl_active[:,
                new_wdm_set_2.ind_min_f - wdm_set.ind_min_f
                : new_wdm_set_2.ind_max_f - wdm_set.ind_min_f + 1],
            new_wdm_set_2,
        )
        analysis_2 = AnalysisContainer(
            DataResidualArray(inj_2),
            XYZ2SensitivityMatrix(new_wdm_set_2, model="scirdv1"),
        )
        log_like_2 = analysis_2.template_likelihood(DataResidualArray(tpl_2))
        mm2 = 1.0 - analysis_2.template_inner_product(
            DataResidualArray(tpl_2), normalize=True,
        )
        log_like_2_layers_list.append(log_like_2)
        mismatch_2_layers_list.append(mm2)

        if (i + 1) % PROGRESS_EVERY == 0 or i == 0:
            elapsed = time.perf_counter() - t_loop_start
            rate = (i + 1) / max(elapsed, 1e-9)
            f_frac = (f0 - m_floor * layer_df) / layer_df
            print(
                f"  [{i+1:4d}/{N_DRAWS}] snr={snr:7.2f}  logL={log_like:+.3e} "
                f"1-O={mismatch:.3e}  m={m_floor:4d} f_frac={f_frac:.3f}\n"
                f"     mm5={mm5:.3e}   mm2={mm2:.3e}   "
                f"({attempt_total} att, {rate:.2f} draw/s, {elapsed:.1f}s)",
                flush=True,
            )

    # --- save NPZ + summary plots -----------------------------------------
    out_npz = f"{OUTPUT_PREFIX}_{int(time.time())}.npz"
    np.savez(
        out_npz,
        params=np.asarray(params_list),
        snr=np.asarray(snr_list),
        log_like=np.asarray(log_like_list),
        mismatch=np.asarray(mismatch_list),
        log_like_5=np.asarray(log_like_5_layers_list),
        mismatch_5=np.asarray(mismatch_5_layers_list),
        log_like_2=np.asarray(log_like_2_layers_list),
        mismatch_2=np.asarray(mismatch_2_layers_list),
        Nf=Nf, Nt=Nt, dt=dt, Nt_sub=Nt_sub, N_sparse=N_sparse, n_pad=n_pad,
    )
    print(f"[save] wrote {out_npz}", flush=True)

    # log-histograms of the three mismatches
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, arr, title in zip(
        axes,
        [mismatch_list, mismatch_5_layers_list, mismatch_2_layers_list],
        ["full-band", "5-layer", "2-layer"],
    ):
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr) & (arr > 0)]
        if len(arr) == 0:
            ax.set_title(title + " (no positive values)")
            continue
        ax.hist(np.log10(arr), bins=40, color="steelblue", alpha=0.8)
        ax.set_xlabel("log10(1 - O)")
        ax.set_ylabel("count")
        ax.set_title(f"{title} (N={len(arr)}, median={np.median(arr):.2e})")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_PREFIX}_hist.png", dpi=120)
    plt.close(fig)
    print(f"[plot] wrote {OUTPUT_PREFIX}_hist.png", flush=True)


if __name__ == "__main__":
    main()

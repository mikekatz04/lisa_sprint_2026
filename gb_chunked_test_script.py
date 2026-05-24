#!/usr/bin/env python
# coding: utf-8

# Drop-in counterpart to gb_lookup_table_test_script.py: all the WDM
# lookup-table machinery is replaced with the chunked-heterodyne template
# pipeline (GBWDMHeterodyne from gb_wdm_het) -- the exact same pipeline
# exercised in gb_chunked_prior_draws.py. Everything else (injection
# generation, settings, analysis containers, MCMC) is kept as close to
# the lookup test script as possible.

import os, sys
import numpy as np
import matplotlib
if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

try:
    import cupy as cp

except (ImportError, ModuleNotFoundError) as e:
    pass

from lisatools.detector import ESAOrbits, EqualArmlengthOrbits
from lisaconstants import ASTRONOMICAL_YEAR
from lisatools.utils.constants import YRSID_SI
from fastlisaresponse import ResponseWrapper
from fastlisaresponse.tdiconfig import TDIConfig
from fastlisaresponse.response import icrs_to_ecliptic
from fastlisaresponse.tdionfly import GBTDIonTheFly

from lisatools.datacontainer import DataResidualArray
from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.domains import TDSettings, TDSignal, FDSettings, FDSignal, WDMSettings, WDMSignal

from eryn.utils import TransformContainer
from eryn.prior import ProbDistContainer, uniform_dist, log_uniform

from eryn.moves import StretchMove
from eryn.ensemble import EnsembleSampler
from eryn.utils import PeriodicContainer

from eryn.state import State
from eryn.backends import HDFBackend

# Chunked-heterodyne template pipeline -- replaces WDMLookupTable /
# GBWDMComputations. Same import path used by gb_chunked_prior_draws.py.
from gb_wdm_het import GBWDMHeterodyne


import time
class GBChunkedWaveWrap:
    """Same shape/contract as the lookup-table version (params (9,)
    scalars -> WDMSignal on wdm_set) but builds the template via
    GBWDMHeterodyne.fill_global instead of WDMLookupTable.get_wdm_coeffs."""
    def __init__(self, chunked, wdm_set, Nf, Nt):
        self.chunked = chunked
        self.wdm_set = wdm_set
        self.Nf = Nf
        self.Nt = Nt

    def __call__(self, *params):
        params_arr = np.asarray(params, dtype=float).reshape(9)
        template_full = np.zeros((3, self.Nf, self.Nt), dtype=float)
        self.chunked.fill_global(
            template_full, [tuple(params_arr.tolist())], factors=None,
        )
        tpl_active = template_full[
            :, self.wdm_set.ind_min_f: self.wdm_set.ind_max_f + 1, :
        ]
        if self.wdm_set.Nt_active != self.wdm_set.Nt:
            tpl_active = tpl_active[:, :, self.wdm_set.active_slice_t]
        return WDMSignal(tpl_active, self.wdm_set)


if __name__ == "__main__":
    backend = "cpu"

    xp = np if backend == "cpu" else cp

    orbits = ESAOrbits(force_backend=backend)
    # ORBIT_DT (s) or ORBIT_LINEAR=1 to configure orbits more densely.
    # Default base orbits use dt_base ~ 1.87 days with 5th-order interp.
    _orbit_dt = os.environ.get("ORBIT_DT", "")
    _orbit_linear = os.environ.get("ORBIT_LINEAR", "0") == "1"
    if _orbit_linear:
        orbits.configure(linear_interp_setup=True)
        print(f"[step] ORBITS: linear_interp_setup=True (dense linear interp)", flush=True)
    elif _orbit_dt.strip():
        orbits.configure(dt=float(_orbit_dt))
        print(f"[step] ORBITS: configured at dt={_orbit_dt}s with cubic spline", flush=True)
    else:
        print(f"[step] ORBITS: base (dt~1.87d, 5th-order interp)", flush=True)
    dt = 10.0  # mojito
    _Tobs = 1. * YRSID_SI
    # NF, NT env overrides -- change wavelet aspect at constant total signal
    # length (Nf*Nt). Defaults preserve current 1460x2560 setup.
    Nf = int(os.environ.get("NF", 1460))
    Nt = int(os.environ.get("NT", 256 * 10))

    wavelet_duration = Nf * dt
    Tobs = Nt * wavelet_duration
    Nobs = Nf * Nt

    tdi_config = TDIConfig('2nd generation')  # mojito

    t_start = int(1 / 2 * YRSID_SI / dt) * dt  # 6 months
    t_arr = np.arange(Nobs) * dt + t_start
    data_inj = np.zeros((3, t_arr.shape[-1]))
    template = np.zeros((3, t_arr.shape[-1]))

    t_ref = t_start

    N_inj = 16384

    gb_tdi_kwargs = dict(
        tdi_config=tdi_config,
        orbits=orbits,
        tdi_chan="XYZ",
        force_backend=backend,
    )
    t_tdi_inj = xp.linspace(t_arr[0], t_arr[-1], N_inj)
    gb_gen_inj = GBTDIonTheFly(
        t_tdi_inj,
        Tobs,
        t_ref,
        1. / dt,
        1,
        **gb_tdi_kwargs
    )

    N = data_inj.shape[-1]
    td_set = TDSettings(N, dt, force_backend=backend)
    freqs = np.fft.rfftfreq(N, dt)
    df = freqs[1] - freqs[0]
    N_fd = len(freqs)
    # TUKEY_ALPHA env var sets a Tukey window on the injection TD->WDM
    # transform. Default 0 (= unity window) matches gb_chunked_prior_draws.py.
    _tukey_alpha = float(os.environ.get("TUKEY_ALPHA", "0.0"))
    if _tukey_alpha > 0:
        window = xp.asarray(signal.windows.tukey(N, alpha=_tukey_alpha))
    else:
        window = xp.ones(N)

    min_freq = 0.0001
    max_freq = 35.0e-3
    fd_set = FDSettings(N_fd, df, min_freq=min_freq, max_freq=max_freq, force_backend=backend)

    # shave edges?
    _EDGE_CUT = int(os.environ.get("EDGE_CUT", "20"))
    min_time = _EDGE_CUT * wavelet_duration
    max_time = (Nt - _EDGE_CUT) * wavelet_duration

    wdm_set = WDMSettings(Nf, Nt, dt, min_freq=min_freq, max_freq=max_freq, min_time=min_time, max_time=max_time)

    # --- chunked-heterodyne generator (replaces lookup-table setup) -------
    # Same construction pattern as gb_chunked_prior_draws.py.
    Nt_sub = int(os.environ.get("NT_SUB", 256))
    N_sparse = int(os.environ.get("N_SPARSE", 256))
    n_pad = int(os.environ.get("N_PAD", Nt_sub // 8))
    chunked = GBWDMHeterodyne(
        Nf=Nf, Nt=Nt, dt=dt, T_full=Tobs, t_ref_full=t_ref,
        Nt_sub=Nt_sub, n_pad=n_pad, N_sparse=N_sparse,
        backend=backend, tdi_gen="2nd generation",
        orbits=orbits,                                  # MUST match injection
        t_obs_start=float(t_start),                     # MUST match t_arr[0]
        N_cp_sig=int(os.environ.get("N_CP_SIG", 0)),
        N_cp_orbit=int(os.environ.get("N_CP_ORBIT", 0)),
    )
    print(f"[step] chunked: n_chunks={len(chunked.geometry['starts'])}, "
          f"T_chunk={chunked.T_chunk:.3e}s, alpha={chunked.tukey_alpha}, "
          f"use_tukey={chunked.use_tukey}", flush=True)

    # F_FRACS sweep: comma-separated f_frac values to test. Defaults span
    # on-grid (~0/1) and mid-grid (~0.5). Set F_FRAC for a single value.
    if "F_FRAC" in os.environ:
        _f_frac_list = [float(os.environ["F_FRAC"])]
    else:
        _f_frac_list = [float(s) for s in os.environ.get("F_FRACS", "0.05,0.5,0.95").split(",")]

    # Shared one-shot construction (does not depend on f_frac).
    output_set = wdm_set
    if output_set != wdm_set:
        raise ValueError("This script requires WDM for the output_set.")
    sens_mat_proto = None  # build once injection settings are known

    print(f"[step] f_fracs={_f_frac_list}", flush=True)

    # Source's m-layer reference -- in-band layer near 3 mHz (the original
    # script's 18 mHz default is outside the WDM band).
    m_ref_source = int(3e-3 / wdm_set.layer_df)

    _results = []
    # M_OFFSET shifts the source's m_floor by an integer relative to m_ref_source.
    _m_offset = int(os.environ.get("M_OFFSET", "0"))
    for _ff_idx, _ff in enumerate(_f_frac_list):
        num_bin = 1
        amp = np.full(num_bin, 8.0e-22)
        f0 = np.full(num_bin, (m_ref_source + _m_offset + _ff) * wdm_set.layer_df)
        # SOURCE_FDOT (Hz/s). Default 1e-17 (essentially fdot=0).
        fdot = np.full(num_bin, float(os.environ.get("SOURCE_FDOT", "1e-17")))
        fddot = np.full(num_bin, 0.0)
        phi0 = np.full(num_bin, 2.09802430298)
        inc = np.full(num_bin, 0.23984234)

        # NEED TO ADD FRAME TRANSFORM FOR PSI IF WORKING IN ECLIPTIC
        psi = np.full(num_bin, 1.234019814)
        lam = np.full(num_bin, 4.09808143)
        beta = np.full(num_bin, float(os.environ.get("BETA", "0.04")))
        params = np.array([amp, f0, fdot, fddot, phi0, inc, psi, lam, beta]).T

        # Fresh injection buffer per iteration.
        _data_inj = xp.zeros_like(data_inj)
        inj_tmp = gb_gen_inj(amp, f0, fdot, fddot, phi0, inc, psi, lam, beta,
                             convert_to_ra_dec=False, return_spline=True)
        _data_inj[:] = inj_tmp.eval_tdi(t_arr)

        data_inj_all = TDSignal(_data_inj, settings=td_set).transform(output_set, window=window)
        injection = DataResidualArray(data_inj_all)
        if sens_mat_proto is None:
            sens_mat_proto = XYZ2SensitivityMatrix(injection.data_res_arr.settings, model="scirdv1")
        sens_mat = sens_mat_proto

        gb_gen_wrap = GBChunkedWaveWrap(chunked, wdm_set, Nf, Nt)
        analysis = AnalysisContainer(injection, sens_mat, signal_gen=gb_gen_wrap)
        wdm_holder = AnalysisContainerArray([analysis])

        # Build template via chunked-het. (No separate Python vs C path
        # like the lookup script; chunked-het is the single C++ template
        # builder.)
        template_fill_wdm = gb_gen_wrap(*params[0])
        check_ll = analysis.template_likelihood(template_fill_wdm)
        check_ip = analysis.template_inner_product(template_fill_wdm)
        overlap = analysis.template_inner_product(template_fill_wdm, normalize=True)

        check_ip_d_d = analysis.inner_product()
        tmp_val2 = analysis.calculate_signal_inner_product(*params[0])
        tmp_val3 = analysis.calculate_signal_likelihood(*params[0], source_only=True)

        _f_frac_meas = (f0[0] - (int(f0[0] / wdm_set.layer_df) * wdm_set.layer_df)) / wdm_set.layer_df
        _mm = float(1.0 - overlap)

        # mm5 -- mismatch restricted to a narrow +-5-layer band around f0.
        _new_wdm_set = WDMSettings(
            wdm_set.Nf, wdm_set.Nt, wdm_set.data_dt,
            min_time=wdm_set.min_time, max_time=wdm_set.max_time,
            min_freq=float(f0[0] - 3 * wdm_set.layer_df),
            max_freq=float(f0[0] + 2 * wdm_set.layer_df),
            force_backend=backend,
        )
        _m_lo = _new_wdm_set.ind_min_f - wdm_set.ind_min_f
        _m_hi = _new_wdm_set.ind_max_f - wdm_set.ind_min_f + 1
        _inj_here = DataResidualArray(WDMSignal(injection[:, _m_lo:_m_hi], _new_wdm_set))
        _sens_here = XYZ2SensitivityMatrix(_new_wdm_set, model="scirdv1")
        _ah5 = AnalysisContainer(_inj_here, _sens_here)
        _tpl_here = DataResidualArray(WDMSignal(template_fill_wdm[:, _m_lo:_m_hi], _new_wdm_set))
        _mm5 = float(1.0 - _ah5.template_inner_product(_tpl_here, normalize=True))

        _results.append((_ff, _f_frac_meas, _mm, _mm5))

        # SAVE_RESIDUAL=1 dumps the (injection - template) WDM array.
        if os.environ.get("SAVE_RESIDUAL", "0") == "1":
            _inj_arr = data_inj_all.arr
            _tpl_arr = template_fill_wdm.arr
            _resid = _inj_arr - _tpl_arr
            _ms = int(f0[0] / wdm_set.layer_df)
            _m_active = _ms - wdm_set.ind_min_f
            np.savez_compressed(
                f"residual_chunked_ff{_ff:.3f}.npz",
                inj=_inj_arr, tpl=_tpl_arr, resid=_resid,
                m_source_active=_m_active, m_source_abs=_ms,
                ind_min_f=wdm_set.ind_min_f, ind_min_t=wdm_set.ind_min_t,
                layer_df=wdm_set.layer_df, layer_dt=wdm_set.layer_dt,
                f0=float(f0[0]),
            )
            _r_per_m = np.sqrt((_resid ** 2).sum(axis=(0, 2)))
            _i_per_m = np.sqrt((_inj_arr ** 2).sum(axis=(0, 2)))
            print(f"  [resid] saved residual_chunked_ff{_ff:.3f}.npz; "
                  f"source at m_abs={_ms} (active {_m_active})")
            print(f"  [resid] per-layer  m_off   ||resid||   ||inj||   ratio")
            for _moff in [-3, -2, -1, 0, 1, 2, 3]:
                _mi = _m_active + _moff
                if 0 <= _mi < _r_per_m.size:
                    print(f"          {_moff:+5d}    {_r_per_m[_mi]:.3e}   "
                          f"{_i_per_m[_mi]:.3e}   "
                          f"{_r_per_m[_mi] / max(_i_per_m[_mi], 1e-30):.3e}")

        print(f"\n=== f_frac sweep [{_ff_idx + 1}/{len(_f_frac_list)}]  "
              f"requested={_ff:.3f}  measured={_f_frac_meas:+.4f}  "
              f"f0={f0[0]*1e3:.5f} mHz ===")
        print(f"[result] base inner_product <d|d>             = {check_ip_d_d}")
        print(f"[result] template_inner_product (chunked-het) = {check_ip}")
        print(f"[result] calculate_signal_inner_product       = {tmp_val2}")
        print(f"[result] template_likelihood (chunked-het)    = {check_ll}")
        print(f"[result] calculate_signal_likelihood          = {tmp_val3}")
        print(f"[result] Noise-weighted mismatch (chunked-het) = {_mm:.3e}")
        print(f"[result] mm5 (+-5-layer band, chunked-het)     = {_mm5:.3e}")

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(11, 6))
        data_inj_all.heatmap(fig=fig, ax=ax1, index=0, add_cax=True)
        template_fill_wdm.heatmap(fig=fig, ax=ax2, index=0)
        ax1.set_title(
            f"injection  f0={f0[0]*1e3:.5f} mHz  phi0={phi0[0]:.4f}  f_frac={_f_frac_meas:+.3f}"
        )
        ax2.set_title(f"chunked-het template  (mm = {_mm:.3e})")
        ax1.set_ylim(f0[0] - 10 * wdm_set.layer_df, f0[0] + 10 * wdm_set.layer_df)
        plt.tight_layout()
        _out_path = f"gb_chunked_ff{_f_frac_meas:+.3f}.png"
        plt.savefig(_out_path, dpi=120)
        if os.environ.get("SHOW_PLOTS", "0") == "1":
            plt.show()
        plt.close(fig)
        print(f"[plot] saved {_out_path}")

    print("\n[summary] f_frac sweep")
    print(f"  Nt_sub={Nt_sub}  N_sparse={N_sparse}  n_pad={n_pad}")
    print("  requested  measured  mm           mm5")
    for _ff, _ff_m, _mm, _mm5 in _results:
        print(f"  {_ff:>9.3f}  {_ff_m:+.4f}   {_mm:.3e}   {_mm5:.3e}")

    if os.environ.get("DEBUG_BREAK", "0") == "1":
        breakpoint()

    # Per-channel AET cross-check.
    from lisatools.sensitivity import AET1SensitivityMatrix
    def xyz_to_aet(arr):
        X, Y, Z = arr[0], arr[1], arr[2]
        A = (Z - X) / np.sqrt(2.0)
        E = (X - 2.0 * Y + Z) / np.sqrt(6.0)
        T = (X + Y + Z) / np.sqrt(3.0)
        return np.stack([A, E, T], axis=0)
    inj_aet = xyz_to_aet(injection.data_res_arr.arr)
    tpl_aet = xyz_to_aet(template_fill_wdm.arr)
    sens_aet = AET1SensitivityMatrix(WDMSignal(inj_aet, wdm_set).settings)
    invC = sens_aet.invC
    prefactor = 4.0 * sens_aet.differential_component
    print(f"==== per-channel AET inner products ====")
    print(f"{'chan':6s} {'<d|d>':>12s} {'<d|h>':>12s} {'<d|h>/<d|d>':>14s}")
    for chan, name in enumerate("AET"):
        d_c = inj_aet[chan]; h = tpl_aet[chan]; ic = invC[chan]
        dd = (d_c * d_c * ic).sum() * prefactor
        dh = (d_c * h * ic).sum() * prefactor
        print(f"  {name:4s} {dd:+12.4e} {dh:+12.4e} "
              f"{(dh/dd if dd != 0 else float('nan')):+14.4f}")
    print()

    plt.rcParams['text.usetex'] = False
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)
    injection.data_res_arr.heatmap(fig=fig, ax=ax1, index=0)
    template_fill_wdm.heatmap(fig=fig, ax=ax2, index=0, add_cax=True)
    ax1.set_title("injection")
    ax2.set_title("chunked-het template")
    plt.tight_layout()
    plt.savefig("gb_chunked_test_heatmap.png", dpi=120)
    plt.close()
    print("Saved heatmap to gb_chunked_test_heatmap.png")

    # ---- Eryn MCMC (vectorized, C++ chunked-het likelihood) ----
    # The injection lives on the WDM active band (Nf_active, Nt_active).
    # The chunked-het C++ kernel iterates m in [0, Nf) on the FULL WDM
    # grid, so we zero-fill the active-band injection (+ diag invC) onto
    # (3, Nf, Nt). Then we compute a one-time scale factor so that
    # chunked.get_ll's bare <d,d> sum matches lisatools' inner_product
    # (which multiplies by 4 * differential_component). See
    # gb_chunked_prior_draws.py lines ~400-450 for the same recipe.
    nch = 3
    inj_active_arr = np.asarray(injection.data_res_arr.arr)
    psd_active = np.asarray(sens_mat.sens_mat)
    if psd_active.ndim == 4:
        psd_diag = np.stack([psd_active[c, c] for c in range(nch)], axis=0)
    else:
        psd_diag = psd_active
    with np.errstate(divide="ignore", invalid="ignore"):
        invC_active = 1.0 / np.where(
            np.isfinite(psd_diag) & (psd_diag > 0),
            psd_diag, np.inf,
        )
    invC_active = np.where(np.isfinite(invC_active), invC_active, 0.0)

    data_d_full = np.zeros((nch, Nf, Nt), dtype=float)
    invC_full   = np.zeros_like(data_d_full)
    ilo = wdm_set.ind_min_f
    ihi = wdm_set.ind_max_f + 1
    if wdm_set.Nt_active == wdm_set.Nt:
        data_d_full[:, ilo:ihi, :] = inj_active_arr
        invC_full  [:, ilo:ihi, :] = invC_active
    else:
        tslice = wdm_set.active_slice_t
        data_d_full[:, ilo:ihi, tslice] = inj_active_arr
        invC_full  [:, ilo:ihi, tslice] = invC_active

    d_d_lt = float(np.real(analysis.inner_product()))
    sum_dd_chunked = float(np.sum(data_d_full * data_d_full * invC_full))
    ll_scale = d_d_lt / max(sum_dd_chunked, 1e-300)
    print(f"[mcmc] d_d (lisatools) = {d_d_lt:.6e}", flush=True)
    print(f"[mcmc] ll_scale (chunked->lisatools) = {ll_scale:.6e}", flush=True)

    # Optional layer-grouping speedup (cuts data_d / invC traffic ~Nf /
    # group_band_layers). Driven by env so the slower direct path stays
    # the default safe choice.
    _use_layer_groups = os.environ.get("USE_LAYER_GROUPS", "0") == "1"
    _group_band_layers = int(os.environ.get("GROUP_BAND_LAYERS", 5))
    _margin_layers = int(os.environ.get("MARGIN_LAYERS", 0))
    if _use_layer_groups:
        print(f"[mcmc] use_layer_groups=True  band={_group_band_layers}  "
              f"margin={_margin_layers}", flush=True)

    def chunked_logl_vec(x, transform_fn=None, source_only=True, **_kw):
        """Vectorized chunked-het log-likelihood for Eryn.

        x: ``(N, ndim_sampled)`` or ``(ndim_sampled,)`` array of
        sampled-basis params; transform_fn is the TransformContainer
        that maps sampled -> physical (9-D) basis. Returns shape ``(N,)``
        log-likelihood values.

        Uses ``chunked.get_ll`` (C++ chunked-het kernel) to compute
        ``<d|h>`` and ``<h|h>`` in one batched call across all walkers,
        then rescales by ``ll_scale`` so units match lisatools
        ``inner_product`` (4 * differential_component * sum).
        """
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr[None, :]
        if transform_fn is not None:
            phys = transform_fn.both_transforms(x_arr.copy())  # (N, 9)
        else:
            phys = x_arr
        params_list = [tuple(phys[i].tolist()) for i in range(phys.shape[0])]
        dh, hh = chunked.get_ll(
            data_d_full, invC_full, params_list,
            use_layer_groups=_use_layer_groups,
            margin_layers=_margin_layers,
            group_band_layers=_group_band_layers,
        )
        dh_lt = ll_scale * np.asarray(dh)
        hh_lt = ll_scale * np.asarray(hh)
        if source_only:
            return -0.5 * (hh_lt - 2.0 * dh_lt)
        return -0.5 * (d_d_lt + hh_lt - 2.0 * dh_lt)

    ntemps = int(os.environ.get("NTEMPS", 10))
    nwalkers = int(os.environ.get("NWALKERS", 20))

    # The order here defines full_basis -- must never change
    full_basis = [
        "amp", "f0", "fdot0", "fddot0", "phi0", "inc", "psi", "lam", "beta"
    ]

    # 8 sampled parameters -- order matches priors_in keys
    sampled_basis = [
        "amp", "f0", "fdot0", "phi0", "cosinc", "psi", "lam", "sinbeta"
    ]

    key_map = {
        "cosinc": "inc",
        "sinbeta": "beta"
    }

    parameter_transforms = {
        'cosinc': np.arccos,
        'sinbeta': np.arcsin,
    }

    tc = TransformContainer(
        input_basis=sampled_basis,
        output_basis=full_basis,
        parameter_transforms=parameter_transforms,
        fill_dict={
            'fddot0': 0.0,
        },
        key_map=key_map
    )

    priors = {"gb": ProbDistContainer({
        "amp": uniform_dist(1e-24, 1e-21),
        "f0": uniform_dist(1e-4, 30e-3),
        "fdot": uniform_dist(1e-19, 1e-12),
        "phi0":    uniform_dist(0.0, 2*np.pi),
        "cosinc":  uniform_dist(-1.0, 1.0),
        "psi":     uniform_dist(0.0, np.pi),
        "lam":     uniform_dist(0.0, 2*np.pi),
        "sinbeta": uniform_dist(-1.0, 1.0),
    })}


    factor_gen = 1e-2

    gen_dist = {"gb": ProbDistContainer({
        "amp":     uniform_dist(amp[0] * (1.0 - factor_gen), amp[0] * (1.0 + factor_gen)),
        "f0":      uniform_dist(f0[0] * (1.0 - 1e-8), f0[0] * (1.0 + 1e-8)),
        "fdot0":   uniform_dist(fdot[0] * (1.0 - factor_gen), fdot[0] * (1.0 + factor_gen)),
        "phi0":    uniform_dist(phi0[0] * (1.0 - factor_gen), phi0[0] * (1.0 + factor_gen)),
        "cosinc":  uniform_dist(np.cos(inc[0]) * (1.0 - factor_gen), np.cos(inc[0]) * (1.0 + factor_gen)),
        "psi":     uniform_dist(psi[0] * (1.0 - factor_gen), psi[0] * (1.0 + factor_gen)),
        "lam":     uniform_dist(lam[0] * (1.0 - factor_gen), lam[0] * (1.0 + factor_gen)),
        "sinbeta": uniform_dist(np.sin(beta[0]) * (1.0 - factor_gen), np.sin(beta[0]) * (1.0 + factor_gen)),
    })}

    ndims = {"gb": len(sampled_basis)}

    periodic_container = PeriodicContainer({"gb": {"phi0": 2 * np.pi, "psi": np.pi, "lam": 2 * np.pi}}, key_order={"gb": sampled_basis})
    fp = os.environ.get("MCMC_BACKEND_PATH", "test_gb_chunked_pe.h5")
    if os.path.exists(fp):
        file_backend = HDFBackend(fp)
        start_state = file_backend.get_last_sample()
    else:
        start_state = State({"gb": gen_dist["gb"].rvs(size=(ntemps, nwalkers, 1))})

    sampler = EnsembleSampler(
        nwalkers,
        ndims,
        chunked_logl_vec,                       # vectorized C++ likelihood
        priors,
        tempering_kwargs=dict(ntemps=ntemps),
        kwargs=dict(
            transform_fn=tc,
            source_only=True,
        ),
        moves=StretchMove(live_dangerously=True),
        branch_names=["gb"],
        periodic=periodic_container,
        backend=fp,
        vectorize=True,                          # batch all walkers per call
    )

    if start_state.log_like is None:
        start_state.log_prior = sampler.compute_log_prior(start_state.branches_coords)
        start_state.log_like = sampler.compute_log_like(start_state.branches_coords, logp=start_state.log_prior)[0]

    print("start log_like: ", start_state.log_like)

    inj_params = params[0, tc.test_inds].copy()
    inj_params[sampled_basis.index("cosinc")] = np.cos(inj_params[sampled_basis.index("cosinc")])
    inj_params[sampled_basis.index("sinbeta")] = np.sin(inj_params[sampled_basis.index("sinbeta")])
    tmp_state = State({"gb": np.tile(inj_params, (ntemps, nwalkers, 1, 1))})
    best_like = sampler.compute_log_like(tmp_state.branches_coords)[0]
    print("best_like: ", best_like)

    nsteps = int(os.environ.get("NSTEPS", 2000))
    burn = int(os.environ.get("BURN", 0))
    thin_by = int(os.environ.get("THIN_BY", 5))
    output_state = sampler.run_mcmc(start_state, nsteps=nsteps, burn=burn, thin_by=thin_by, progress=True)

    # ---- Posterior plot ----
    # Cold chain only (temperature index 0). Eryn stores chains as
    # (nsteps_kept, ntemps, nwalkers, nleaves_max, ndim).
    discard_frac = float(os.environ.get("DISCARD_FRAC", 0.25))
    n_stored = sampler.get_chain()["gb"].shape[0]
    discard = int(n_stored * discard_frac)
    chain = sampler.get_chain(discard=discard)["gb"]   # (ns, ntemps, nwalkers, nleaves, ndim)
    cold = chain[:, 0]                                  # (ns, nwalkers, nleaves, ndim)
    samples = cold.reshape(-1, cold.shape[-1])

    try:
        import corner
        fig = corner.corner(
            samples, labels=sampled_basis, truths=inj_params,
            show_titles=True, title_fmt=".4e", quantiles=[0.16, 0.5, 0.84],
        )
        fig.savefig("gb_chunked_posterior.png", dpi=120)
        plt.close(fig)
        print("Saved posterior corner plot to gb_chunked_posterior.png")
    except ImportError:
        nparams = samples.shape[-1]
        fig, axes = plt.subplots(nparams, nparams, figsize=(14, 14))
        for i in range(nparams):
            for j in range(nparams):
                ax = axes[i, j]
                if i == j:
                    ax.hist(samples[:, i], bins=40, color="steelblue", alpha=0.85)
                    ax.axvline(inj_params[i], color="red", lw=1)
                elif i > j:
                    ax.scatter(samples[:, j], samples[:, i], s=1, alpha=0.3, c="steelblue")
                    ax.axvline(inj_params[j], color="red", lw=0.8)
                    ax.axhline(inj_params[i], color="red", lw=0.8)
                else:
                    ax.axis("off")
                if i == nparams - 1:
                    ax.set_xlabel(sampled_basis[j])
                if j == 0 and i > 0:
                    ax.set_ylabel(sampled_basis[i])
        fig.suptitle("Posterior (chunked-het GB MCMC)", fontsize=12)
        fig.tight_layout()
        fig.savefig("gb_chunked_posterior.png", dpi=120)
        plt.close(fig)
        print("Saved posterior (matplotlib fallback) to gb_chunked_posterior.png")

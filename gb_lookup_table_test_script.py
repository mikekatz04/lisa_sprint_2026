#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys
import numpy as np
import matplotlib
# Use Agg by default so the verification run is non-interactive. If the user
# wants the heatmap window, they can override with MPLBACKEND=Qt5Agg etc.
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
from fastlisaresponse.gbcomps import GBWDMComputations

from lisatools.datacontainer import DataResidualArray
from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.domains import TDSettings, TDSignal, FDSettings, FDSignal, WDMSettings, WDMSignal, WDMLookupTable

from eryn.utils import TransformContainer
from eryn.prior import ProbDistContainer, uniform_dist, log_uniform

from eryn.moves import StretchMove
from eryn.ensemble import EnsembleSampler
from eryn.utils import PeriodicContainer

from eryn.state import State
from eryn.backends import HDFBackend
# credit Michael Katz and Alessandro Santini (with internal code contrubtions in docs)


    # HIGHLY RECOMMEND RUNNING THESE THINGS IN A SCRIPT IN THE TERMINAL, OTHERWISE BE CAREFUL TO RUN CELLS IN ORDER AS MUCH AS POSSIBLE

import time
class GBLookupWaveWrap:
    def __init__(self, t_arr, t_tdi_sparse, Tobs, t_ref, dt, num_bin, gb_tdi_kwargs, td_set, output_set, td_window):
        self.t_arr, self.t_tdi_sparse = t_arr, t_tdi_sparse
        self.Tobs, self.t_ref, self.dt, self.num_bin, self.gb_tdi_kwargs = Tobs, t_ref, dt, num_bin, gb_tdi_kwargs
        self.td_set, self.output_set = td_set, output_set
        assert isinstance(output_set, WDMSettings)
        self.td_window = td_window

        # TODO: maybe redo this?
        self.gb_gen = GBTDIonTheFly(
            self.t_tdi_sparse, self.Tobs, self.t_ref, self.dt, 1,
            **self.gb_tdi_kwargs
        )

    def __call__(self, *params):
        
        params = np.asarray([params])
        assert params.shape[-1] == 9
        # st = time.perf_counter()

        # print("check 1", time.perf_counter() - st)
        # st = time.perf_counter()
        wave_tmp = self.gb_gen(*params.T, convert_to_ra_dec=False, return_spline=True)

        t_arr = self.output_set.t_arr + self.t_ref
        f_deriv_tdi = wave_tmp.tdi_phase_spl(np.tile(t_arr, (1, 3, 1)), derivative=1)[0] / (2 * np.pi)
        f_deriv_ref = wave_tmp.phase_ref_spl(t_arr[None, :], derivative=1)[0] / (2 * np.pi)
        f_deriv = f_deriv_ref + f_deriv_tdi

        fdot_deriv_tdi = wave_tmp.tdi_phase_spl(np.tile(t_arr, (1, 3, 1)), derivative=2)[0] / (2 * np.pi)
        fdot_deriv_ref = wave_tmp.phase_ref_spl(t_arr[None, :], derivative=2)[0]  / (2 * np.pi)
        fdot_deriv = fdot_deriv_ref + fdot_deriv_tdi
        tdi_amp = wave_tmp.tdi_amp_spl(np.tile(t_arr, (1, 3, 1)))[0]
        tdi_phase = wave_tmp.tdi_phase_spl(np.tile(t_arr, (1, 3, 1)))[0]
        ref_phase = wave_tmp.phase_ref_spl(t_arr[None, :])[0]
        
        # print("check 2", time.perf_counter() - st)
        # st = time.perf_counter()
        # breakpoint()
        # check_tdi_full = wave_tmp.eval_tdi(t_arr)
        # check_tdi_phase = -np.angle(check_tdi_full * np.exp(1j * ref_phase))

        # ref_phase = wave_tmp.phase_ref  # 2 * np.pi * int(f0[0] / wdm_settings.layer_df) * wdm_settings.layer_df * (output_deriv.t_arr - t_ref) 
        # tdi_phase = np.array([
        #     -np.angle(wave_tmp.X * np.exp(1j * ref_phase)),
        #     -np.angle(wave_tmp.Y * np.exp(1j * ref_phase)),
        #     -np.angle(wave_tmp.Z * np.exp(1j * ref_phase)),
        # ])

        # pi/2 PHASE SHIFT !!!!!!!!!!!!!!!!!!!!!!!!!
        phi_t = ((tdi_phase + ref_phase) + np.pi / 2.).flatten().copy() #  (np.angle(wave_tmp.X).squeeze())# [:-2] # % (2 * np.pi)
        freq_t = f_deriv.flatten().copy()
        fdot_t = np.full_like(freq_t, 0.0)  # fdot_deriv.flatten().copy()
        amp_t = tdi_amp.flatten().copy()

        n_arr = np.tile(xp.arange(self.output_set.Nt)[self.output_set.active_slice_t], (3, 1))
        n_arr_in = n_arr.flatten().copy()
        n_min = self.output_set.ind_min_t
        m_min = self.output_set.ind_min_f

        _wdm_coeffs, _m_layers = wdm_lookup_table.get_wdm_coeffs(
            amp_t, phi_t, freq_t, fdot_t, n_arr_in,
            num_m_layers=int(os.environ.get("NUM_M_LAYERS", 2)),
        )
        wdm_coeffs = _wdm_coeffs.reshape(3, -1, _wdm_coeffs.shape[-1])
        m_layers = _m_layers.reshape(3, -1, _wdm_coeffs.shape[-1])
        n_layers = np.repeat(n_arr[:, :, None], m_layers.shape[-1], axis=-1)
        gb_fill_wave = xp.zeros((3, wdm_set.Nf_active, wdm_set.Nt_active))

        keep_m = (m_layers >= wdm_set.ind_min_f) & (m_layers <= wdm_set.ind_max_f)
        keep_n = (m_layers >= wdm_set.ind_min_f) & (m_layers <= wdm_set.ind_max_f)
        keep = keep_m & keep_n

        # print("check 3", time.perf_counter() - st)
        # st = time.perf_counter()
        channel_ind = np.repeat(np.arange(3)[:, None], m_layers.shape[-1] * m_layers.shape[-2], axis=-1).reshape(m_layers.shape)
        gb_fill_wave[channel_ind[keep], m_layers[keep] - m_min, n_layers[keep] - n_min] = wdm_coeffs[keep]
        # gb_fill_wave[:] = xp.roll(gb_fill_wave, 2, axis=-1)
        # print("check 4", time.perf_counter() - st)
        # st = time.perf_counter()
        gb_fill_wave_wdm = WDMSignal(gb_fill_wave, self.output_set)
        # print("check 5", time.perf_counter() - st)
        return gb_fill_wave_wdm


if __name__ == "__main__":
    backend = "cpu"

    xp = np if backend == "cpu" else cp

    orbits = ESAOrbits(force_backend=backend)
    dt = 10.0  # mojito
    _Tobs = 1. * YRSID_SI
    # between half day and 3/4 day. Will be very close to half day
    # (Nf, Nt, wavelet_duration) = WDMSettings.adjust_to_even_bins(0.5 * 24 * 3600.0, 0.75 * 24 * 3600.0, dt, _Tobs)
    Nt = 256 * 10
    Nf = 1460

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

    num_bin = 1
    amp = np.full(num_bin, 8.0e-22)
    f0 = np.full(num_bin, 18.0e-3)  # (ind + i / num) * wdm_settings.layer_df)
    fdot = np.full(num_bin, 1e-14)
    fddot = np.full(num_bin, 0.0)
    phi0 = np.full(num_bin, 2.09802430298)
    inc = np.full(num_bin, 0.23984234)

    # NEED TO ADD FRAME TRANSFORM FOR PSI IF WORKING IN ECLIPTIC
    psi = np.full(num_bin, 1.234019814)
    lam = np.full(num_bin, 4.09808143)
    beta = np.full(num_bin, 0.04)
    params = np.array([amp, f0, fdot, fddot, phi0, inc, psi, lam, beta]).T

    N = data_inj.shape[-1]
    td_set = TDSettings(N, dt, force_backend=backend)
    freqs = np.fft.rfftfreq(N, dt)
    df = freqs[1] - freqs[0]
    N_fd = len(freqs)
    window = xp.asarray(signal.windows.tukey(N, alpha=0.05))

    min_freq = 0.0029493407356002777  # 255 * wdm_set.layer_df
    max_freq = 0.00306500115660421  # 265 * wdm_set.layer_df
    fd_set = FDSettings(N_fd, df, min_freq=min_freq, max_freq=max_freq, force_backend=backend)

    # shave edges?
    min_time = 20 * wavelet_duration
    max_time = (Nt - 20) * wavelet_duration

    wdm_set = WDMSettings(Nf, Nt, dt, min_freq=min_freq, max_freq=max_freq, min_time=min_time, max_time=max_time)
    inj_tmp = gb_gen_inj(amp, f0, fdot, fddot, phi0, inc, psi, lam, beta, convert_to_ra_dec=False, return_spline=True)
    data_inj[:] = inj_tmp.eval_tdi(t_arr)

    output_set = wdm_set

    if output_set != wdm_set:
        raise ValueError("This script requires WDM for the output_set.")
    
    # plot
    # plt.rcParams['text.usetex'] = False
    # template_sparse.heatmap()
    # plt.show()
    # plt.close()

    data_inj_all = TDSignal(data_inj, settings=td_set).transform(output_set, window=window)
    injection = DataResidualArray(data_inj_all)
    sens_mat = XYZ2SensitivityMatrix(injection.data_res_arr.settings, model="scirdv1")

    
    ## mcmc functions

    store_path = "wdm_lookup_new_all_time_layers_1.h5"
        
    ## lookup table setup
    if os.path.exists(store_path):
        wdm_lookup_table = WDMLookupTable.from_file(store_path, force_backend=backend)
        _wdm_settings = WDMSettings(*wdm_lookup_table.args, **wdm_lookup_table.kwargs)
        if not _wdm_settings.eq_without_inds(wdm_set):
            raise ValueError("WDM Settings are not equivalent to lookup table. Either adjust to lookup table settings or regenerate the table.")

        # Nt = wdm_settings.Nt
        # Nf = wdm_settings.Nf
        # N = wdm_settings.N
        # Tobs = wdm_settings.Tobs

    else:
        time_layers = wdm_set.Nt
        td_window = xp.asarray(signal.windows.tukey(wdm_set.Nf * time_layers, alpha=0.05))
        m_ref = int(3e-3 / wdm_set.layer_df)
        EPS_FREQ        = float(os.environ.get("EPS_FREQ", 0.001))
        NUM_LAYERS_DIFF = int(os.environ.get("NUM_LAYERS_DIFF", 5))
        print(f"[step] table build with EPS_FREQ={EPS_FREQ}, NUM_LAYERS_DIFF={NUM_LAYERS_DIFF}", flush=True)
        norm_freq_single_layer, m_diffs, _ = WDMLookupTable.apply_eps_frequency(
            EPS_FREQ, wdm_set, m_ref=m_ref, num_layers_diff=NUM_LAYERS_DIFF
        )

        fdot_vals = np.array([0.0])
        # fdot_vals = WDMLookupTable.apply_eps_fdot(0.2, wdm_set, fdot_max_factor=1.0)

        nchannel = 3
        wdm_lookup_table = WDMLookupTable(wdm_set, nchannel, norm_freq_single_layer=norm_freq_single_layer, m_diffs=m_diffs, fdot_vals=fdot_vals, m_ref=m_ref, batch_size_gen=5, td_window=td_window, store_path=store_path)

    # this tests cubic spline accuracy for python setup
    # C setup currently does central differencing at the wdm grid
    N_sparse = int(os.environ.get("N_SPARSE", 4096))
    print(f"[step] N_sparse={N_sparse}", flush=True)
    t_tdi_sparse = xp.linspace(t_arr[0], t_arr[-1], N_sparse)

    gb_comps = GBWDMComputations(wdm_lookup_table, Tobs, t_ref, orbits=orbits, tdi_config=tdi_config, force_backend=backend)
    gb_gen_wrap = GBLookupWaveWrap(
        t_arr, 
        t_tdi_sparse, 
        Tobs,
        t_ref,
        dt,
        params.shape[0],
        gb_tdi_kwargs,
        td_set, 
        output_set,
        window
    )

    analysis = AnalysisContainer(injection, sens_mat, signal_gen=gb_gen_wrap)

    wdm_holder = AnalysisContainerArray([analysis])

    template_fill = xp.zeros(3 * np.prod(wdm_set.basis_shape_active), dtype=float)
    # Pass convert_to_ra_dec=False so the C kernel uses the same ecliptic frame
    # as the injection and Python wrap (default would apply ecliptic→ICRS, shifting
    # the source location and making C and Python results diverge).
    gb_comps.fill_global_wdm(template_fill, params, wdm_holder, data_index=None, convert_to_ra_dec=False)
    template_fill_wdm = WDMSignal(template_fill.reshape((3,) + wdm_set.basis_shape_active), wdm_set)
    check_ll_2 = analysis.template_likelihood(template_fill_wdm)  # template_likelihood ignores psd likelihood by default
    check_ip_2 = analysis.template_inner_product(template_fill_wdm)
    check_ip_d_d = analysis.inner_product()
    overlap = analysis.template_inner_product(template_fill_wdm, normalize=True)

    # ---- swap-likelihood vs get-likelihood cross-check -------------------------
    # Placed before the Python-side gb_gen_wrap() call so it runs regardless of
    # whether the Python lookup path is healthy. Detailed description below in
    # the comment block before the configuration setup.
    _run_swap_ll_check = True
    if _run_swap_ll_check:
        # The injection f0 in `params` (18 mHz) sits far outside the WDM active
        # band [ind_min_f, ind_max_f]*layer_df, so both get_ll and swap_ll would
        # trivially return zero on it. Pick a test frequency in the middle of
        # the active band so the kernel actually accumulates contributions.
        layer_df = wdm_set.layer_df
        m_lo, m_hi = wdm_set.ind_min_f, wdm_set.ind_max_f
        m_mid_A = (m_lo + m_hi) // 2
        m_mid_B = m_mid_A + 3  # 3-layer offset; still inside the band if it fits
        if m_mid_B > m_hi:
            m_mid_B = m_hi
        f0_A = m_mid_A * layer_df
        f0_B = m_mid_B * layer_df

        params_A = params.copy()
        params_B = params.copy()
        params_A[:, 1] = f0_A
        params_B[:, 1] = f0_B
        print(f"[swap_ll test] active band layers=[{m_lo},{m_hi}], layer_df={layer_df:.6e}")
        print(f"[swap_ll test] sourceA f0={f0_A:.6e} (layer {m_mid_A}), sourceB f0={f0_B:.6e} (layer {m_mid_B})")

        _ = gb_comps.get_ll_wdm(params_A, wdm_holder, data_index=None, noise_index=None, convert_to_ra_dec=False)
        d_h_A_ref = float(gb_comps.d_h_out[0])
        h_h_A_ref = float(gb_comps.h_h_out[0])

        _ = gb_comps.get_ll_wdm(params_B, wdm_holder, data_index=None, noise_index=None, convert_to_ra_dec=False)
        d_h_B_ref = float(gb_comps.d_h_out[0])
        h_h_B_ref = float(gb_comps.h_h_out[0])

        like_A_ref = -0.5 * (gb_comps.d_d + h_h_A_ref - 2 * d_h_A_ref)
        like_B_ref = -0.5 * (gb_comps.d_d + h_h_B_ref - 2 * d_h_B_ref)

        params_add_in    = np.stack([params_A[0], params_A[0], params_B[0]], axis=0)
        params_remove_in = np.stack([params_A[0], params_B[0], params_A[0]], axis=0)

        (
            like_add,
            like_remove,
            d_h_add,
            d_h_remove,
            add_add,
            remove_remove,
            add_remove,
        ) = gb_comps.get_swap_ll_wdm(
            params_add_in, params_remove_in, wdm_holder,
            data_index=None, noise_index=None, convert_to_ra_dec=False,
        )

        def _xp_to_np(a):
            return a.get() if hasattr(a, "get") else np.asarray(a)

        d_h_add        = _xp_to_np(d_h_add)
        d_h_remove     = _xp_to_np(d_h_remove)
        add_add        = _xp_to_np(add_add)
        remove_remove  = _xp_to_np(remove_remove)
        add_remove     = _xp_to_np(add_remove)
        like_add       = _xp_to_np(like_add)
        like_remove    = _xp_to_np(like_remove)

        def _rel(a, b):
            denom = max(abs(b), 1e-300)
            return abs(a - b) / denom

        # bin 0: same source, all outputs == get_ll(sourceA)
        bin0_rel = max(
            _rel(d_h_add[0],       d_h_A_ref),
            _rel(d_h_remove[0],    d_h_A_ref),
            _rel(add_add[0],       h_h_A_ref),
            _rel(remove_remove[0], h_h_A_ref),
            _rel(add_remove[0],    h_h_A_ref),
            _rel(like_add[0],      like_A_ref),
            _rel(like_remove[0],   like_A_ref),
        )

        # bin 1: add=A, remove=B
        bin1_rel = max(
            _rel(d_h_add[1],       d_h_A_ref),
            _rel(d_h_remove[1],    d_h_B_ref),
            _rel(add_add[1],       h_h_A_ref),
            _rel(remove_remove[1], h_h_B_ref),
            _rel(like_add[1],      like_A_ref),
            _rel(like_remove[1],   like_B_ref),
        )

        # bin 2: add=B, remove=A. Must mirror bin 1 after swap.
        bin2_rel = max(
            _rel(d_h_add[2],       d_h_B_ref),
            _rel(d_h_remove[2],    d_h_A_ref),
            _rel(add_add[2],       h_h_B_ref),
            _rel(remove_remove[2], h_h_A_ref),
            _rel(like_add[2],      like_B_ref),
            _rel(like_remove[2],   like_A_ref),
        )

        # swap symmetry: cross term must be the same whichever side is "add"
        sym_rel = _rel(add_remove[1], add_remove[2])

        # Cauchy-Schwarz on each bin: |<h_a|h_r>|^2 <= <h_a|h_a> * <h_r|h_r>
        cs_eps = 1e-9
        cs_bin = []
        for k in range(3):
            lhs = add_remove[k] ** 2
            rhs = (1.0 + cs_eps) * add_add[k] * remove_remove[k]
            cs_bin.append((lhs, rhs, lhs <= rhs))

        print(f"==== swap_ll vs get_ll cross-check ====")
        print(f"  sourceA: d_h = {d_h_A_ref:+.8e}   h_h = {h_h_A_ref:+.8e}   like = {like_A_ref:+.8e}")
        print(f"  sourceB: d_h = {d_h_B_ref:+.8e}   h_h = {h_h_B_ref:+.8e}   like = {like_B_ref:+.8e}")
        print(f"  bin0 (A,A)  rel_max = {bin0_rel:.3e}")
        print(f"             d_h_add={d_h_add[0]:+.6e}  d_h_remove={d_h_remove[0]:+.6e}")
        print(f"             add_add={add_add[0]:+.6e}  remove_remove={remove_remove[0]:+.6e}  add_remove={add_remove[0]:+.6e}")
        print(f"  bin1 (A,B)  rel_max = {bin1_rel:.3e}")
        print(f"             d_h_add={d_h_add[1]:+.6e}  d_h_remove={d_h_remove[1]:+.6e}")
        print(f"             add_add={add_add[1]:+.6e}  remove_remove={remove_remove[1]:+.6e}  add_remove={add_remove[1]:+.6e}")
        print(f"  bin2 (B,A)  rel_max = {bin2_rel:.3e}")
        print(f"             d_h_add={d_h_add[2]:+.6e}  d_h_remove={d_h_remove[2]:+.6e}")
        print(f"             add_add={add_add[2]:+.6e}  remove_remove={remove_remove[2]:+.6e}  add_remove={add_remove[2]:+.6e}")
        print(f"  swap symmetry add_remove(A,B) vs add_remove(B,A)  rel = {sym_rel:.3e}")
        for k, (lhs, rhs, ok) in enumerate(cs_bin):
            status = "ok" if ok else "FAIL"
            print(f"  Cauchy-Schwarz bin{k}: |add_remove|^2={lhs:+.6e}  <h_a|h_a><h_r|h_r>={rhs:+.6e}  [{status}]")

        swap_tol = float(os.environ.get("SWAP_LL_TOL", 1e-10))
        sym_tol  = float(os.environ.get("SWAP_LL_SYM_TOL", 1e-9))
        worst = max(bin0_rel, bin1_rel, bin2_rel)
        cs_ok = all(c[2] for c in cs_bin)
        if worst > swap_tol:
            print(f"[FAIL] swap_ll <-> get_ll disagree above tol={swap_tol}; worst rel = {worst:.3e}")
        elif sym_rel > sym_tol:
            print(f"[FAIL] swap-symmetry violated: add_remove(A,B) != add_remove(B,A) (rel = {sym_rel:.3e}, tol = {sym_tol})")
        elif not cs_ok:
            print(f"[FAIL] Cauchy-Schwarz violated on at least one bin")
        else:
            print(f"[ok] swap_ll matches get_ll on each per-template piece (worst rel = {worst:.3e}, tol = {swap_tol}),")
            print(f"     cross term is symmetric under add↔remove (rel = {sym_rel:.3e}, tol = {sym_tol}),")
            print(f"     and obeys Cauchy-Schwarz on every bin.\n")


    # ---- chain-rule gradient cross-check --------------------------------------
    # Same setup as the swap_ll block above: source A and source B inside the
    # active WDM band.  We call the new C/CUDA gradient kernels
    #
    #     gb_comps.get_ll_grad_wdm(...)         (shape (num_bin, 9))
    #     gb_comps.get_swap_ll_grad_wdm(...)    (grad_add, grad_remove)
    #
    # which compute the per-binary parameter gradient of L = -1/2 (d_d + h_h - 2 d_h)
    # (and of the swap log-likelihood ratio) via per-pixel central differences on
    # top of the same fast_wdm_inner / wdm_lookup pipeline used by get_ll / swap_ll.
    #
    # As a reference we run a Python-side finite difference of the *same*
    # likelihood values (get_ll_wdm / get_swap_ll_wdm) with the *same* per-param
    # step sizes ``param_eps`` -- so the two methods are computing the same
    # mathematical object, just at different granularity (per-pixel vs per-run).
    # They must agree to round-off.
    _run_grad_check = bool(int(os.environ.get("RUN_GRAD_CHECK", "1")))
    if _run_grad_check:
        # Same per-parameter FD step that the C kernel will use by default
        # (see GBWDMComputations._DEFAULT_PARAM_EPS).  Keep them consistent so
        # the two FDs cancel exactly.
        param_eps_default = np.array(GBWDMComputations._DEFAULT_PARAM_EPS, dtype=np.float64)
        nparams = params_A.shape[1]
        assert param_eps_default.shape[0] == nparams

        def _xp_to_np_local(a):
            return a.get() if hasattr(a, "get") else np.asarray(a)

        # --- get_ll gradient: source A
        grad_C = gb_comps.get_ll_grad_wdm(
            params_A, wdm_holder,
            param_eps=param_eps_default,
            data_index=None, noise_index=None, convert_to_ra_dec=False,
        )
        grad_C = _xp_to_np_local(grad_C)[0]   # (nparams,)

        def _ll_at(p_arr):
            """Recompute L = -0.5 (d_d + h_h - 2 d_h) at the supplied params."""
            _ = gb_comps.get_ll_wdm(
                p_arr, wdm_holder,
                data_index=None, noise_index=None, convert_to_ra_dec=False,
            )
            d_h = float(_xp_to_np_local(gb_comps.d_h_out)[0])
            h_h = float(_xp_to_np_local(gb_comps.h_h_out)[0])
            return -0.5 * (gb_comps.d_d + h_h - 2 * d_h)

        grad_fd = np.zeros(nparams, dtype=np.float64)
        for k in range(nparams):
            eps_k = param_eps_default[k]
            if eps_k <= 0.0:
                continue
            pp = params_A.copy(); pp[:, k] = pp[:, k] + eps_k
            pm = params_A.copy(); pm[:, k] = pm[:, k] - eps_k
            grad_fd[k] = (_ll_at(pp) - _ll_at(pm)) / (2.0 * eps_k)

        param_names = ("amp", "f0", "fdot", "fddot", "phi0", "iota", "psi", "lam", "beta")
        print(f"==== get_ll gradient (C/CUDA chain rule) vs Python FD of get_ll ====")
        worst_grad = 0.0
        for k in range(nparams):
            denom = max(abs(grad_fd[k]), 1e-300)
            rel = abs(grad_C[k] - grad_fd[k]) / denom
            worst_grad = max(worst_grad, rel)
            print(f"  {param_names[k]:>5s}  C={grad_C[k]:+.8e}  FD={grad_fd[k]:+.8e}  rel={rel:.2e}")
        grad_tol = float(os.environ.get("GRAD_LL_TOL", 1e-8))
        status = "ok" if worst_grad < grad_tol else "FAIL"
        print(f"  worst rel = {worst_grad:.3e}   tol = {grad_tol}   [{status}]\n")

        # --- swap_ll gradient: (add, remove) on the three bins from above
        grad_add_C, grad_remove_C = gb_comps.get_swap_ll_grad_wdm(
            params_add_in, params_remove_in, wdm_holder,
            param_eps_add=param_eps_default,
            param_eps_remove=param_eps_default,
            data_index=None, noise_index=None, convert_to_ra_dec=False,
        )
        grad_add_C = _xp_to_np_local(grad_add_C)
        grad_remove_C = _xp_to_np_local(grad_remove_C)

        def _ll_diff_at(p_add, p_remove):
            """ll_diff = L(after swap) - L(before swap) at supplied params.

            Mirrors gb_wdm_swap_ll_kernel: returns -1/2 (-2 d_h_add + 2 d_h_remove
            - 2 add_remove + add_add + remove_remove). Same convention used by
            the C swap_ll_grad kernel."""
            (
                _like_add, _like_remove,
                d_h_add_v, d_h_remove_v,
                add_add_v, remove_remove_v, add_remove_v,
            ) = gb_comps.get_swap_ll_wdm(
                p_add, p_remove, wdm_holder,
                data_index=None, noise_index=None, convert_to_ra_dec=False,
            )
            d_h_a = _xp_to_np_local(d_h_add_v)
            d_h_r = _xp_to_np_local(d_h_remove_v)
            aa    = _xp_to_np_local(add_add_v)
            rr    = _xp_to_np_local(remove_remove_v)
            ar    = _xp_to_np_local(add_remove_v)
            return -0.5 * (-2.0 * d_h_a + 2.0 * d_h_r - 2.0 * ar + aa + rr)

        # FD w.r.t. add params
        grad_add_fd = np.zeros_like(grad_add_C)
        for k in range(nparams):
            eps_k = param_eps_default[k]
            if eps_k <= 0.0:
                continue
            pa_p = params_add_in.copy(); pa_p[:, k] = pa_p[:, k] + eps_k
            pa_m = params_add_in.copy(); pa_m[:, k] = pa_m[:, k] - eps_k
            ll_p = _ll_diff_at(pa_p, params_remove_in)
            ll_m = _ll_diff_at(pa_m, params_remove_in)
            grad_add_fd[:, k] = (ll_p - ll_m) / (2.0 * eps_k)

        # FD w.r.t. remove params
        grad_remove_fd = np.zeros_like(grad_remove_C)
        for k in range(nparams):
            eps_k = param_eps_default[k]
            if eps_k <= 0.0:
                continue
            pr_p = params_remove_in.copy(); pr_p[:, k] = pr_p[:, k] + eps_k
            pr_m = params_remove_in.copy(); pr_m[:, k] = pr_m[:, k] - eps_k
            ll_p = _ll_diff_at(params_add_in, pr_p)
            ll_m = _ll_diff_at(params_add_in, pr_m)
            grad_remove_fd[:, k] = (ll_p - ll_m) / (2.0 * eps_k)

        print(f"==== swap_ll gradient (C/CUDA chain rule) vs Python FD of swap_ll ====")
        worst_swap_grad = 0.0
        for which, gC, gFD in (("add", grad_add_C, grad_add_fd),
                                ("remove", grad_remove_C, grad_remove_fd)):
            print(f"  >>> theta_{which}")
            for bin_idx in range(gC.shape[0]):
                for k in range(nparams):
                    denom = max(abs(gFD[bin_idx, k]), 1e-300)
                    rel = abs(gC[bin_idx, k] - gFD[bin_idx, k]) / denom
                    worst_swap_grad = max(worst_swap_grad, rel)
                # only print the bin's worst-row to keep the log readable;
                # uncomment the per-k print above if you need full detail.
                row_rel = np.max(np.abs(gC[bin_idx] - gFD[bin_idx])
                                 / np.maximum(np.abs(gFD[bin_idx]), 1e-300))
                print(f"    bin{bin_idx}  worst rel = {row_rel:.3e}")
                # full per-parameter dump for inspection (concise)
                for k in range(nparams):
                    print(f"      {param_names[k]:>5s}  C={gC[bin_idx,k]:+.6e}  FD={gFD[bin_idx,k]:+.6e}")
        status = "ok" if worst_swap_grad < grad_tol else "FAIL"
        print(f"  swap-gradient worst rel = {worst_swap_grad:.3e}   tol = {grad_tol}   [{status}]\n")

    # The wrap's __call__ takes the 9 scalar params as *args (see GBLookupWaveWrap above).
    py_wdm_lookup = gb_gen_wrap(*params[0])
    tmp_val1 = analysis.template_inner_product(py_wdm_lookup)
    tmp_val2 = analysis.calculate_signal_inner_product(*params[0])
    tmp_val3 = analysis.calculate_signal_likelihood(*params[0], source_only=True)
    overlap_py = analysis.template_inner_product(py_wdm_lookup, normalize=True)

    print(f"\n[result] base inner_product <d|d>            = {check_ip_d_d}")
    print(f"[result] template_inner_product (C lookup)     = {check_ip_2}")
    print(f"[result] template_inner_product (py lookup)    = {tmp_val1}")
    print(f"[result] calculate_signal_inner_product        = {tmp_val2}")
    print(f"[result] template_likelihood (C lookup)        = {check_ll_2}\n")
    print(f"[result] template_likelihood (py lookup)        = {tmp_val3}\n")
    print(f"[result] Noise-weighted mismatch (C lookup)        = {1.0 - overlap}\n")
    print(f"[result] Noise-weighted mismatch (py lookup)        = {1.0 - overlap_py}\n")

    # Per-channel AET cross-check: convert XYZ → AET in WDM space and compare
    # each orthogonal AET channel separately (they decouple under AET1Sens).
    from lisatools.sensitivity import AET1SensitivityMatrix
    def xyz_to_aet(arr):
        X, Y, Z = arr[0], arr[1], arr[2]
        A = (Z - X) / np.sqrt(2.0)
        E = (X - 2.0 * Y + Z) / np.sqrt(6.0)
        T = (X + Y + Z) / np.sqrt(3.0)
        return np.stack([A, E, T], axis=0)
    inj_aet = xyz_to_aet(injection.data_res_arr.arr)
    c_tpl_aet = xyz_to_aet(template_fill_wdm.arr)
    py_tpl_aet = xyz_to_aet(py_wdm_lookup.arr)
    sens_aet = AET1SensitivityMatrix(WDMSignal(inj_aet, wdm_set).settings)
    invC = sens_aet.invC
    prefactor = 4.0 * sens_aet.differential_component
    print(f"==== per-channel AET inner products ====")
    print(f"{'chan':6s} {'<d|d>':>12s} {'<d|h_C>':>12s} {'<d|h_py>':>12s} {'r_C':>9s} {'r_py':>9s}")
    for chan, name in enumerate("AET"):
        d_c = inj_aet[chan]; hC = c_tpl_aet[chan]; hPy = py_tpl_aet[chan]; ic = invC[chan]
        dd = (d_c*d_c*ic).sum()*prefactor
        dhC = (d_c*hC*ic).sum()*prefactor
        dhPy = (d_c*hPy*ic).sum()*prefactor
        print(f"  {name:4s} {dd:+12.4e} {dhC:+12.4e} {dhPy:+12.4e} "
              f"{(dhC/dd if dd!=0 else float('nan')):+9.4f} "
              f"{(dhPy/dd if dd!=0 else float('nan')):+9.4f}")
    print()

    plt.rcParams['text.usetex'] = False
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True)
    injection.data_res_arr.heatmap(fig=fig, ax=ax1, index=0)
    template_fill_wdm.heatmap(fig=fig, ax=ax2, index=0, add_cax=True)
    py_wdm_lookup.heatmap(fig=fig, ax=ax3, index=0)
    ax1.set_title("injection")
    ax2.set_title("C lookup template")
    ax3.set_title("Python lookup template")
    plt.tight_layout()
    plt.savefig("gb_lookup_test_heatmap.png", dpi=120)
    plt.close()
    print("Saved heatmap to gb_lookup_test_heatmap.png")

    # Stop here unless the user opts into the MCMC stage. The MCMC below takes a
    # long time; for plain inner-product verification we don't need it.
    # if os.environ.get("RUN_MCMC", "0") != "1":
    #     print("Verification complete. Set RUN_MCMC=1 to also run the MCMC stage.")
    #     sys.exit(0)
    exit()
    analysis_mcmc = AnalysisContainer(injection, sens_mat, signal_gen=gb_gen_wrap)
    
    ntemps = 10
    nwalkers = 20


    # The order here defines full_basis — must never change
    full_basis = [
        "amp", "f0", "fdot0", "fddot0", "phi0", "inc", "psi", "lam", "beta"
    ]

    # 12 sampled parameters — order matches priors_in keys
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
        parameter_transforms=parameter_transforms,   # no transforms needed: M is linear, qS/qK are cosines
        fill_dict={
            'fddot0': 0.0,
        },
        key_map=key_map
    )

    priors = {"gb": ProbDistContainer({
        "amp": uniform_dist(1e-24, 1e-21),
        "f0": uniform_dist(1e-4, 30e-3),
        "fdot": uniform_dist(1e-19, 1e-12),      
        "phi0":     uniform_dist(0.0, 2*np.pi),    
        "cosinc":    uniform_dist(-1.0, 1.0),        
        "psi":     uniform_dist(0.0, np.pi),     
        "lam": uniform_dist(0.0, 2*np.pi),    
        "sinbeta":  uniform_dist(-1.0, 1.0),       
    })}


    factor_gen = 1e-2

    gen_dist = {"gb": ProbDistContainer({
        "amp": uniform_dist(amp[0] * (1.0 - factor_gen), amp[0] * (1.0 + factor_gen)),
        "f0": uniform_dist(f0[0] * (1.0 - 1e-8), f0[0] * (1.0 + 1e-8)),  # different value for frequency
        "fdot0":  uniform_dist(fdot[0] *  (1.0 - factor_gen), fdot[0] * (1.0 + factor_gen)),       # dimensionless spin
        "phi0":    uniform_dist(phi0[0] *  (1.0 - factor_gen), phi0[0] * (1.0 + factor_gen)),       # semi-latus rectum
        "cosinc":       uniform_dist(np.cos(inc[0]) *  (1.0 - factor_gen), np.cos(inc[0]) * (1.0 + factor_gen)),         # eccentricity
        "psi":     uniform_dist(psi[0] *  (1.0 - factor_gen), psi[0] * (1.0 + factor_gen)),        # luminosity distance [Gpc]
        "lam":    uniform_dist(lam[0] *  (1.0 - factor_gen), lam[0] * (1.0 + factor_gen)),        # cos(polar spin angle)
        "sinbeta":     uniform_dist(np.sin(beta[0]) *  (1.0 - factor_gen), np.sin(beta[0]) * (1.0 + factor_gen)),     # azimuthal spin angle [rad]
    })}

    ndims = {"gb": len(sampled_basis)}

    periodic_container = PeriodicContainer({"gb": {"phi0": 2 * np.pi, "psi": np.pi, "lam": 2 * np.pi}}, key_order={"gb": sampled_basis})
    fp = f"test_gb_lookup_pe_new_new_1.h5"
    if os.path.exists(fp):
        file_backend = HDFBackend(fp)
        start_state = file_backend.get_last_sample()
    else:
        start_state = State({"gb": gen_dist["gb"].rvs(size=(ntemps, nwalkers, 1))})
        
    sampler = EnsembleSampler(
        nwalkers,
        ndims,                              # list of ndims, one per branch
        analysis_mcmc.eryn_likelihood_function,
        priors,
        tempering_kwargs=dict(ntemps=ntemps),
        kwargs=dict(
            transform_fn=tc,
            source_only=True,
        ),
        moves=StretchMove(live_dangerously=True),  # allows for testing
        branch_names=["gb"],
        periodic=periodic_container,
        backend=fp
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
    
    nsteps = 2000
    burn = 0
    output_state = sampler.run_mcmc(start_state, nsteps=nsteps, burn=burn, thin_by=5, progress=True)



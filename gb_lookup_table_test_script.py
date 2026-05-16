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

        _wdm_coeffs, _m_layers = wdm_lookup_table.get_wdm_coeffs(amp_t, phi_t, freq_t, fdot_t, n_arr_in, num_m_layers=2)
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
    Nt = 256
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
    f0 = np.full(num_bin, 3.0e-3)  # (ind + i / num) * wdm_settings.layer_df)
    fdot = np.full(num_bin, 1e-16)
    fddot = np.full(num_bin, 0.0)
    phi0 = np.full(num_bin, 2.09802430298)
    inc = np.full(num_bin, 0.23984234)

    # NEED TO ADD FRAME TRANSFORM FOR PSI IF WORKING IN ECLIPTIC
    psi = np.full(num_bin, 1.234019814)
    lam = np.full(num_bin, 4.09808143)
    beta = np.full(num_bin, 1.1)
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
    min_time = 0 * wavelet_duration
    max_time = (Nt - 0) * wavelet_duration

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
        norm_freq_single_layer, m_diffs, _ = WDMLookupTable.apply_eps_frequency(0.0005, wdm_set, m_ref=m_ref, num_layers_diff=5)
            
        fdot_vals = np.array([0.0])
        # fdot_vals = WDMLookupTable.apply_eps_fdot(0.2, wdm_set, fdot_max_factor=1.0) 

        nchannel = 3
        wdm_lookup_table = WDMLookupTable(wdm_set, nchannel, norm_freq_single_layer=norm_freq_single_layer, m_diffs=m_diffs, fdot_vals=fdot_vals, m_ref=m_ref, batch_size_gen=5, td_window=td_window, store_path=store_path)

    # this tests cubic spline accuracy for python setup
    # C setup currently does central differencing at the wdm grid
    N_sparse = 2048
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

    # The wrap's __call__ takes the 9 scalar params as *args (see GBLookupWaveWrap above).
    py_wdm_lookup = gb_gen_wrap(*params[0])
    tmp_val1 = analysis.template_inner_product(py_wdm_lookup)
    tmp_val2 = analysis.calculate_signal_inner_product(*params[0])

    print(f"\n[result] base inner_product <d|d>            = {check_ip_d_d}")
    print(f"[result] template_inner_product (C lookup)     = {check_ip_2}")
    print(f"[result] template_inner_product (py lookup)    = {tmp_val1}")
    print(f"[result] calculate_signal_inner_product        = {tmp_val2}")
    print(f"[result] template_likelihood (C lookup)        = {check_ll_2}\n")

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



"""
Cross-check the lisatools optimal SNR against the FFT/IFFT time-slide
matched-filter SNR for an EMRI signal projected through the LISA response.

Setup:
  * Single-mode EMRI: only (l, m, k, n) = (2, 2, 0, 0) (with the m -> -m
    partner so the time-domain waveform is real).
  * Tobs ~ 0.25 yr, dt = 10 s, 2nd-generation TDI via fastlisaresponse.
  * Two TDI channel choices: AET (diagonal noise) and XYZ (3x3 CSD).
  * Same template = data, so the matched filter SNR(tau) peaks at tau = 0
    and equals the lisatools optimal SNR there.

For each TDI choice we:
  (1) compute SNR via lisatools.AnalysisContainer.snr(),
  (2) compute SNR(tau) via 4 * df * N * irfft( H^* invC H ) / sqrt<h|h>,
      and report the value at tau = 0 plus the peak.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from lisatools.detector import EqualArmlengthOrbits
from lisatools.utils.constants import YRSID_SI
from lisatools.datacontainer import DataResidualArray
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.sensitivity import AET1SensitivityMatrix, XYZ1SensitivityMatrix
from lisatools.domains import TDSettings

from fastlisaresponse import ResponseWrapper
from fastlisaresponse.tdiconfig import TDIConfig

from few.waveform import GenerateEMRIWaveform


# -----------------------------------------------------------------------------
# 1. Time grid + orbits + TDI generation config
# -----------------------------------------------------------------------------
backend = "cpu"
dt      = 10.0
Tobs    = 2.0 / 12.0      # ~ 2 months in years -- short for speed
t_start = 0.5 * YRSID_SI  # arbitrary epoch into the orbits

orbits     = EqualArmlengthOrbits(force_backend=backend)
tdi_config = TDIConfig("1st generation")

# -----------------------------------------------------------------------------
# 2. EMRI generator (FastKerrEccentricEquatorialFlux), single-mode at call time
# -----------------------------------------------------------------------------
emri_gen = GenerateEMRIWaveform(
    "FastKerrEccentricEquatorialFlux",
    return_list=False,
    inspiral_kwargs={"DENSE_STEPPING": 0, "max_init_len": int(1e4),
                     "force_backend": backend},
    sum_kwargs={"pad_output": True},
    frame="detector",
    mode_selector_kwargs={"mode_selection_threshold": 1e-5},
    force_backend=backend,
)

# Common ResponseWrapper kwargs
response_kwargs = dict(
    Tobs=Tobs,
    dt=dt,
    index_lambda=8,
    index_beta=7,
    flip_hx=True,
    force_backend=backend,
    tdi=tdi_config,
    order=20,
    remove_garbage="zero",
    is_ecliptic_latitude=False,
    t_buffer=3e4,
    t0=t_start,
)

response_AET = ResponseWrapper(emri_gen, orbits=orbits, tdi_chan="AET",
                               **response_kwargs)
response_XYZ = ResponseWrapper(emri_gen, orbits=orbits, tdi_chan="XYZ",
                               **response_kwargs)

# -----------------------------------------------------------------------------
# 3. Source parameters (matching the existing test script in this workspace)
# -----------------------------------------------------------------------------
m1, m2, a    = 1e6, 1e1, 0.99
p0, e0, xI0  = 6.1, 0.3, +1.0
dist         = 2.0                                    # Gpc
beta, lam    = -0.418762312, 0.2538922432234         # ecliptic
qS, phiS     = np.pi / 2 - beta, lam                 # polar / azimuth
qK, phiK     = 0.2340980298542, 4.098234232
Phi_phi0     = 1.0803123123
Phi_theta0   = 1.9823423423
Phi_r0       = 4.32094823423

inj_params = np.array([
    m1, m2, a, p0, e0, xI0,
    dist, qS, phiS, qK, phiK,
    Phi_phi0, Phi_theta0, Phi_r0,
])

# Restrict to a SINGLE harmonic mode
runtime_kwargs = dict(
    mode_selection=[(2, 2, 0, 0)],
    include_minus_mkn=True,
)

# -----------------------------------------------------------------------------
# 4. Generate the TDI signals
# -----------------------------------------------------------------------------
print("Generating EMRI + LISA response (AET, single mode)...")
hAET = np.asarray(response_AET(*inj_params, convert_to_ra_dec=False,
                               **runtime_kwargs))
print(f"  AET shape: {hAET.shape}")

print("Generating EMRI + LISA response (XYZ, single mode)...")
hXYZ = np.asarray(response_XYZ(*inj_params, convert_to_ra_dec=False,
                               **runtime_kwargs))
print(f"  XYZ shape: {hXYZ.shape}")

assert hAET.shape == hXYZ.shape, "AET and XYZ outputs must have the same shape"
N = hAET.shape[-1]

# -----------------------------------------------------------------------------
# 5. lisatools SNR via AnalysisContainer (auto TD -> FD)
# -----------------------------------------------------------------------------
td_set = TDSettings(N, dt, force_backend=backend)

dra_AET   = DataResidualArray(hAET, input_signal_domain=td_set)
sens_AET  = AET1SensitivityMatrix(dra_AET.data_res_arr.settings,
                                  model="scirdv1")
ana_AET   = AnalysisContainer(dra_AET, sens_AET)
snr_AET_lt = ana_AET.snr()

dra_XYZ   = DataResidualArray(hXYZ, input_signal_domain=td_set)
sens_XYZ  = XYZ1SensitivityMatrix(dra_XYZ.data_res_arr.settings,
                                  model="scirdv1")
ana_XYZ   = AnalysisContainer(dra_XYZ, sens_XYZ)
snr_XYZ_lt = ana_XYZ.snr()

print(f"\nlisatools optimal SNR:")
print(f"  AET:  {snr_AET_lt:.6f}")
print(f"  XYZ:  {snr_XYZ_lt:.6f}")

# -----------------------------------------------------------------------------
# 6. FFT / IFFT time-slide matched-filter SNR(tau)
# -----------------------------------------------------------------------------
def correlation_snr_curve(dra, sens, snr_lt, N):
    """Return (lags, SNR(tau)) using the same H, invC, df as lisatools."""
    fd_settings = dra.data_res_arr.settings
    df          = fd_settings.df
    H           = dra.data_res_arr.arr               # (nchannels, Nf_active)
    invC        = sens.invC                          # (...,)+ (Nf_active,)

    # Build the cross-spectral integrand on the *full* one-sided rfft grid
    # (length N//2 + 1) so that irfft gives back length-N time samples.
    Nf_full   = N // 2 + 1
    f0        = fd_settings.ind_min                  # active band start bin
    f1        = fd_settings.ind_max + 1              # active band stop bin

    # Match lisatools' first-NaN-drop behaviour: zero out a bin whose invC
    # is NaN (this is how the LISA sensitivity diverges at f=0).
    if invC.ndim == 2:                               # AET (nchannels, Nf)
        # collapse channel sum first: integrand_active(f) = sum_c H_c* H_c invC_c
        integrand_active = np.sum(np.conj(H) * H * invC, axis=0)
        bad = np.isnan(integrand_active) | ~np.isfinite(integrand_active)
        integrand_active = np.where(bad, 0.0, integrand_active)
    elif invC.ndim == 3:                             # XYZ (3, 3, Nf)
        integrand_active = np.einsum("if,ijf,jf->f",
                                     np.conj(H), invC, H)
        bad = np.isnan(integrand_active) | ~np.isfinite(integrand_active)
        integrand_active = np.where(bad, 0.0, integrand_active)
    else:
        raise ValueError(f"unexpected invC.ndim = {invC.ndim}")

    # Pad zeros outside the FDSettings active band so we can irfft length N
    integrand = np.zeros(Nf_full, dtype=integrand_active.dtype)
    integrand[f0:f1] = integrand_active

    # Time-slide via inverse rDFT.
    # corr[n] = (h|h)(tau_n) for tau_n = n * dt
    # factor of 2 difference with lisatools so adding 1/2 factor
    corr = (1. / 2.) * 4.0 * df * N * np.fft.irfft(integrand, n=N)

    snr_tau = corr / snr_lt   # divide by sqrt<h|h>
    lags    = (np.arange(N) - N // 2) * dt
    return lags, np.fft.fftshift(snr_tau)

lags_AET, snr_AET_tau = correlation_snr_curve(dra_AET, sens_AET, snr_AET_lt, N)
lags_XYZ, snr_XYZ_tau = correlation_snr_curve(dra_XYZ, sens_XYZ, snr_XYZ_lt, N)

# tau = 0 sits at index N // 2 after fftshift
i0 = N // 2

print(f"\nFFT-IFFT cross-correlation SNR(tau = 0):")
print(f"  AET:  {snr_AET_tau[i0]:.6f}    (peak |SNR| = "
      f"{np.max(np.abs(snr_AET_tau)):.6f})")
print(f"  XYZ:  {snr_XYZ_tau[i0]:.6f}    (peak |SNR| = "
      f"{np.max(np.abs(snr_XYZ_tau)):.6f})")

print(f"\nDifference SNR(tau=0) - lisatools SNR:")
print(f"  AET:  {snr_AET_tau[i0] - snr_AET_lt: .3e}")
print(f"  XYZ:  {snr_XYZ_tau[i0] - snr_XYZ_lt: .3e}")

# -----------------------------------------------------------------------------
# 7. Plot
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
ax[0].plot(lags_AET, snr_AET_tau, lw=0.7)
ax[0].axhline( snr_AET_lt, color="gray", ls=":", label=f"lisatools = {snr_AET_lt:.3f}")
ax[0].axhline(-snr_AET_lt, color="gray", ls=":")
ax[0].axvline(0.0, color="k", ls="--", alpha=0.3)
ax[0].set_ylabel("SNR(tau)")
ax[0].set_title("AET (diagonal noise)")
ax[0].legend(loc="upper right")

ax[1].plot(lags_XYZ, snr_XYZ_tau, lw=0.7, color="C1")
ax[1].axhline( snr_XYZ_lt, color="gray", ls=":", label=f"lisatools = {snr_XYZ_lt:.3f}")
ax[1].axhline(-snr_XYZ_lt, color="gray", ls=":")
ax[1].axvline(0.0, color="k", ls="--", alpha=0.3)
ax[1].set_xlabel("time shift tau [s]")
ax[1].set_ylabel("SNR(tau)")
ax[1].set_title("XYZ (full 3x3 CSD)")
ax[1].legend(loc="upper right")

plt.tight_layout()
out_png = os.path.join(os.path.dirname(__file__), "emri_snr_correlation.png")
plt.savefig(out_png, dpi=130)
print(f"\nSaved plot -> {out_png}")

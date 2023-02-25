#!/usr/bin/env python3

import numpy
import numpy.random


def lorentzian_peak(x: numpy.ndarray, center: float, half_width: float
		) -> numpy.ndarray:
	"""
	L(x) = 1/pi * Gamma / ((x - x0)^2 + Gamma^2)

	Gamma = half peak width
	x0 = peak center
	"""
	return half_width / numpy.pi / ((x - center) ** 2 + half_width ** 2)


def get_wavenum(wavenum_low, wavenum_high, wavenum_step, wavenum_jitter,
		wavenum_drift_sigma) -> numpy.ndarray:
	# base wavenumber
	wn = numpy.arange(wavenum_low, wavenum_high, wavenum_step, dtype=float)
	# add drift (float)
	wn += numpy.random.normal(0, wavenum_drift_sigma)
	# add jitter (ndarray)
	wn += numpy.random.normal(0, wavenum_jitter, size=len(wn))
	return wn


def create_mock_dataset(*, n_spectra=20, wavenum_low=400.0, wavenum_high=1800.0,
		wavenum_step=1.5, wavenum_jitter=0.10, wavenum_drift_sigma=5,
		n_peaks=5, peak_intens_poisson_lam=6, peak_width_normal_mu=10,
		peak_width_normal_sigma=3, noise_std=0.05) -> numpy.ndarray:
	# wavenumber used in output
	wavenum = get_wavenum(wavenum_low, wavenum_high, wavenum_step,
		wavenum_jitter, wavenum_drift_sigma)
	# output intensity matrix
	intens = numpy.empty((n_spectra, len(wavenum)), dtype=float)

	# generate high-resolution base curve
	gen_x = numpy.arange(wavenum_low, wavenum_high, 0.01, dtype=float)
	gen_y = numpy.zeros(gen_x.shape, dtype=float)
	for i in range(n_peaks):
		# stack all lorentzian peaks
		center = numpy.random.uniform(wavenum_low, wavenum_high)
		peak_width = numpy.random.normal(peak_width_normal_mu,
			peak_width_normal_sigma)
		peak_intens = numpy.random.poisson(peak_intens_poisson_lam)
		gen_y += lorentzian_peak(gen_x, center, peak_width / 2) * peak_intens

	# generate each spectra
	for v in intens:
		rand_wavenum = get_wavenum(wavenum_low, wavenum_high, wavenum_step,
			wavenum_jitter, wavenum_drift_sigma)
		# interp from the high-resolution base curve then add noise
		rand_intens = numpy.interp(rand_wavenum, gen_x, gen_y) \
			+ numpy.random.normal(0, noise_std, size=len(rand_wavenum))
		# normalize
		v[:] = rand_intens / numpy.linalg.norm(rand_intens, ord=2)

	return numpy.vstack([wavenum, intens])


if __name__ == "__main__":
	n_datasets = 4
	for i in range(n_datasets):
		dataset = create_mock_dataset()
		numpy.savetxt("mock_%02u.tsv" % i, dataset, fmt="%.4f", delimiter="\t")

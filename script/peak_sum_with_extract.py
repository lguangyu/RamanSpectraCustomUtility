#!/usr/bin/env python3

import argparse
import glob
import numpy
import os
import shutil
import sys


def get_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("input", type = str, nargs = "+",
		help = "input files (accept wildcards)")
	ap.add_argument("--low", type = float, required = True,
		metavar = "float",
		help = "lower boundary of threshold-test window (required)")
	ap.add_argument("--high", type = float, required = True,
		metavar = "float",
		help = "higher boundary of threshold-test window (required)")
	ap.add_argument("-t", "--threshold", type = float, required = True,
		metavar = "float", 
		help = "threshold used in threshold-test; a spectrum passes the test if"
			" it has maximum intensity measure within this [low, high] range "
			"passing (being greater-equal than) this threshold (required)")
	ap.add_argument("-x", "--extract-dir", type = str, default = None,
		metavar = "dir",
		help = "copy spectra files that passed threshold-test into this dir; "
			"by default, no copy will be performed")

	# parse and refine args
	args = ap.parse_args()

	return args


class TwoColumnRamanSpectra(object):
	def __init__(self, path: str, shifts, intensities, *ka, **kw):
		super().__init__(*ka, **kw)
		self.path			= path
		self.shifts			= shifts
		self.intensities	= intensities
		return

	@property
	def raw_title(self) -> str:
		"""
		get the file name as title from file path
		"""
		return os.path.basename(self.path)

	def get_range(self, low, high):
		mask = (self.shifts >= low) & (self.shifts <= high)
		return self.shifts[mask], self.intensities[mask]

	@classmethod
	def from_file(cls, fname):
		data = numpy.loadtxt(fname, dtype = float, delimiter = "\t")
		new = cls(
			path		= fname,
			shifts		= data[:, 0],
			intensities	= data[:, 1],
		)
		return new


def load_spectra(paths) -> list:
	ret = list()
	for i in paths:
		for f in glob.glob(i):
			ret.append(TwoColumnRamanSpectra.from_file(f))
	return ret


def prepare_extract_dir(extract_dir = None):
	if extract_dir is None:
		return
	# if not exists, create extract_dir
	# else, make sure it's a directory
	if not os.path.exists(extract_dir):
		os.makedirs(extract_dir)
	elif not os.path.isdir(extract_dir):
		raise FileExistsError("target path '%s' exists but is not a directory"\
			% extract_dir)
	return


def extract_to_dir(s: TwoColumnRamanSpectra, extract_dir = None):
	if extract_dir is None:
		return
	dst = os.path.join(extract_dir, s.raw_title)
	shutil.copy(s.path, dst)
	return


def peak_threshold_test_with_extract(spectra_list, *, range_low: float,
		range_high: float, threshold: float, extract_dir = None) -> dict:
	# prepare the extrct dir, is ok to be None
	prepare_extract_dir(extract_dir)

	# stats for threshold-test
	n_passed	= 0
	peak_sum	= 0.0
	for s in spectra_list:
		range_shift, range_intensity = s.get_range(range_low, range_high)
		range_max = range_intensity.max()
		# threshold test in the range [low, high]
		if range_max >= threshold:
			n_passed += 1
			peak_sum += range_max
			# extract to dir if passed the test
			extract_to_dir(s, extract_dir)

	ret = dict(
		n_passed	= n_passed,
		peak_sum	= peak_sum,
	)
	return ret


def main():
	args = get_args()
	# load spectra data and extract
	spectra = load_spectra(args.input)
	test_res = peak_threshold_test_with_extract(spectra,
		range_low	= args.low,
		range_high	= args.high,
		threshold	= args.threshold,
		extract_dir	= args.extract_dir,
	)
	# output the stats
	print("%u spectra processed" % len(spectra))
	print("%u spectra passed threshold-test" % test_res["n_passed"])
	print("sum intensity sum of peaks passed threshold-test: %.4f"\
		% test_res["peak_sum"])
	return


if __name__ == "__main__":
	main()

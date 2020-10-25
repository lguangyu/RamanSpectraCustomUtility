#!/usr/bin/env python3

import argparse
import functools
import glob
import io
import json
import numpy
import os
import sys
# custom lib
import pylib


NORM_METH = dict()
def add_normalization_method(key: str):
	def decorator(func):
		NORM_METH[key] = func
		return func
	return decorator

@add_normalization_method("l1")
def l1_norm_meth(vec: numpy.ndarray):
	vec /= numpy.linalg.norm(vec, ord = 1)
	return vec

@add_normalization_method("l2")
def l1_norm_meth(vec: numpy.ndarray):
	vec /= numpy.linalg.norm(vec, ord = 2)
	return vec

@add_normalization_method("min_max")
def l1_norm_meth(vec: numpy.ndarray):
	# normalize to min=0, max = 1
	vec -= vec.min()
	vec /= vec.max()
	return vec

def get_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("input", nargs = "+", type = str,
		help = "input Raman spectra as .txt dumps (accept wildcards)")
	ap.add_argument("-n", "--normalize", type = str, default = "none",
		choices = list(NORM_METH.keys()),
		help = "normalization method (default: none)")
	ap.add_argument("-l", "--shift-lower-range", type = float, default = 400.0,
		metavar = "float",
		help = "lower boundary for extraction range (default: 400.0)")
	ap.add_argument("-u", "--shift-upper-range", type = float, default = 1800.0,
		metavar = "float",
		help = "upper boundary for extraction range (default: 1800.0)")
	ap.add_argument("-b", "--bin-size", type = float,
		metavar = "float",
		help = "binning window size (default: not binning)")
	ap.add_argument("-o", "--output", type = str, default = "-",
		metavar = "json",
		help = "write outputs to this <json> file instead of stdout")
	# refine args
	args = ap.parse_args()
	if args.output == "-":
		args.output = sys.stdout
	return args


class RamanSpectra(object):
	@classmethod
	def from_file(cls, file):
		"""
		load a Raman spectra text dump
		"""
		raw = numpy.loadtxt(file, delimiter = "\t", dtype = float)
		new = cls()
		# title
		file_name		= (file.name if isinstance(file, io.IOBase) else file)
		new.title		= file_name.replace(os.path.sep, ".")
		# shift
		new.shift		= raw[:, 0]
		# intensity
		new.intensity	= raw[:, 1]
		return new


	def filter_range(self, low_bound, high_bound):
		mask = numpy.logical_and(self.shift >= low_bound,
			self.shift <= high_bound)
		self.shift = self.shift[mask]
		self.intensity = self.intensity[mask]
		return self


	def bin(self, *, low_bound, high_bound, bin_size = None):
		if bin_size is None:
			# no need to bin, just filter the range
			return self.filter_range(low_bound, high_bound)
		assert bin_size > 0
		# binning
		old_shift = self.shift
		new_shift = numpy.arange(low_bound, high_bound, bin_size, dtype = float)
		new_intens = numpy.zeros(len(new_shift), dtype = float)
		for i, st in enumerate(new_shift):
			# ranges to slice from original shifts
			rng1, rng2 = st, st + bin_size
			mask = numpy.logical_and(old_shift >= rng1, old_shift < rng2)
			if mask.any():
				# intensities in range
				rng_intens = self.intensity[mask].mean()
			else:
				# if no data in range, reset to zero
				print("warning: shift windown %.3f-%.3f contains no data: "
					"reset to 0 (zero)" % (rng1, rng2), file = sys.stderr)
				rng_intens = 0
			new_intens[i] = rng_intens
		# apply new data
		self.shift		= new_shift + (bin_size / 2)
		self.intensity	= new_intens
		return self

	def normalize(self, method):
		normalize_inplace = NORM_METH[method]
		normalize_inplace(self.intensity)
		return self

	@property
	def max_intensity(self):
		return self.intensity.max()
	@property
	def min_intensity(self):
		return self.intensity.max()


class RamanSpectraJSONEncoder(json.JSONEncoder):
	@functools.wraps(json.JSONEncoder.default)
	def default(self, o):
		if isinstance(o, RamanSpectra):
			ret = dict(
				title		= o.title,
				shift		= o.shift.tolist(),
				intensity	= o.intensity.tolist(),	
			)
		else:
			ret = super(RamanSpectraJSONEncoder, self).default(o)
		return ret


def main():
	args = get_args()
	# transform
	data = list()
	for i in args.input:
		for f in glob.glob(i):
			spec = RamanSpectra.from_file(f)
			if (spec.max_intensity == 0) and (spec.min_intensity == 0):
				print("warning: skipped '%s' due to all-zero intensity" % file)
				continue
			spec.bin(low_bound = args.shift_lower_range,
				high_bound = args.shift_upper_range, bin_size = args.bin_size)\
				.normalize(args.normalize)
			data.append(spec)
	# output
	with pylib.file_util.get_fp(args.output, "w") as fp:
		json.dump(data, fp, sort_keys = True, cls = RamanSpectraJSONEncoder)
	return


if __name__ == "__main__":
	main()

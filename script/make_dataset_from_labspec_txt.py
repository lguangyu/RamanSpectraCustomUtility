#!/usr/bin/env python3

import argparse
import glob
import itertools
import numpy
import os
import sys
# custom lib
import pylib


NORM_METH = pylib.registry.get("normalize_1d")


def get_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("input", nargs = "+", type = str,
		help = "input Raman spectra as .txt dumps (accept wildcards)")
	ap.add_argument("-n", "--normalize", type = str,
		default = NORM_METH.default_key,
		choices = NORM_METH.get_key_list(),
		help = "normalization method (default: %s)" % NORM_METH.default_key)
	ap.add_argument("-l", "--min-wavenumber", type = pylib.util.NonNegFloat,
		default = 400.0, metavar = "float",
		help = "min wavenumber of extraction window (default: 400)")
	ap.add_argument("-u", "--max-wavenumber", type = pylib.util.NonNegFloat,
		default = 1800.0, metavar = "float",
		help = "max wavenumber of extraction window (default: 1800)")
	ap.add_argument("-b", "--bin-size", type = float,
		metavar = "float",
		help = "binning window size (default: not binning)")
	ap.add_argument("--mark-hypothetical-noise-flag", action = "store_true",
		help = "test if individual spectra is hypothetical noise based on "
			"in-house build algorithm (this is a testing feature); result will "
			"be stored as 'hypothetical_noise_flag' field in output spectra "
			"(default: no)")
	ap.add_argument("-o", "--output", type = str, default = "-",
		metavar = "json",
		help = "write outputs to this <json> file instead of stdout")
	# refine args
	args = ap.parse_args()
	if args.output == "-":
		args.output = sys.stdout
	return args


class LabSpecTxtDumpRamanSpectra(pylib.raman_spectra.RamanSpectra):
	def __init__(self, *ka, file, **kw):
		super().__init__(*ka, **kw)
		self.file = file
		return

	@classmethod
	def from_labspec_txt_dump(cls, file):
		with pylib.util.get_fp(file, "r") as fp:
			raw		= numpy.loadtxt(fp, delimiter = "\t", dtype = float)
			fname	= fp.name

		new = cls(
			title	= fname.replace(os.path.sep, "."),
			wavnum	= raw[:, 0],
			inten	= raw[:, 1],
			file	= fname,
		)
		return new

	def mark_hypothetical_noise_flag(self):
		mask = (995 <= self.wavnum) & (self.wavnum <= 1010)
		noise_flag = (self.inten[mask] <= 18).all()
		self.hypothetical_noise_flag = noise_flag
		return noise_flag

	def normalize(self, method, *, inplace = False):
		meth = NORM_METH.get(method)
		ret = self if inplace else self.copy()
		ret.set_data(ret.wavnum, meth(ret.inten))
		return ret

	def json_serialize(self):
		ret = super().json_serialize()
		ret["file"] = self.file
		if hasattr(self, "hypothetical_noise_flag"):
			ret["hypothetical_noise_flag"] = bool(self.hypothetical_noise_flag)
		return ret


def load_spectra_with_wildcards(files) -> list:
	ret = list()
	for f in itertools.chain(*[glob.glob(i) for i in files]):
		ret.append(LabSpecTxtDumpRamanSpectra.from_labspec_txt_dump(f))
	return ret


def main():
	args = get_args()
	# load data
	spectra = load_spectra_with_wildcards(args.input)
	# transform
	for s in spectra:
		if args.mark_hypothetical_noise_flag:
			s.mark_hypothetical_noise_flag()
		s.bin(min_wn = args.min_wavenumber, max_wn = args.max_wavenumber,
			bin_size = args.bin_size, inplace = True)
		s.normalize(args.normalize, inplace = True)
	# output
	pylib.json_io.dump_json(spectra, args.output, sort_keys = True)
	return


if __name__ == "__main__":
	main()

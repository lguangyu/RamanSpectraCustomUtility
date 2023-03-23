#!/usr/bin/env python3

import argparse
import os
import sys

# custom lib
import opu_analysis_lib as oal


def get_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("datadir", type=str,
		help="input directory to scan LabSpec txt dumps")
	ap.add_argument("--extension", "-x", type=str, default=".txt",
		metavar="str",
		help="the extension of target files process [.txt]")
	ap.add_argument("--recursive", "-r", action="store_true",
		help="also search subdirectories of <datadir> [no]")
	ap.add_argument("--verbose", "-v", action="store_true",
		help="increase verbosity [off]")
	ap.add_argument("--delimiter", "-d", type=str, default="\t",
		metavar="char",
		help="delimiter in text-based input and output [<tab>]")
	ap.add_argument("--output", "-o", type=str, default="-",
		metavar="tsv",
		help="output dataset file [<stdout>]")

	ag = ap.add_argument_group("binning and normalization")
	ag.add_argument("--bin-size", "-b", type=oal.util.PosFloat, default=None,
		metavar="float",
		help="bin size to reconcile wavenumbers in multiple datasets, if left "
			"default, no binning will be performed [off]")
	ag.add_argument("--wavenum-low", "-L", type=oal.util.PosFloat,
		default=400, metavar="float",
		help="lower boundry of wavenumber of extract for analysis [400]")
	ag.add_argument("--wavenum-high", "-H", type=oal.util.PosFloat,
		default=1800, metavar="float",
		help="higher boundry of wavenumber of extract for analysis [1800]")
	ag.add_argument("--normalize", "-N", type=str,
		default=oal.SpectraDataset.norm_meth.default_key,
		choices=oal.SpectraDataset.norm_meth.list_keys(),
		help="normalize method after loading/binning/filtering dataset [%s]"
			% oal.SpectraDataset.norm_meth.default_key)

	# parse and refine args
	args = ap.parse_args()
	# need to add extension separator (usually .) if not so
	# this is require to make compatibility using os.path.splitext()
	if not args.extension.startswith(os.extsep):
		args.extension = os.extsep + args.extension
	if args.output == "-":
		args.output = sys.stdout
	return args


def iter_file_by_ext(path, ext, *, recursive=False) -> iter:
	for i in os.scandir(path):
		if i.is_dir() and recursive:
			yield from iter_file_by_ext(i, ext, recursive=recursive)
		elif i.is_file() and os.path.splitext(i.path)[1] == ext:
			yield i.path
	return


def read_and_combine_labspec_txt_dumps(path, ext, *, recursive=False,
		delimiter="\t", bin_size=None, wavenum_low=None, wavenum_high=None,
		normalize=oal.SpectraDataset.norm_meth.default_key,
	) -> oal.SpectraDataset:
	# read files in directory
	spectra = [oal.SpectraDataset.from_labspec_txt_dump(i,
		delimiter=delimiter, spectrum_name=os.path.basename(i),
		bin_size=bin_size, wavenum_low=wavenum_low, wavenum_high=wavenum_high)
		for i in iter_file_by_ext(path, ext, recursive=recursive)]
	# concatenate into a single dataset
	dataset = oal.SpectraDataset.concatenate(*spectra)
	return dataset


def main():
	args = get_args()
	dataset = read_and_combine_labspec_txt_dumps(args.datadir, args.extension,
		recursive=args.recursive, delimiter=args.delimiter,
		bin_size=args.bin_size, wavenum_low=args.wavenum_low,
		wavenum_high=args.wavenum_high, normalize=args.normalize)
	dataset.save_file(args.output, delimiter=args.delimiter,
		with_spectra_names=True)
	return


if __name__ == "__main__":
	main()

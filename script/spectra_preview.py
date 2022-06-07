#!/usr/bin/env python3

import argparse
import json
import matplotlib
import matplotlib.pyplot
import numpy
import sys
# custom lib
import mpllayout
import pylib


def get_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("input", type = str, nargs = "?", default = "-",
		help = "Raman spectra data (json)")
	ap.add_argument("--output-prefix", "-P", type = str,
		metavar = "prefix",
		help = "output spectra preview images using this prefix "
			"(default: <input>)")
	ap.add_argument("--dpi", type = pylib.util.PosInt, default = 300,
		metavar = "int",
		help = "output image resolution (default: 300)")
	# parse and refine args
	args = ap.parse_args()
	if args.input == "-":
		args.input = sys.stdin
	if args.output_prefix is None:
		args.output_prefix = "input" if args.input is sys.stdin else args.input
	return args


def create_layout():
	lc = mpllayout.LayoutCreator(
		left_margin		= 0.5,
		right_margin	= 0.5,
		top_margin		= 0.5,
		bottom_margin	= 0.5,
	)

	spectra = lc.add_frame("spectra")
	spectra.set_anchor("bottomleft")
	spectra.set_size(8.0, 3.0)

	layout = lc.create_figure_layout()

	return layout


def plot_spectra(png, *, shift, intensity, dpi = 300):
	# setup layout
	layout = create_layout()
	figure = layout["figure"]

	# plot
	axes = layout["spectra"]
	axes.plot(shift, intensity, linestyle = "-", linewidth = 1.5,
		color = "#4040ff", label = "spectra")
	axes.axhline(0, linestyle = "-", linewidth = 1.0, color = "#c0c0c0")

	# misc
	axes.set_xlim(400, 2000)
	axes.set_ylim(-20, max(intensity) * 1.05)
	axes.set_xlabel(r"wavenumber (cm$^{-1}$)", fontsize = 14)
	axes.set_ylabel(r"intensity", fontsize = 14)

	# save and clean-up
	figure.savefig(png, dpi = dpi)
	matplotlib.pyplot.close()
	return


def main():
	args = get_args()
	# load data
	data = pylib.json_io.load_json(args.input)
	# plot all spectra in dataset
	for s in data:
		png = "%s%s.png" % (args.output_prefix, s["title"])
		plot_spectra(png,
			shift = s["shift"],
			intensity = s["intensity"],
			dpi = args.dpi,
		)
	return


if __name__ == "__main__":
	main()


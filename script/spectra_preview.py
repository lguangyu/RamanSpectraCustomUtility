#!/usr/bin/env python3

import argparse
import json
import matplotlib
import matplotlib.pyplot
import numpy
import os
import sys
# custom lib
import pylib


def get_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("input", type = str, nargs = "?", default = "-",
		help = "Raman spectra data (json)")
	ap.add_argument("--output-prefix", "-P", type = str,
		metavar = "prefix",
		help = "output spectra preview images using this prefix "
			"(default: <input>)")
	# parse and refine args
	args = ap.parse_args()
	if args.input == "-":
		args.input = sys.stdin
	if args.output_prefix is None:
		args.output_prefix = "input" if args.input is sys.stdin else args.input
	return args


def load_raman_spectra_json(file):
	with pylib.util.get_fp(file, "r") as fp:
		ret = json.load(fp)
	return ret


def setup_layout(figure):
	# layout
	layout = dict()
	layout["figure"] = figure
	# margins
	left_margin_inch	= 0.5
	right_margin_inch	= 0.5
	bottom_margin_inch	= 0.5
	top_margin_inch		= 0.5
	# spectra axes dimensions
	spec_width_inch		= 8.0
	spec_height_inch	= 3.0
	# figure dimensions
	figure_width_inch	= left_margin_inch + spec_width_inch + right_margin_inch
	figure_height_inch	= bottom_margin_inch + spec_height_inch\
		+ top_margin_inch
	figure.set_size_inches(figure_width_inch, figure_height_inch)
	# spectra axes
	spec_width	= spec_width_inch / figure_width_inch
	spec_height	= spec_height_inch / figure_height_inch
	spec_left	= left_margin_inch / figure_width_inch
	spec_bottom	= bottom_margin_inch / figure_height_inch
	spec = figure.add_axes([spec_left, spec_bottom, spec_width, spec_height])
	layout["spectra"] = spec

	return layout


def plot_spectra(png, *, shift, intensity, dpi = 300):
	# setup layout
	figure = matplotlib.pyplot.figure()
	layout = setup_layout(figure)

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
	matplotlib.pyplot.savefig(png, dpi = dpi)
	matplotlib.pyplot.close()
	return


def main():
	args = get_args()
	# load data
	data = load_raman_spectra_json(args.input)
	# plot all spectra in dataset
	for s in data:
		png = "%s%s.png" % (args.output_prefix, s["title"])
		plot_spectra(png, shift = s["shift"], intensity = s["intensity"])
	return


if __name__ == "__main__":
	main()


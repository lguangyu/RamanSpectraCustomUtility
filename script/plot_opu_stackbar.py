#!/usr/bin/env python3

import argparse
import collections
import matplotlib
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot
import numpy
import sys
# custom lib
import mpllayout
import pylib


def get_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("input", type = str, nargs = "?", default = "-",
		help = "input opu config json")
	ap.add_argument("-c", "--opu", type = str, required = True,
		metavar = "txt",
		help = "the 3-column opu clustering result, containing the opu labels "
			"(required)")
	ap.add_argument("-p", "--plot", type = str, default = "-",
		metavar = "png", 
		help = "the output image (default: <stdout>)")
	ap.add_argument("--dpi", type = pylib.util.PosInt, default = 300,
		metavar = "int",
		help = "output image resolution (default: 300)")

	# refine args
	args = ap.parse_args()
	if args.input == "-":
		args.input = sys.stdin
	if args.plot == "-":
		args.plot = sys.stdout.buffer
	return args


def load_opu_result(file) -> dict:
	title	= list()
	opu		= list()
	group	= list()
	with pylib.util.get_fp(file, "r") as fp:
		for line in fp:
			t, o, g, *_ = line.rstrip().split("\t")
			title.append(t)
			opu.append(int(o))
			group.append(g)
	ret = dict(
		title	= title,
		opu		= opu,
		group	= group,
	)
	return ret


def count_group_otu_abund(opures) -> (dict, dict):
	count = collections.defaultdict(collections.Counter)
	for o, g in zip(opures["opu"], opures["group"]):
		count[g][o] += 1
	perc = collections.defaultdict(dict)
	for g, v in count.items():
		group_total = sum(v.values())
		for o, c in v.items():
			perc[g][o] = c / group_total
	return count, perc




def sort_top_opus(opures, max_opus = None) -> list:
	count = collections.Counter(opures["opu"])
	sorted_opu = sorted(count.items(), key = lambda x: x[1], reverse = True)
	# remove singleton opus (opus having only one member)
	# eliminate count, only return the opu label
	ret = [i[0] for i in sorted_opu if i[1] >= 2]
	# cut the list if max_opus is set
	if max_opus is not None:
		ret = ret[:max_opus]
	return ret


def create_layout(n_group):
	lc = mpllayout.LayoutCreator(
		left_margin		= 0.7,
		right_margin	= 1.5,
		top_margin		= 0.5,
		bottom_margin	= 2.0,
	)

	axes = lc.add_frame("axes")
	axes.set_anchor("bottomleft")
	axes.set_size(0.2 * n_group, 3.0)

	# create layout
	layout = lc.create_figure_layout()

	# apply axes style
	axes = layout["axes"]
	for sp in axes.spines.values():
		sp.set_visible(False)
	axes.set_facecolor("#f0f0f8")
	axes.tick_params(
		left = True, labelleft = True,
		right = False, labelright = False,
		bottom = True, labelbottom = True,
		top = False, labeltop = False
	)

	return layout


class CyclicIndexedList(list):
	def __getitem__(self, index):
		return super().__getitem__(index % len(self))


def get_class_colors_list():
	# get colors prototype
	proto = matplotlib.cm.get_cmap("Set3").colors\
		+ matplotlib.cm.get_cmap("Set2").colors
		#+ matplotlib.cm.get_cmap("Accent").colors[:-1]
		#+ matplotlib.cm.get_cmap("Set3").colors\
		#+ matplotlib.cm.get_cmap("Set2").colors\
	# translate to hex colors
	proto = [matplotlib.colors.to_hex(c) for c in proto]
	# remove identical colors
	ret, proto_set = CyclicIndexedList([]), set()
	for c in proto:
		if c not in proto_set:
			ret.append(c)
			proto_set.add(c)
	return ret


def plot_stackbar(png, *, groups, opuperc, plotopu, dpi = 300):
	n_group = len(groups)

	# create layout
	layout = create_layout(n_group)
	figure = layout["figure"]

	# plot stackbars
	opu_color_list = get_class_colors_list()
	handles = list()
	bottom = numpy.zeros(n_group, dtype = float)
	x = numpy.arange(n_group) + 0.5
	# plot major opus
	axes = layout["axes"]
	for o in (plotopu):
		h = [opuperc[g["name"]].get(o, 0.0) for g in groups]
		bar = axes.bar(x, h, width = 0.8, bottom = bottom, align = "center",
			edgecolor = "none", facecolor = opu_color_list[o],
			label = "OPU_%u" % o
		)
		bottom += h
		handles.append(bar)
	# plot aggregated minor opus
	plotopu = set(plotopu)
	h = numpy.zeros(n_group, dtype = float)
	for i, g in enumerate(groups):
		for o, v in opuperc[g["name"]].items():
			if (o not in plotopu):
				h[i] += v
	# only plot if mh is not all 0
	if (h > 0.0).any():
		bar = axes.bar(x, h, width = 0.8, bottom = bottom, align = "center",
			edgecolor = "none", facecolor = "#d0d0d0",
			label = "other minior",
		)
		bottom += h
		handles.append(bar)

	# legend
	axes.legend(handles = handles, loc = 2, bbox_to_anchor = (1.02, 1.02),
		fontsize = 10, handlelength = 0.8, frameon = False,
	)

	# misc
	axes.set_xlim(0, n_group)
	axes.set_ylim(0.0, 1.0)
	axes.set_ylabel("OPU abundance", fontsize = 12)
	axes.set_xticks(x)
	axes.set_xticklabels([g["name"] for g in groups], fontsize = 10,
		rotation = 90
	)

	# save fig and clean up
	figure.savefig(png, dpi = dpi)
	matplotlib.pyplot.close()
	return


def main():
	args = get_args()
	# load group configuration
	groups = pylib.json_io.load_json(args.input)
	# load opu results
	opures = load_opu_result(args.opu)
	# count the distributions of opus in each group
	opucount, opuperc = count_group_otu_abund(opures)
	# calculate percentage of each opu in each group
	# filter and show only non-singleton opus
	plotopu = sort_top_opus(opures)
	# plot
	plot_stackbar(args.plot,
		groups = groups,
		opuperc = opuperc,
		plotopu = plotopu,
	)
	return


if __name__ == "__main__":
	main()

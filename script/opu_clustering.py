#!/usr/bin/env python3

import argparse
import matplotlib
import matplotlib.cm
import matplotlib.colors
import matplotlib.patches
import matplotlib.pyplot
import numpy
import scipy
import scipy.cluster
import sklearn
import sklearn.cluster
import sys
# custom lib
import mpllayout
import pylib


METRIC_METH = pylib.registry.get("cluster_metric")
CUTOFFOPT_METH = pylib.registry.get("hca_cutoff_optimizer")


def get_args():
	def cutoff_threshold_type(v):
		if v in CUTOFFOPT_METH.get_key_list():
			return v
		return pylib.util.NonNegFloat(v)

	ap = argparse.ArgumentParser()
	ap.add_argument("input", type = str, nargs = "?", default = "-",
		help = "input group config json")
	ap.add_argument("-m", "--metric", type = str,
		default = METRIC_METH.default_key,
		choices = METRIC_METH.get_key_list(),
		help = "metric for HCA (default: %s)" % METRIC_METH.default_key)
	ap.add_argument("-t", "--cutoff-threshold",
		type = cutoff_threshold_type,
		default = 0.7, metavar = "float or " +\
			(" ").join(["'%s'" % i for i in CUTOFFOPT_METH.get_key_list()]),
		help = "OPU cutoff threshold HCA (default: 0.7)")
	ap.add_argument("-o", "--output", type = str, default = "-",
		metavar = "tsv",
		help = "output 2-column tsv to list the cluster label for each spectra "
			"(default: <stdout>)")
	ap.add_argument("-p", "--plot", type = str, metavar = "png",
		help = "if set, also output a metric heatmap and HCA dendrogram plot "
			"image (default: no)")
	ap.add_argument("--dpi", type = pylib.util.PosInt, default = 300,
		metavar = "int",
		help = "output image resolution (default: 300)")
	ap.add_argument("--with-hypothetical-noise-flag", action = "store_true",
		help = "mark hypothetical noise flag in both tsv and plot outputs; "
			"enabling this feature assumes the input data is generated with "
			"--mark-hypothetical-noise-flag enabled and each spectra is "
			"marked if being noise individually "
			"(see script make_dataset_from_labspec_txt.py); if not, all "
			"spectra missing this hypothetical noise information "
			"are assumed to be non-noise spctra (default: no)")

	# refine args
	args = ap.parse_args()
	if args.input == "-":
		args.input = sys.stdin
	if args.output == "-":
		args.output = sys.stdout
	return args


def load_spectra_data_by_group_cfg(group_cfg_list: list):
	color	= list()
	group	= list()
	title	= list()
	inten	= list()
	wavnum	= None
	noise_flag = list()

	for cfg in group_cfg_list:
		for s in pylib.json_io.load_json(cfg["file"]):
			color.append(cfg["color"]) # sample label color
			group.append(cfg["name"]) # sample group name
			title.append(s["title"]) # sample title
			inten.append(s["inten"]) # sample intensity data
			noise_flag.append(s.get("hypothetical_noise_flag", False))
		wavnum = s["wavnum"] # wavnum is shared by all spectra

	ret = dict(
		color	= numpy.asarray(color, dtype = object),
		group	= numpy.asarray(group, dtype = object),
		title	= numpy.asarray(title, dtype = object),
		inten	= numpy.asarray(inten, dtype = float),
		wavnum	= numpy.asarray(wavnum, dtype = float),
		noise_flag = numpy.asarray(noise_flag, dtype = bool),
	)
	return ret


class OneWayHCAWithPlot(object):
	def __init__(self, *ka, metric, cutoff_target, linkage = "average",
			dendrogram_orientation = "right", **kw):
		super().__init__(*ka, **kw)
		self.metric = metric
		self.cutoff_target = cutoff_target
		self.cutoff_opt = CUTOFFOPT_METH.get(cutoff_target)
		self.dendrogram_orientation = dendrogram_orientation
		self._hca = sklearn.cluster.AgglomerativeClustering(
			linkage = linkage, affinity = "precomputed",
			distance_threshold = 0, n_clusters = None
		)
		return

	@property
	def cutoff(self) -> str or float:
		return self.cutoff_opt.cutoff_final
	@property
	def linkage(self) -> str:
		return self._hca.linkage
	@property
	def n_leaves(self) -> int:
		return self._hca.n_leaves_
	@property
	def labels(self) -> int:
		return self._hca.labels_
	@property
	def n_clusters(self) -> int:
		return self.labels.max() + 1

	@staticmethod
	def _calc_linkage_matrix(hca):
		# this function is adapted from 'https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html' as of version 1.1.1

		# create the counts of samples under each node
		counts = numpy.zeros(hca.children_.shape[0])
		n_samples = len(hca.labels_)
		for i, merge in enumerate(hca.children_):
			current_count = 0
			for child_idx in merge:
				if child_idx < n_samples:
					current_count += 1 # leaf node
				else:
					current_count += counts[child_idx - n_samples]
			counts[i] = current_count

		linkage_matrix = numpy.column_stack(
			[hca.children_, hca.distances_, counts]
		).astype(float)
		return linkage_matrix

	def _optimize_cutoff(self, n_step) -> float:
		# this function must be called after self.data_mat, self.metric_meth,
		# self.dist_mat, and self._hca has been setup.
		# in usual cases, it should be called internally by self.feed()
		cutoff_list = numpy.linspace(
			self.dist_mat.min(),
			self.dist_mat.max(),
			n_step
		)
		self.cutoff_opt.optimize(
			model			= self._hca,
			data			= self.data_mat,
			dist			= self.dist_mat,
			cutoff_list		= cutoff_list,
			cutoff_final	= self.cutoff_target
		)
		return self.cutoff


	def feed(self, data, *, title_list):
		if len(data) != len(title_list):
			raise ValueError("data and title_list lengths mismatch")
		self.data_mat = data
		self.title_list = title_list
		# calculate distance matrix
		self.metric_meth = METRIC_METH.get(self.metric)
		self.dist_mat = self.metric_meth(data)
		# find the cutoff
		self._optimize_cutoff(n_step = 100)
		# calculate clusters, using sklearn's backend
		self._hca.set_params(distance_threshold = self.cutoff)
		self._hca.fit(self.dist_mat)
		# calculate linkage matrix
		self.linkage_matrix = self._calc_linkage_matrix(self._hca)
		# make dendrogram using scipy's backend
		self.dendrogram = scipy.cluster.hierarchy.dendrogram(
			self.linkage_matrix, orientation = self.dendrogram_orientation,
			no_plot = True
		)
		return self

	def save_labels(self, file, *aux_fields, delimiter = "\t"):
		with pylib.util.get_fp(file, "w") as fp:
			for i in range(len(self.title_list)):
				fields = [self.title_list[i], self.labels[i]]
				fields.extend([str(f[i]) for f in aux_fields])
				print(delimiter.join([str(i) for i in fields]), file = fp)
		return

	def plot_heatmap(self, heatmap_axes, colorbar_axes) -> dict:
		# heatmap
		ax = heatmap_axes
		heatmap_data = self.metric_meth.to_plot_data(self.dist_mat)
		pcolor = ax.pcolor(heatmap_data[numpy.ix_(self.dendrogram["leaves"],
			self.dendrogram["leaves"])], cmap = self.metric_meth.cmap,
			vmin = self.metric_meth.vmin, vmax = self.metric_meth.vmax)
		#
		ax.set_xlim(0, self.n_leaves)
		ax.set_ylim(0, self.n_leaves)

		# colorbar
		ax = colorbar_axes
		cbar = colorbar_axes.figure.colorbar(pcolor, cax = ax,
			ticklocation = "bottom", orientation = "horizontal")
		# misc
		cbar.outline.set_visible(False)
		cbar.set_label(self.metric_meth.name_str, fontsize = 14)

		ret = dict(heatmap = pcolor, colorbar = cbar)
		return ret

	def _dendro_get_adjusted_dmax(self, dmax, i2d_ratio) -> float:
		return dmax / (1 - 1 / (2 * self.n_leaves * i2d_ratio))

	def plot_dendrogram(self, dendro_axes, *, i2d_ratio: float) -> list:
		lines = list() # the return list containing all lines drawn
		ax = dendro_axes
		for xys in zip(self.dendrogram["dcoord"], self.dendrogram["icoord"]):
			line = matplotlib.lines.Line2D(*xys, linestyle = "-",
				linewidth = 1.0, color = "#4040ff", zorder = 3)
			ax.add_line(line)
			lines.append(line)
		# misc
		ax.grid(axis = "x", linestyle = "-", linewidth = 1.0, color = "#ffffff",
			zorder = 2)
		ax.set_xlim(0, self._dendro_get_adjusted_dmax(
			dmax = numpy.max(self.dendrogram["dcoord"]),
			i2d_ratio = i2d_ratio)
		)
		ax.set_ylim(0, 10 * self.n_leaves)

		return lines


def create_layout(add_noise_bar = False):
	lc = mpllayout.LayoutCreator(
		left_margin		= 0.2,
		right_margin	= 0.2,
		top_margin		= 1.0,
		bottom_margin	= 0.5,
	)

	pbar_width		= 0.6
	group_bar_width	= 0.2
	noise_bar_width	= 0.2
	cbar_height		= 0.4
	heatmap_size	= 8.0
	dendro_width	= 2.5
	axes_gap		= 0.1

	pbar_l = lc.add_frame("pbar_l")
	pbar_l.set_anchor("bottomleft", offsets = (0, pbar_width + axes_gap))
	pbar_l.set_size(pbar_width, heatmap_size)

	heatmap = lc.add_frame("heatmap")
	heatmap.set_anchor("bottomleft", ref_frame = pbar_l,
		ref_anchor = "bottomright", offsets = (axes_gap, 0))
	heatmap.set_size(heatmap_size, heatmap_size)

	colorbar = lc.add_frame("colorbar")
	colorbar.set_anchor("topleft", ref_frame = heatmap,
		ref_anchor = "bottomleft", offsets = (0, -axes_gap))
	colorbar.set_size(heatmap_size, cbar_height)

	pbar_r = lc.add_frame("pbar_r")
	pbar_r.set_anchor("bottomleft", ref_frame = heatmap,
		ref_anchor = "bottomright", offsets = (axes_gap, 0))
	pbar_r.set_size(pbar_width, heatmap_size)

	group_bar = lc.add_frame("group_bar")
	group_bar.set_anchor("bottomleft", ref_frame = pbar_r,
		ref_anchor = "bottomright", offsets = (axes_gap / 2, 0))
	group_bar.set_size(group_bar_width, heatmap_size)

	if add_noise_bar:
		noise_bar = lc.add_frame("noise_bar")
		noise_bar.set_anchor("bottomleft", ref_frame = group_bar,
			ref_anchor = "bottomright", offsets = (axes_gap / 2, 0))
		noise_bar.set_size(noise_bar_width, heatmap_size)

	dendro = lc.add_frame("dendro")
	dendro.set_anchor("bottomleft", 
		ref_frame = (noise_bar if add_noise_bar else group_bar),
		ref_anchor = "bottomright", offsets = (axes_gap, 0))
	dendro.set_size(dendro_width, heatmap_size)

	# create layout
	layout = lc.create_figure_layout()
	layout["dendro_i2d"] = dendro.get_width() / dendro.get_height()

	# apply axes style
	for n in ["pbar_l", "colorbar", "pbar_r", "group_bar", "noise_bar", "dendro"]:
		axes = layout.get(n, None)
		if axes is not None:
			for sp in axes.spines.values():
				sp.set_visible(False)
			axes.set_facecolor("#f0f0f8")

	for n in ["pbar_l", "heatmap", "colorbar", "pbar_r", "group_bar", "noise_bar"]:
		axes = layout.get(n, None)
		if axes is not None:
			axes.tick_params(
				left = False, labelleft = False,
				right = False, labelright = False,
				bottom = False, labelbottom = False,
				top = False, labeltop = False
			)
	layout["dendro"].tick_params(
		left = False, labelleft = False,
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


def plot_hca_label_pbar(axes, hca, *, color_list: CyclicIndexedList):
	for i, leaf in enumerate(hca.dendrogram["leaves"]):
		label = hca.labels[leaf]
		patch = matplotlib.patches.Rectangle((0, i), 1, 1,
			edgecolor = "none", facecolor = color_list[label]
		)
		axes.add_patch(patch)
	axes.set_xlim(0, 1)
	axes.set_ylim(0, hca.n_leaves)
	return


def plot_hca_group_bar(axes, hca, *, group_color_list):
	for i, leaf in enumerate(hca.dendrogram["leaves"]):
		patch = matplotlib.patches.Rectangle((0, i), 1, 1,
			edgecolor = "none", facecolor = group_color_list[leaf]
		)
		axes.add_patch(patch)
	axes.set_xlim(0, 1)
	axes.set_ylim(0, hca.n_leaves)
	return


def plot_hca_noise_bar(axes, hca, *, noise_flag_list):
	for i, leaf in enumerate(hca.dendrogram["leaves"]):
		if noise_flag_list[leaf]:
			patch = matplotlib.patches.Rectangle((0, i), 1, 1,
				edgecolor = "none", facecolor = "#606060",
			)
			axes.add_patch(patch)
	axes.set_xlim(0, 1)
	axes.set_ylim(0, hca.n_leaves)
	return


def plot_hca(png, hca, *, spectra_data, plot_noise_flag = False,
		dpi = 300):
	# skip plotting if png is None
	if png is None:
		return

	layout = create_layout(add_noise_bar = plot_noise_flag)
	figure = layout["figure"]

	# plot heatmap
	axes = layout["heatmap"]
	hca.plot_heatmap(axes, layout["colorbar"])
	# misc
	axes.set_title("OPU clustering (hierarchical)\n"
		"metric=%s; linkage=%s; cutoff=%s; #clusters=%u"\
		% (hca.metric_meth.name_str, hca.linkage,
		hca.cutoff_opt.cutoff_final_str, hca.n_clusters),
		fontsize = 16
	)

	# plot dendrogram
	axes = layout["dendro"]
	hca.plot_dendrogram(axes, i2d_ratio = layout["dendro_i2d"])
	axes.axvline(hca.cutoff, linestyle = "-", linewidth = 1.0,
		color = "#ff0000", zorder = 4)

	# plot pbar
	label_color_list = get_class_colors_list()
	plot_hca_label_pbar(layout["pbar_l"], hca, color_list = label_color_list)
	plot_hca_label_pbar(layout["pbar_r"], hca, color_list = label_color_list)

	# plot group bar
	plot_hca_group_bar(layout["group_bar"], hca,
		group_color_list = spectra_data["color"])

	# plot hypothetical noise bar
	if plot_noise_flag:
		plot_hca_noise_bar(layout["noise_bar"], hca,
			noise_flag_list = spectra_data["noise_flag"])


	# save fig and clean up
	figure.savefig(png, dpi = dpi)
	matplotlib.pyplot.close()
	return


def main():
	args = get_args()
	# load group configuration
	groups = pylib.json_io.load_json(args.input)
	# load spectra data into 'groups', inplace
	data = load_spectra_data_by_group_cfg(groups)
	# run hca
	hca = OneWayHCAWithPlot(
		metric = args.metric,
		cutoff_target = args.cutoff_threshold,
		linkage = "average",
		dendrogram_orientation = "right",
	)
	hca.feed(data["inten"], title_list = data["title"])
	# output hca labels
	aux_fields = list()
	if args.with_hypothetical_noise_flag:
		aux_fields.append(data["noise_flag"])
	hca.save_labels(args.output, *aux_fields)

	# plot hca
	plot_hca(args.plot, hca,
		spectra_data = data,
		plot_noise_flag = args.with_hypothetical_noise_flag,
		dpi = args.dpi)
	return


if __name__ == "__main__":
	main()

#!/usr/bin/env python3

import argparse
import itertools
import json
import matplotlib
import matplotlib.cm
import matplotlib.patches
import matplotlib.pyplot
import numpy
import scipy
import scipy.cluster
import scipy.spatial
import sklearn
import sklearn.cluster
import sklearn.decomposition
import sklearn.metrics
import sys
# custom lib
import pylib


HEATMAP_QUALITY = {
	"preconfig": {
		"low": {
			"dpi": 75,
			"cell_size": 0.02,
		},
		"medium": {
			"dpi": 150,
			"cell_size": 0.05,
		},
		"high": {
			"dpi": 300,
			"cell_size": 0.08,
		},
	},
	"default": "high",
}

METRICS = dict()
def _metric_euclidean(X):
	ret = sklearn.metrics.pairwise.pairwise_distances(X,
		metric = "euclidean")
	return ret
METRICS["euclidean"] = {"method": _metric_euclidean, "max": 1.4142}
def _metric_corrdist_cossim(X):
	r_mat = numpy.corrcoef(X, rowvar = True)
	ret = numpy.sqrt(2 * (1.0 - r_mat))
	return ret
METRICS["corrdist_cossim"] = {"method": _metric_corrdist_cossim, "max": 2.0}
def _metric_corrdist_matlab(X):
	r_mat = numpy.corrcoef(X, rowvar = True)
	ret = 2 * (1.0 - r_mat)
	return ret
METRICS["corrdist_matlab"] = {"method": _metric_corrdist_matlab, "max": 2.0}


def get_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("input", type = str, nargs = "?", default = "-",
		help = "input group config json")
	ap.add_argument("-m", "--metric", type = str, default = "euclidean",
		choices = sorted(METRICS.keys()),
		help = "distance metric (default: euclidean)")
	ap.add_argument("-t", "--cutoff-threshold", type = float, default = 0.7,
		metavar = "float",
		help = "cutoff threshold in hierarchical clustering (default: 0.7)")
	ap.add_argument("--list-cluster-labels", action = "store_true",
		help = "print clustering results (default: off)")
	# add figure omit options
	for i in ["heatmap", "stack-bar", "pca"]:
		ap.add_argument("--no-plot-%s" % i, action = "store_true",
			help = "do not make the %s figure (default: off)" % i)
	ap.add_argument("--without-pca-labels", action = "store_true",
		help = "generate a plain unlabeled pca figure (default: off)")
	ap.add_argument("-P", "--plot-prefix", type = str,
		metavar = "png",
		help = "plot to <png> file using this prefix (default: <input>)")
	ap.add_argument("--heatmap-quality", type = str,
			default = HEATMAP_QUALITY["default"],
			choices = sorted(HEATMAP_QUALITY["preconfig"].keys()),
			help = "quality of heatmap plot output (default: %s)"\
				% HEATMAP_QUALITY["default"])
	# refine args
	args = ap.parse_args()
	if args.input == "-":
		args.input = sys.stdin
	if (args.plot_prefix is None):
		args.plot_prefix = "input" if args.input is sys.stdin else args.input
	return args


def load_json(fjson, *ka, **kw):
	with pylib.file_util.get_fp(fjson, "r") as fp:
		ret = json.load(fp, *ka, **kw)
	return ret


def make_sample_2d_data(group_config):
	color		= list()
	group		= list()
	intensity	= list()
	sample		= list()
	for c in group_config:
		for s in c["data"]:
			color.append(c["color"])
			group.append(c["name"])
			intensity.append(s["intensity"])
			sample.append(s["title"])
	ret = dict(
		color		= color,
		group		= group,
		intensity	= numpy.asarray(intensity, dtype = float),
		sample		= sample,
	)
	return ret


def get_dist_matrix(X, metric = "euclidean"):
	# X must be 2d array
	X = numpy.asarray(X, dtype = float)
	if X.ndim != 2:
		raise ValueError("X must be 2d array, got '%u'd" % X.ndim)
	n = len(X) # number of samples
	# different metrics
	metric_meth = METRICS[metric]["method"]
	ret = metric_meth(X)
	return ret


class HierarchicalCluster(object):
	def __init__(self, *ka, **kw):
		super(HierarchicalCluster, self).__init__(*ka, **kw)
		self.hca		= None
		self.dendrogram	= None
		self.dist_mat	= None
		self.dist_sq	= None
		self.linkage	= None
		self.metric		= None
		return

	@classmethod
	def from_dist_matrix(cls, dist_mat, *, linkage = "average",
			cutoff = 0.7):
		# return object
		ret = cls()
		ret.dist_mat	= dist_mat
		ret.dist_sq		= scipy.spatial.distance.squareform(dist_mat,
			checks = False)
		# make linkage
		ret.linkage		= scipy.cluster.hierarchy.linkage(ret.dist_sq, 
			method = linkage, optimal_ordering = True)
		# make dendrogram
		ret.dendrogram	= scipy.cluster.hierarchy.dendrogram(ret.linkage,
			orientation = "right", no_plot = True, distance_sort = True)
		# now use sklearn to find the clustering
		ret.hca = sklearn.cluster.AgglomerativeClustering(linkage = linkage,
			affinity = "precomputed", distance_threshold = cutoff,
			n_clusters = None)
		ret.hca.fit(ret.dist_mat)
		return ret

	@property
	def n_clusters(self):
		return max(self.hca.labels_) + 1


def apply_frameless_style(axes):
	for sp in axes.spines.values():
		sp.set_visible(False)
	axes.set_facecolor("#E0E0F0")
	axes.tick_params(left = False, right = False, top = False, bottom = False,
		labelleft = False, labelright = False,
		labeltop = False, labelbottom = False)
	return


def get_class_colors_list():
	ret = matplotlib.cm.get_cmap("Set3").colors\
		+ matplotlib.cm.get_cmap("Set2").colors
		#+ matplotlib.cm.get_cmap("Accent").colors[:-1]
		#+ matplotlib.cm.get_cmap("Set3").colors\
		#+ matplotlib.cm.get_cmap("Set2").colors\
	return ret * 100 # expand the list with replicates


def draw_heatmap_cluster_boxes():
	pass

def get_axes_hwratio(axes) -> float:
	if not isinstance(axes, matplotlib.axes.Axes):
		raise TypeError("axes must be matplotlib.axes.Axes, not '%s'"\
			% type(axes).__name__)
	trans = axes.transAxes
	wmin, hmin = trans.transform((0, 0))
	wmax, hmax = trans.transform((1, 1))
	return (hmax - hmin) / (wmax - wmin)


def plot_heatmap(png, *, sample_data, clustering, metric = "unknown",
		quality = HEATMAP_QUALITY["default"]):
	quality_config = HEATMAP_QUALITY["preconfig"][quality]
	# layout
	# margin
	left_margin_inch	= 0.5
	right_margin_inch	= 0.5
	bottom_margin_inch	= 0.5
	top_margin_inch		= 1.0
	interspace_inch		= 0.2
	# heatmap
	n_samples			= len(clustering.dist_mat)
	heatmap_cell_inch	= quality_config["cell_size"]
	heatmap_label_inch	= 1.80 # space for texts at left/bottom
	heatmap_size_inch	= n_samples * heatmap_cell_inch
	# cluster bar
	clstbar_width_inch	= 1.0
	clstbar_height_inch	= heatmap_size_inch
	# dendrogram
	dendro_width_inch	= 6.0
	dendro_height_inch	= heatmap_size_inch

	# create figure
	figure_width_inch	= left_margin_inch + heatmap_label_inch\
		+ interspace_inch + clstbar_width_inch + interspace_inch\
		+ heatmap_size_inch + interspace_inch + clstbar_width_inch\
		+ interspace_inch + dendro_width_inch + right_margin_inch
	figure_height_inch	= bottom_margin_inch + heatmap_label_inch\
		+ interspace_inch + heatmap_size_inch + top_margin_inch
	figure = matplotlib.pyplot.figure(
		figsize = (figure_width_inch, figure_height_inch))

	# heatmap axes
	heatmap_left		= (left_margin_inch + heatmap_label_inch\
		+ interspace_inch + clstbar_width_inch + interspace_inch)\
		/ figure_width_inch
	heatmap_bottom		= (bottom_margin_inch + heatmap_label_inch\
		+ interspace_inch) / figure_height_inch
	heatmap_width		= heatmap_size_inch / figure_width_inch
	heatmap_height		= heatmap_size_inch / figure_height_inch
	heatmap = figure.add_axes([heatmap_left, heatmap_bottom,
		heatmap_width, heatmap_height])
	apply_frameless_style(heatmap)

	# left clstbar axes
	clstbar_width		= clstbar_width_inch / figure_width_inch
	clstbar_height		= clstbar_height_inch / figure_height_inch
	left_clstbar_left	= (left_margin_inch + heatmap_label_inch\
		+ interspace_inch) / figure_width_inch
	left_clstbar_bottom	= heatmap_bottom # align to heatmap
	left_clstbar = figure.add_axes([left_clstbar_left, left_clstbar_bottom,
		clstbar_width, clstbar_height])
	apply_frameless_style(left_clstbar)

	# right clstbar axes
	right_clstbar_left		= (left_margin_inch + heatmap_label_inch\
		+ interspace_inch + clstbar_width_inch + interspace_inch\
		+ heatmap_size_inch + interspace_inch) / figure_width_inch
	right_clstbar_bottom	= heatmap_bottom # align to heatmap
	right_clstbar = figure.add_axes([right_clstbar_left, right_clstbar_bottom,
		clstbar_width, clstbar_height])
	apply_frameless_style(right_clstbar)

	# dendrogram axes
	dendro_left			= (left_margin_inch + heatmap_label_inch\
		+ interspace_inch + clstbar_width_inch + interspace_inch\
		+ heatmap_size_inch + interspace_inch + clstbar_width_inch\
		+ interspace_inch) / figure_width_inch
	dendro_bottom		= heatmap_bottom # align to heatmap
	dendro_width		= dendro_width_inch / figure_width_inch
	dendro_height		= dendro_height_inch / figure_height_inch
	dendrogram = figure.add_axes([dendro_left, dendro_bottom, dendro_width,
		dendro_height])
	apply_frameless_style(dendrogram)
	# re-enable labels at bottom
	dendrogram.tick_params(bottom = True, labelbottom = True)

	# plot heatmap
	cscale			= matplotlib.cm.get_cmap("gnuplot2_r") # color map
	sort_idx		= clustering.dendrogram["leaves"]
	sorted_dist_mat	= clustering.dist_mat[numpy.ix_(sort_idx, sort_idx)]
	heatmap_plot	= heatmap.pcolor(sorted_dist_mat, cmap = cscale,
		vmin = 0.0, vmax = METRICS[metric]["max"])
	cluster_boxed	= draw_heatmap_cluster_boxes()
	heatmap.set_xlim(0, n_samples)
	heatmap.set_ylim(0, n_samples)

	# plot cluster bars and sample labels
	class_colors = get_class_colors_list()
	xlab_offset = -interspace_inch / heatmap_size_inch * n_samples
	ylab_offset = -(interspace_inch * 2 + clstbar_width_inch) / heatmap_size_inch\
		* n_samples
	for new_idx, old_idx in enumerate(sort_idx):
		cls_idx = clustering.hca.labels_[old_idx]
		# plot patch both y (left and right) axes
		for i in [left_clstbar, right_clstbar]:
			patch = matplotlib.patches.Rectangle(xy = (0, new_idx),
				width = 1, height = 1, fill = True, edgecolor = "none",
				facecolor = class_colors[cls_idx])
			i.add_patch(patch)
		# label text
		text		= sample_data["group"][old_idx] + "_"\
			+ sample_data["sample"][old_idx]
		text_color	= sample_data["color"][old_idx]
		# aside x-axis
		heatmap.text(new_idx + 0.5, xlab_offset, text, clip_on = False,
			color = text_color, rotation = 90, fontsize = 8,
			horizontalalignment = "center", verticalalignment = "top")
		# aside y-axis
		heatmap.text(ylab_offset, new_idx + 0.5, text, clip_on = False,
			color = text_color, fontsize = 8,
			horizontalalignment = "right", verticalalignment = "center")
	for i in [left_clstbar, right_clstbar]:
		i.set_xlim(0, 1)
		i.set_ylim(0, n_samples)

	# plot dendrogram
	dendrogram.grid(linestyle = "-", linewidth = 1.0, color = "#FFFFFF")
	dendro_data = clustering.dendrogram
	for xs, ys in zip(dendro_data["dcoord"], dendro_data["icoord"]):
		line = matplotlib.lines.Line2D(xs, ys,
			linestyle = "-", linewidth = 2.0, color = "#4040FF")
		dendrogram.add_line(line)
	# plot cutoff threshold
	dendrogram.axvline(clustering.hca.distance_threshold, linestyle = "-",
		linewidth = 2.0, color = "#FF0000")
	# make dendrogram aligned
	dmax = max(itertools.chain(*dendro_data["dcoord"]))
	hwratio = get_axes_hwratio(dendrogram)
	tmax = dmax * (hwratio / 2 / n_samples + 1)
	dendrogram.set_xlim(0, tmax)
	dendrogram.set_ylim(0, 10 * n_samples)

	# misc
	heatmap.set_title("Hierarchical Clustering and Dendrogram\n"
		"metric=%s; linkage=%s; cutoff=%.2f; clusters=%u" % (metric,
		clustering.hca.linkage, clustering.hca.distance_threshold,
		clustering.n_clusters), fontsize = 20)

	# save and clean up
	matplotlib.pyplot.savefig(png, dpi = quality_config["dpi"])
	matplotlib.pyplot.close()
	return


def get_cluster_abund(sample_data, clustering, group_config):
	# group stats
	group		= [i.copy() for i in group_config]
	n_groups	= len(group)
	group_to_id	= {v["name"]: i for i, v in enumerate(group_config)}
	# count labels in each group/cluster
	# note columns are groups
	clust_cnts	= numpy.zeros((clustering.n_clusters, n_groups), dtype = int)
	for i, g in enumerate(sample_data["group"]):
		clust_cnts[clustering.hca.labels_[i], group_to_id[g]] += 1
	# count cluster abundances in each group
	sum_cnts	= clust_cnts.sum(axis = 0, keepdims = True)
	clust_abund	= clust_cnts / sum_cnts
	#clust_abund[numpy.isnan(clust_abund)] = 0

	# return dict
	ret = dict(
		group	= group,
		counts	= clust_cnts,
		abund	= clust_abund,
	)
	return ret


def plot_abund_bars(png, *, sample_data, clustering, group_config):
	# number of groups (stack bars) shown
	group_mask			= [not i.get("invisible", False) for i in group_config]
	n_groups			= sum(group_mask)

	# layout
	# margin
	left_margin_inch	= 0.6
	right_margin_inch	= 2.5
	bottom_margin_inch	= 1.5
	top_margin_inch		= 0.5
	# axes
	axes_col_width_inch	= 0.5
	axes_width_inch		= axes_col_width_inch * n_groups
	axes_height_inch	= 4.0

	# create figure
	figure_width_inch	= left_margin_inch + axes_width_inch\
		+ right_margin_inch
	figure_height_inch	= bottom_margin_inch + axes_height_inch\
		+ top_margin_inch
	figure = matplotlib.pyplot.figure(
		figsize = (figure_width_inch, figure_height_inch))

	# create axes
	axes_left		= left_margin_inch / figure_width_inch
	axes_bottom		= bottom_margin_inch / figure_height_inch
	axes_width		= axes_width_inch / figure_width_inch
	axes_height		= axes_height_inch / figure_height_inch
	axes = figure.add_axes([axes_left, axes_bottom, axes_width,
		axes_height])
	# style axes
	for sp in axes.spines.values():
		sp.set_visible(False)
	axes.set_facecolor("#E0E0F0")
	axes.grid(axis = "y", linestyle = "-", linewidth = 1.0, color = "#FFFFFF")
	axes.tick_params(bottom = False)

	# analysis
	opu_abund		= get_cluster_abund(sample_data, clustering, group_config)
	# FUCK begins here
	_argsort_abund	= numpy.flip(opu_abund["abund"].mean(axis = 1).argsort())
	# now use opu_abund["abund"][_argsort_abund, :] should be in descending order

	# plot
	class_colors	= get_class_colors_list()
	cumu_abund		= numpy.zeros(n_groups, dtype = float)
	handles = list()
	#for abund, color in zip(opu_abund["abund"][_argsort_abund, :], class_colors):
	max_shown_opus = 15 # plot first #opus or so
	for i in range(min(len(_argsort_abund), max_shown_opus)):
		sorted_order = _argsort_abund[i]
		abund = opu_abund["abund"][sorted_order]
		color = class_colors[i]
		shown_abund = abund[group_mask] # mask invisible groups
		bar_plot = axes.bar(numpy.arange(n_groups), height = shown_abund, width = 0.7,
			bottom = cumu_abund, color = color, edgecolor = "none", zorder = 2)
		cumu_abund += shown_abund
		# handle for legend
		handles.append(matplotlib.patches.Rectangle((0, 0), 0, 0,
			edgecolor = "none", facecolor = color, label = "PAO_OPU_%u" % (i + 1)))
	# group the rest (if any) into 'other'
	if len(_argsort_abund) > max_shown_opus:
		abund = opu_abund["abund"][_argsort_abund[max_shown_opus:]].sum(axis = 0)
		color = "#f0f0f0"
		shown_abund = abund[group_mask] # mask invisible groups
		bar_plot = axes.bar(numpy.arange(n_groups), height = shown_abund, width = 0.7,
			bottom = cumu_abund, color = color, edgecolor = "none", zorder = 2)
		cumu_abund += shown_abund
		# handle for legend
		handles.append(matplotlib.patches.Rectangle((0, 0), 0, 0,
			edgecolor = "none", facecolor = color, label = "Other minor OPUs"))

	# legend
	axes.legend(handles = handles, loc = 2, bbox_to_anchor = (1.02, 1.02))

	# misc
	axes.set_xlim(-0.5, n_groups - 0.5)
	axes.set_ylim(0, 1.0)
	# x axis
	x_ticks			= numpy.arange(n_groups)
	x_ticklabels	= [g["name"] for g, m in zip(group_config, group_mask) if m]
	axes.set_xticks(x_ticks)
	axes.set_xticklabels(x_ticklabels, fontsize = 12, rotation = 90)
	# y axis
	y_ticks = numpy.linspace(0, 1.0, 6)
	axes.set_yticks(y_ticks)
	axes.set_title("Cluster Abundances", fontsize = 14)

	# save and clean up
	matplotlib.pyplot.savefig(png, dpi = 300)
	matplotlib.pyplot.close()
	return


def plot_opu_pca(png, *, sample_data, clustering, group_config,
		without_labels = False):
	# number of groups shown
	group_mask			= [not i.get("invisible", False) for i in group_config]

	# layout
	# margin
	left_margin_inch	= 1.2
	right_margin_inch	= 0.5
	bottom_margin_inch	= 0.9
	top_margin_inch		= 0.5
	# axes
	axes_width_inch		= 6.0
	axes_height_inch	= 6.0

	# create figure
	figure_width_inch	= left_margin_inch + axes_width_inch\
		+ right_margin_inch
	figure_height_inch	= bottom_margin_inch + axes_height_inch\
		+ top_margin_inch
	figure = matplotlib.pyplot.figure(
		figsize = (figure_width_inch, figure_height_inch))

	# create axes
	axes_left		= left_margin_inch / figure_width_inch
	axes_bottom		= bottom_margin_inch / figure_height_inch
	axes_width		= axes_width_inch / figure_width_inch
	axes_height		= axes_height_inch / figure_height_inch
	axes = figure.add_axes([axes_left, axes_bottom, axes_width,
		axes_height])
	# style axes
	for sp in axes.spines.values():
		sp.set_visible(False)
	axes.set_facecolor("#E0E0F0")
	axes.grid(linestyle = "-", linewidth = 1.0, color = "#FFFFFF")
	axes.tick_params(labelsize = 16)
	axes.axvline(0.0, linestyle = "--", linewidth = 0.5, color = "#000000",
		zorder = 2)
	axes.axhline(0.0, linestyle = "--", linewidth = 0.5, color = "#000000",
		zorder = 2)

	# analysis
	# opu stats
	opu_abund	= get_cluster_abund(sample_data, clustering, group_config)
	# pca
	pca			= sklearn.decomposition.PCA(n_components = 2)
	#pca			= sklearn.decomposition.PCA(n_components = 2, whiten = True)
	transformed	= pca.fit_transform(opu_abund["abund"].T)

	# plot groups
	axes.scatter(transformed[:, 0][group_mask], transformed[:, 1][group_mask],
		marker = "o", s = 80, linewidth = 2.0, edgecolor = "#4040FF",
		facecolor = "#FFFFFF40", zorder = 3)
	if not without_labels:
		for g, m, xy in zip(group_config, group_mask, transformed):
			if not m:
				# m 
				continue
			# labels
			axes.text(*xy, g["name"], fontsize = 16, fontweight = "bold",
				horizontalalignment = "left", verticalalignment = "bottom")
	# plot opus
	opu_id = 0
	class_colors = get_class_colors_list()
	for xy, c in zip(pca.components_.T, class_colors):
		opu_id += 1 # opu id should increase even if this opu is not drawn
		# do not draw arrow/text if the arrow length is too small
		if numpy.linalg.norm(xy) >= 0.05:
			# arrow
			arrow = matplotlib.patches.FancyArrowPatch((0, 0), xy, zorder = 3,
				arrowstyle = "->, head_length = 0.10, head_width = 0.05",
				linestyle = "-", edgecolor = c, linewidth = 3.0,
				mutation_scale = 100, clip_on = False)
			axes.add_patch(arrow)
			# text
			if not without_labels:
				axes.text(*xy, "OPU_%u" % opu_id, fontsize = 14, color = c,
					horizontalalignment = "left", verticalalignment = "bottom")
		else:
			print("opu_%u is insignificant" % opu_id, file = sys.stderr)

	# misc
	axes.set_xlim(-1.00, 1.00)
	axes.set_ylim(-1.00, 1.00)
	axes.set_xlabel("PC1 (%.1f%%)" % (pca.explained_variance_ratio_[0] * 100),
		fontsize = 22, fontweight = "bold")
	axes.set_ylabel("PC1 (%.1f%%)" % (pca.explained_variance_ratio_[1] * 100),
		fontsize = 22, fontweight = "bold")
	#axes.set_title("PCA/OPU abundances", fontsize = 14)

	# save and clean up
	matplotlib.pyplot.savefig(png, dpi = 300)
	matplotlib.pyplot.close()
	return


def print_sample_cluster_labels(fname, *, sample_data, clustering, group_config):
	# FUCK begins here
	opu_abund		= get_cluster_abund(sample_data, clustering, group_config)
	_argsort_abund	= numpy.flip(opu_abund["abund"].mean(axis = 1).argsort())
	inv_cls_map		= numpy.empty(len(_argsort_abund), dtype = int)
	inv_cls_map[_argsort_abund] = numpy.arange(len(_argsort_abund))
	with open(fname, "w") as fp:
		for i, cls in enumerate(clustering.hca.labels_):
			sample_name = sample_data["group"][i] + "/" \
				+ sample_data["sample"][i]
			fp.write("%s\t%u\n" % (sample_name, inv_cls_map[cls]))
	return


def main():
	args = get_args()
	# load group
	group = load_json(args.input)
	# load data
	for i in group:
		i["data"] = load_json(i["file"])
	# clustering
	sample_data	= make_sample_2d_data(group)
	dist_mat	= get_dist_matrix(sample_data["intensity"],
		metric = args.metric)
	cluster		= HierarchicalCluster.from_dist_matrix(dist_mat,
		linkage = "average", cutoff = args.cutoff_threshold)
	# print sample cluster labels
	if args.list_cluster_labels:
		print_sample_cluster_labels(
			"%s.cluster.%s.labels.txt" % (args.plot_prefix, args.metric),
			sample_data = sample_data, clustering = cluster,
			group_config = group)
	# plot
	if not args.no_plot_heatmap:
		plot_heatmap("%s.heatmap.%s.png" % (args.plot_prefix, args.metric),
			sample_data = sample_data, clustering = cluster,
			metric = args.metric, quality = args.heatmap_quality)
	if not args.no_plot_stack_bar:
		plot_abund_bars("%s.bar.%s.png" % (args.plot_prefix, args.metric),
			sample_data = sample_data, clustering = cluster,
			group_config = group)
	if not args.no_plot_pca:
		plot_opu_pca("%s.pca.%s.png" % (args.plot_prefix, args.metric),
			sample_data = sample_data, clustering = cluster,
			group_config = group, without_labels = args.without_pca_labels)
	return


if __name__ == "__main__":
	main()

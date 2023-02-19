#!/usr/bin/env python3

import collections
import matplotlib
import matplotlib.cm
import matplotlib.colors
import matplotlib.patches
import matplotlib.pyplot
import numpy
import scipy.cluster
import sys
# custom lib
import mpllayout
from . import util
from . import future # import sklearn.cluster.AgglomerativeClustering here
from . import registry
from .analysis_dataset_routine import AnalysisDatasetRoutine


class AnalysisHCARoutine(AnalysisDatasetRoutine):
	"""
	hierarchical clustering analysis routines, including run the clustering,
	save labels and plot heatmap/dendrogram figure
	"""
	metric_reg = registry.get("cluster_metric")
	cutoff_opt_reg = registry.get("hca_cutoff_optimizer")

	@property
	def cutoff(self):
		return self.cutoff_opt.cutoff_final
	@property
	def hca_labels(self) -> int:
		return self.hca.labels_
	@property
	def n_clusters(self) -> int:
		return self.hca_labels.max() + 1
	@property
	def remapped_hca_label(self) -> list:
		"""
		remapped_hca_label will only report clusters which have spectra
		count more than <opu_min_size> used when calling run_hca() in addition,
		the remapped label is also sorted in descending order by count, meaning
		that the lower the label value, larger the cluster
		"""
		return [self._label_remap.get(i, None) for i in self.hca_labels]
	@property
	def remapped_hca_label_unique(self) -> list:
		# this sort lambda ensures that None will not raise an error and
		# is always at the end of this array
		# None represents the opu "label" of small clusters
		return sorted(set(self.remapped_hca_label),
			key = lambda x: sys.maxsize if x is None else x)


	def run_hca(self, *, metric=metric_reg.default_key, cutoff=0.7,
			linkage="average", opu_min_size=None):
		# create the hca object
		self.metric = self.metric_reg.get(metric)
		self.cutoff_opt = self.cutoff_opt_reg.get(cutoff)
		self.cutoff_pend = cutoff
		self.linkage = linkage
		self.opu_min_size = opu_min_size
		self.hca = future.sklearn_cluster_AgglomerativeClustering(
			linkage=self.linkage, metric="precomputed",
			# metric="precomputed" as we manually compute the distance matrix
			# note that old scikit-learn library (pre 23.0.0) uses 'affinity'
			# using 'metric' keyword here will raise an error
			distance_threshold=0, n_clusters=None
			# distance_threshold=0 is a placeholer, it will be replaced by
			# cutoff_opt.cutoff_final when optimization is finished
		)
		# calculate distance matrix
		self.dist_mat = self.metric(self.dataset.intens)
		# find the cutoff
		self.__optimize_cutoff(n_step = 100)
		# calculate clusters, using sklearn's backend
		self.hca.set_params(distance_threshold=self.cutoff)
		self.hca.fit(self.dist_mat)
		# calculate linkage matrix
		self.linkage_matrix = self.__calc_linkage_matrix(self.hca)
		# make dendrogram using scipy's backend
		self.dendrogram = scipy.cluster.hierarchy.dendrogram(
			self.linkage_matrix, orientation="right",
			no_plot=True
		)
		# sort opu labels
		self.__sort_and_filter_cluster_labels(opu_min_size)
		return self


	def save_hca_labels(self, f, delimiter="\t"):
		if not f:
			return
		with util.get_fp(f, "w") as fp:
			for i in zip(self.dataset.spectra_names, self.remapped_hca_label):
				name, label = i
				# if a label is None (when cluster size below opu_min_size),
				# write the label as "-" instead
				print((delimiter).join([name, str(label or "-")]), file = fp)
		return


	def plot_hca(self, png, *, dpi = 300):
		if png is None:
			return
		# create figure layout
		layout = self.__create_layout()
		figure = layout["figure"]

		# plot heatmap
		ax = layout["heatmap"]
		self.__plot_heatmap(ax, layout["colorbar"])

		# plot dendrogram
		ax = layout["dendro"]
		self.__plot_dendrogram(ax, i2d_ratio = layout["dendro_i2d"])
		ax.axvline(self.cutoff, linestyle = "-", linewidth = 1.0,
			color = "#ff0000", zorder = 4)

		# plot pbar
		self.__plot_hca_cluster_bar(layout["pbar_l"])
		self.__plot_hca_cluster_bar(layout["pbar_r"])

		# plot group bar
		self.__plot_hca_biosample_bar(layout["biosample_bar"])

		## plot group legend
		#plot_group_legend(layout["dendro"], group_data = group_data)

		# misc
		figure.suptitle("OPU clustering (hierarchical)\n"
			"metric=%s; linkage=%s; cutoff=%s; raw clusters=%u; "
			"OPU min. size=%u"\
			% (self.metric.name_str, self.linkage,
				self.cutoff_opt.cutoff_final_str, self.n_clusters,
				self.opu_min_size,
			), fontsize = 16
		)

		# save fig and clean up
		figure.savefig(png, dpi = dpi)
		matplotlib.pyplot.close()
		return


	@property
	def clusters_colors(self) -> util.CyclicIndexedList:
		# get preliminary colors by internal colormaps
		prelim = matplotlib.cm.get_cmap("Set3").colors\
			+ matplotlib.cm.get_cmap("Set2").colors
			#+ matplotlib.cm.get_cmap("Accent").colors[:-1]
			#+ matplotlib.cm.get_cmap("Set3").colors\
			#+ matplotlib.cm.get_cmap("Set2").colors\
		# translate to color hex colors and remove identical colors
		colors = util.drop_replicate(map(matplotlib.colors.to_hex, prelim))
		return util.CyclicIndexedList(colors)


	@staticmethod
	def __calc_linkage_matrix(hca):
		# this function is adapted from:
		# 'https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html' as of version 1.1.1
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


	def __optimize_cutoff(self, n_step=100) -> float:
		# in usual cases, this should only be called by self.run_hca()
		cutoff_list = numpy.linspace(
			self.dist_mat.min(),
			self.dist_mat.max(),
			n_step
		)
		self.cutoff_opt.optimize(
			model=self.hca,
			data=self.dataset.intens,
			dist=self.dist_mat,
			cutoff_list=cutoff_list,
			cutoff_pend=self.cutoff_pend
		)
		return self.cutoff_opt.cutoff_final


	def __sort_and_filter_cluster_labels(self, opu_min_size=None):
		if opu_min_size is None: opu_min_size = 0
		sorted_labels = collections.Counter(self.hca_labels).most_common()
		label_remap = dict()
		for i, v in enumerate(sorted_labels):
			label, count = v
			if count >= opu_min_size:
				label_remap[label] = i
		self._label_remap = label_remap
		return


	def __create_layout(self, legend_space=1.0):
		lc = mpllayout.LayoutCreator(
			left_margin		= 0.2,
			right_margin	= legend_space + 0.2,
			top_margin		= 0.7,
			bottom_margin	= 0.5,
		)

		pbar_width			= 0.6
		biosample_bar_width	= 0.2
		noise_bar_width		= 0.2
		cbar_height			= 0.4
		heatmap_size		= 8.0
		dendro_width		= 2.5
		axes_gap			= 0.1

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

		biosample_bar = lc.add_frame("biosample_bar")
		biosample_bar.set_anchor("bottomleft", ref_frame = pbar_r,
			ref_anchor = "bottomright", offsets = (axes_gap / 2, 0))
		biosample_bar.set_size(biosample_bar_width, heatmap_size)

		dendro = lc.add_frame("dendro")
		dendro.set_anchor("bottomleft", 
			ref_frame = biosample_bar,
			ref_anchor = "bottomright", offsets = (axes_gap, 0))
		dendro.set_size(dendro_width, heatmap_size)

		# create layout
		layout = lc.create_figure_layout()
		layout["dendro_i2d"] = dendro.get_width() / dendro.get_height()

		# apply axes style
		for n in ["colorbar", "biosample_bar", "dendro"]:
			axes = layout[n]
			for sp in axes.spines.values():
				sp.set_visible(False)
			axes.set_facecolor("#f0f0f8")

		for n in ["pbar_l", "pbar_r"]:
			axes = layout[n]
			for sp in axes.spines.values():
				sp.set_edgecolor("#c0c0c0")
			axes.set_facecolor("#f0f0f8")

		for n in ["pbar_l", "heatmap", "colorbar", "pbar_r", "biosample_bar"]:
			axes = layout[n]
			axes.tick_params(
				left=False, labelleft=False,
				right=False, labelright=False,
				bottom=False, labelbottom=False,
				top=False, labeltop=False
			)
		layout["dendro"].tick_params(
			left=False, labelleft=False,
			right=False, labelright=False,
			bottom=True, labelbottom=True,
			top=False, labeltop=False
		)

		return layout


	def __plot_heatmap(self, heatmap_axes, colorbar_axes) -> dict:
		# heatmap
		ax = heatmap_axes
		heatmap_data = self.metric.to_plot_data(self.dist_mat)
		pcolor = ax.pcolor(heatmap_data[numpy.ix_(self.dendrogram["leaves"],
			self.dendrogram["leaves"])], cmap = self.metric.cmap,
			vmin = self.metric.vmin, vmax = self.metric.vmax)
		#
		ax.set_xlim(0, self.dataset.n_spectra)
		ax.set_ylim(0, self.dataset.n_spectra)

		# colorbar
		ax = colorbar_axes
		cbar = colorbar_axes.figure.colorbar(pcolor, cax = ax,
			ticklocation = "bottom", orientation = "horizontal")
		# misc
		cbar.outline.set_visible(False)
		cbar.set_label(self.metric.name_str, fontsize = 14)

		ret = dict(heatmap = pcolor, colorbar = cbar)
		return ret


	def __dendro_get_adjusted_dmax(self, dmax, i2d_ratio) -> float:
		return dmax / (1 - 1 / (2 * self.dataset.n_spectra * i2d_ratio))


	def __plot_dendrogram(self, ax, *, i2d_ratio: float) -> list:
		lines = list() # the return list containing all lines drawn
		for xys in zip(self.dendrogram["dcoord"], self.dendrogram["icoord"]):
			line = matplotlib.lines.Line2D(*xys, linestyle = "-",
				linewidth = 1.0, color = "#4040ff", zorder = 3)
			ax.add_line(line)
			lines.append(line)
		# misc
		ax.grid(axis = "x", linestyle = "-", linewidth = 1.0, color = "#ffffff",
			zorder = 2)
		ax.set_xlim(0, self.__dendro_get_adjusted_dmax(
			dmax = numpy.max(self.dendrogram["dcoord"]),
			i2d_ratio = i2d_ratio)
		)
		ax.set_ylim(0, 10 * self.dataset.n_spectra)

		return lines


	def __plot_hca_cluster_bar(self, ax):
		remapped_hca_label = self.remapped_hca_label
		color_list = self.clusters_colors
		for i, leaf in enumerate(self.dendrogram["leaves"]):
			label = remapped_hca_label[leaf]
			facecolor = "#ffffff" if label is None else color_list[label]
			patch = matplotlib.patches.Rectangle((0, i), 1, 1,
				edgecolor="none", facecolor=facecolor
			)
			ax.add_patch(patch)
		# add label
		ax.text(0.5, 0.0, "OPU ", fontsize=12, rotation=90,
			horizontalalignment="center", verticalalignment="top"
		)

		ax.set_xlim(0, 1)
		ax.set_ylim(0, self.dataset.n_spectra)
		return


	def __plot_hca_biosample_bar(self, ax):
		for i, leaf in enumerate(self.dendrogram["leaves"]):
			patch = matplotlib.patches.Rectangle((0, i), 1, 1,
				edgecolor = "none", facecolor = self.biosample_color[leaf]
			)
			ax.add_patch(patch)
		# add label
		ax.text(0.5, 0.0, "biosample ", fontsize = 12, rotation = 90,
			horizontalalignment = "center", verticalalignment = "top"
		)

		ax.set_xlim(0, 1)
		ax.set_ylim(0, self.dataset.n_spectra)
		return

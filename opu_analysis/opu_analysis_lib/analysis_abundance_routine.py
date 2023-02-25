#!/usr/bin/env python3

import collections

import matplotlib
import matplotlib.pyplot
import numpy

# custom lib
import mpllayout

from . import registry, util
from .analysis_hca_routine import AnalysisHCARoutine


class AnalysisAbundanceRoutine(AnalysisHCARoutine):
	"""
	routines to calculate OPU abundances in each biosample; most methods of this
	class should be run after AnalysisHCARoutine.run_hca() has been called, and
	the results become available
	"""
	biplot_meth_reg = registry.get("dim_red_visualize")

	@util.with_check_data_avail(check_data_attr="hca", dep_method="run_hca")
	def plot_opu_abundance_stackbar(self, *, plot_to="show", dpi=300):
		if plot_to is None:
			return
		# calculating biosample opu stats
		biosample_stats = self.__get_biosample_opu_stats()
		n_biosample = len(biosample_stats)

		# create layout
		layout = self.__stackbar_create_layout()
		figure = layout["figure"]
		figure.set_dpi(dpi)

		# plot stackbars
		color_list = self.clusters_colors
		handles = list()
		bottom = numpy.zeros(n_biosample, dtype=float)
		x = numpy.arange(n_biosample) + 0.5  # center of each bar
		# plot major opus
		ax = layout["axes"]
		for l in self.remapped_hca_label_unique:
			h = [i["opu_abunds"].get(l, 0) for i in biosample_stats]
			edgecolor = "#404040" if l is None else "none"
			facecolor = "#ffffff" if l is None else color_list[l]
			label = "other minor" if l is None else "OPU_%02u" % l
			bar = ax.bar(x, h, width=0.8, bottom=bottom, align="center",
				edgecolor=edgecolor, linewidth=0.5, facecolor=facecolor,
				label=label
			)
			bottom += h
			handles.append(bar)

		# legend
		ax.legend(handles=handles, loc=2, bbox_to_anchor=(1.02, 1.02),
			fontsize=10, handlelength=0.8, frameon=False,
		)

		# misc
		ax.set_xlim(0, n_biosample)
		ax.set_ylim(0.0, 1.0)
		ax.set_ylabel("OPU abundance", fontsize=12)
		ax.set_xticks(x)
		ax.set_xticklabels([i["name"] for i in biosample_stats], fontsize=10,
			rotation=90
		)

		# save fig and clean up
		if plot_to == "show":
			matplotlib.pyplot.show()
			ret = None
		elif plot_to == "jupyter":
			ret = figure
		else:
			figure.savefig(plot_to)
			matplotlib.pyplot.close()
			ret = None
		return ret

	@util.with_check_data_avail(check_data_attr="hca", dep_method="run_hca")
	def plot_opu_abundance_biplot(self, *, method=biplot_meth_reg.default_key,
			plot_to="show", dpi=300):
		if plot_to is None:
			return

		# calculating biosample opu stats
		biosample_stats = self.__get_biosample_opu_stats()
		n_biosample = len(biosample_stats)
		hca_labels = [i for i in self.remapped_hca_label_unique
			if i is not None]
		n_label = len(hca_labels)

		# abund_mat, n_biosample * n_label
		abund_mat = numpy.empty((n_biosample, n_label), dtype=float)
		for i in range(n_biosample):
			for j, l in enumerate(hca_labels):
				abund_mat[i, j] = biosample_stats[i]["opu_abunds"].get(l, 0)

		# run dimensionality reduction
		biplot_meth = self.biplot_meth_reg.get(method)
		biplot_meth(abund_mat)

		# create layout
		layout = self.__biplot_create_layout()
		figure = layout["figure"]
		figure.set_dpi(dpi)

		# plot biplot, sample
		ax = layout["axes"]
		sample_xy = biplot_meth.sample_points_for_plot
		ax.scatter(*sample_xy, marker="o", s=40, edgecolor="#4040ff",
			linewidth=1.0, facecolor="#ffffff40", zorder=1)
		for xy, s in zip(sample_xy.T, self.biosample_unique):
			ax.text(*xy, s, fontsize=6, zorder=3,
				rotation=self.__perp_text_rotation(*xy), rotation_mode="anchor",
				horizontalalignment="center",
				verticalalignment="bottom" if xy[1] >= 0 else "top")

		# plot biplot, feature
		feature_xy = biplot_meth.feature_points_for_plot
		for xy, l in zip(feature_xy, hca_labels):
			# use annotate() to draw arrows acan
			ax.annotate("", xy, (0, 0),
				arrowprops=dict(
					arrowstyle="-|>",
					linewidth=1.5,
					edgecolor="#ffa040",
					facecolor="#ffa040",
				),
				zorder=2,
			)
			ax.text(*xy, "OPU_%02u" % l, fontsize=6, color="#ff4040", zorder=3,
				rotation=self.__perp_text_rotation(*xy), rotation_mode="anchor",
				horizontalalignment="center",
				verticalalignment="bottom" if xy[1] >= 0 else "top")

		# misc
		ax.axvline(0, linestyle="--", linewidth=1.0, color="#808080", zorder=1)
		ax.axhline(0, linestyle="--", linewidth=1.0, color="#808080", zorder=1)
		coord_max = biplot_meth.sample_points_for_plot.max() * 1.1
		ax.set_xlim(-coord_max, coord_max)
		ax.set_ylim(-coord_max, coord_max)
		ax.set_xlabel(biplot_meth.xlabel_str, fontsize=14)
		ax.set_ylabel(biplot_meth.ylabel_str, fontsize=14)

		# save fig and clean up
		if plot_to == "show":
			matplotlib.pyplot.show()
			ret = None
		elif plot_to == "jupyter":
			ret = figure
		else:
			figure.savefig(plot_to)
			matplotlib.pyplot.close()
			ret = None
		return ret

	def __get_biosample_opu_stats(self) -> list:
		"""
		biosample stats based on self.biosample, self.biosample_color, and
		self.remapped_hca_label;
		returns a list of dict elements each represents stats of a biosample, in
		the same order follows the encounting order in self.biosample

		stats include:
		name: biosample name
		n_spectra: number of spectra in that biosample
		color: color of that biosample
		opu_counts: spectra counts in each opu, as in remapped opu labels
		"""
		stats = collections.defaultdict(
			lambda: dict(n_spectra=0, color=None,
				opu_counts=collections.Counter())
		)
		# go over the each biosample and biosample_color
		for s, c, l in zip(self.biosample, self.biosample_color,
				self.remapped_hca_label):
			st = stats[s]
			st["name"] = s
			st["n_spectra"] += 1
			st["color"] = c
			st["opu_counts"][l] += 1
		# calculate opu abundnaces
		for st in stats.values():
			st["opu_abunds"] = {
				l: c / st["n_spectra"] for l, c in st["opu_counts"].items()
			}
		# return a list with the order we want
		self.biosample_opu_stats = [stats[i] for i in self.biosample_unique]
		return self.biosample_opu_stats

	def __stackbar_create_layout(self):
		lc = mpllayout.LayoutCreator(
			left_margin=0.7,
			right_margin=1.5,
			top_margin=0.5,
			bottom_margin=2.0,
		)

		axes = lc.add_frame("axes")
		axes.set_anchor("bottomleft")
		axes.set_size(0.2 * len(self.biosample_opu_stats), 3.0)

		# create layout
		layout = lc.create_figure_layout()

		# apply axes style
		axes = layout["axes"]
		for sp in axes.spines.values():
			sp.set_visible(False)
		axes.set_facecolor("#f0f0f8")
		axes.tick_params(
			left=True, labelleft=True,
			right=False, labelright=False,
			bottom=True, labelbottom=True,
			top=False, labeltop=False
		)

		return layout

	def __biplot_create_layout(self):
		lc = mpllayout.LayoutCreator(
			left_margin=1.0,
			right_margin=0.2,
			top_margin=0.2,
			bottom_margin=0.7,
		)

		axes = lc.add_frame("axes")
		axes.set_anchor("bottomleft")
		axes.set_size(4.0, 4.0)

		# create layout
		layout = lc.create_figure_layout()

		# apply axes style
		axes = layout["axes"]
		for sp in axes.spines.values():
			sp.set_visible(False)
		axes.set_facecolor("#f0f0f8")
		axes.tick_params(
			left=True, labelleft=True,
			right=False, labelright=False,
			bottom=True, labelbottom=True,
			top=False, labeltop=False
		)

		return layout

	@staticmethod
	def __perp_text_rotation(x, y) -> float:
		t_rot_tan = -x / y
		return numpy.math.degrees(numpy.math.atan(t_rot_tan))

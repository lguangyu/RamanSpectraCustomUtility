#!/usr/bin/env python3

import collections

import matplotlib
import matplotlib.pyplot
import numpy

# custom lib
import mpllayout

from . import util
from .analysis_hca_routine import AnalysisHCARoutine


class AnalysisAbundanceRoutine(AnalysisHCARoutine):
	"""
	routines to calculate OPU abundances in each biosample; most methods of this
	class should be run after AnalysisHCARoutine.run_hca() has been called, and
	the results become available
	"""

	def get_biosample_opu_stats(self) -> list:
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
		# return a list with the order we want, and preserve the encounting
		# order of each unique value
		unique_biosample = util.drop_replicate(self.biosample)
		self.biosample_opu_stats = [stats[i] for i in unique_biosample]
		return self.biosample_opu_stats

	def plot_opu_abundance_stackbar(self, png, *, dpi=300):
		if not png:
			return
		# calculating biosample opt stats
		biosample_stats = self.get_biosample_opu_stats()
		n_biosample = len(biosample_stats)

		# create layout
		layout = self.__create_layout()
		figure = layout["figure"]

		# plot stackbars
		color_list = self.clusters_colors
		handles = list()
		bottom = numpy.zeros(n_biosample, dtype=float)
		x = numpy.arange(n_biosample) + 0.5  # center of each bar
		# plot major opus
		ax = layout["axes"]
		for l in self.remapped_hca_label_unique:
			h = [i["opu_counts"][l] / i["n_spectra"] for i in biosample_stats]
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
		figure.savefig(png, dpi=dpi)
		matplotlib.pyplot.close()
		return

	def __create_layout(self):
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

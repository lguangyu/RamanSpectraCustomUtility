#!/usr/bin/env python3

import matplotlib
import matplotlib.pyplot
import numpy

# custom lib
import mpllayout

from . import registry, util
from .analysis_hca_routine import AnalysisHCARoutine


class AnalysisFeatureScoreRoutine(AnalysisHCARoutine):
	"""
	routines to calculate feature scores to differentiate OPUs; most methods of
	this class should be run after AnalysisHCARoutine.run_hca() has been called,
	and the results become available
	"""
	score_meth = registry.get("feature_score")

	@util.with_check_data_avail(check_data_attr="hca", dep_method="run_hca")
	def rank_features(self, method=score_meth.default_key) -> dict:
		"""
		rank features with label info acquired from HCA clustering

		return a dict of index arrays, each array represents the sorted indices
		of features distinguishing each opu from the rest of opus in one-vs-rest
		(OVR) fashion;
		feature indicies are ranked in descending order; i.e. the first element
		is the index of the most important feature, etc.

		results are saved in self.feature_rank_index
		"""
		score_meth = self.score_meth.get(method)
		self.feature_score_meth = score_meth  # save into object for later use
		# calculate in ovr
		ret = dict()
		for label in self.remapped_hca_label_unique:
			if label is None:  # do not calculate None
				continue
			y = numpy.equal(self.remapped_hca_label, label).astype(int)
			ret[label] = score_meth(self.dataset.intens, y)
		self.feature_rank_index = ret
		return ret

	@util.with_check_data_avail(check_data_attr="feature_rank_index",
		dep_method="rank_features")
	def save_opu_feature_rank_table(self, f, *, delimiter="\t"):
		if not f:
			return
		with util.get_fp(f, "w") as fp:
			for l in sorted(self.feature_rank_index.keys()):
				rank_index = [str(self.dataset.wavenum[i])
					for i in self.feature_rank_index[l]]
				print(delimiter.join(["OPU_%02u" % l] + rank_index), file=fp)
		return

	@util.with_check_data_avail(check_data_attr="feature_rank_index",
		dep_method="rank_features")
	def plot_opu_feature_rank(self, *, plot_to="show", dpi=300):
		if plot_to is None:
			return

		# prepare the rank matrix
		n_wavenum = self.dataset.n_wavenum  # n_wavenum is number of features
		plot_opu_labels = sorted(self.feature_rank_index.keys())
		n_opus = len(plot_opu_labels)
		rank_mat = numpy.empty((n_opus, n_wavenum), dtype=int)
		for i, v in enumerate(plot_opu_labels):
			rank_mat[i, self.feature_rank_index[v]] = numpy.arange(n_wavenum)

		# create layout
		layout = self.__create_layout(n_row=n_opus)
		figure = layout["figure"]
		figure.set_dpi(dpi)

		# plot heatmap
		wavenum_low = self.dataset.wavenum_low
		wavenum_high = self.dataset.wavenum_high
		ax = layout["axes"]
		X = numpy.linspace(wavenum_low, wavenum_high, n_wavenum + 1)
		Y = numpy.arange(0, n_opus + 1)
		ax.pcolor(X.reshape(1, -1), Y.reshape(-1, 1), rank_mat,
			cmap="viridis_r", zorder=2
		)
		# plot mask
		for i in range(n_opus):
			mask_y = (1.0 - (rank_mat[i] / rank_mat[i].max())) * 0.9
			interp_mask_y = numpy.interp(X, self.dataset.wavenum, mask_y)
			ax.fill_between(X, interp_mask_y + i, 1 + i,
				edgecolor="none", facecolor="#f0f0f8", zorder=3
			)

		# add opu label text on the right
		text_x = wavenum_low + (wavenum_high - wavenum_low) * 1.02
		for i, label in enumerate(plot_opu_labels):
			ax.text(text_x, i + 0.5, "OPU_%02u" % label, clip_on=False,
				color="#000000", fontsize=10,
				horizontalalignment="left", verticalalignment="center"
			)

		# misc
		ax.set_xlim(wavenum_low, wavenum_high)
		ax.set_ylim(0, n_opus)
		title = "Feature rank by %s" % self.feature_score_meth.name_str
		ax.set_title(title, fontsize=12)

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

	def __create_layout(self, n_row) -> dict:
		lc = mpllayout.LayoutCreator(
			left_margin=0.2,
			right_margin=1.2,
			top_margin=0.5,
			bottom_margin=0.4,
		)

		axes = lc.add_frame("axes")
		axes.set_anchor("bottomleft")
		axes.set_size(5.0, 0.25 * n_row)

		# create layout
		layout = lc.create_figure_layout()

		# apply axes style
		axes = layout["axes"]
		for sp in axes.spines.values():
			sp.set_visible(False)
		axes.set_facecolor("#f0f0f8")
		axes.tick_params(
			left=False, labelleft=False,
			right=False, labelright=False,
			bottom=True, labelbottom=True,
			top=False, labeltop=False
		)

		return layout

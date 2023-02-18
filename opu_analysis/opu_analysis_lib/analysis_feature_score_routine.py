#!/usr/bin/env python3

import matplotlib
import matplotlib.pyplot
import numpy
# custom lib
import mpllayout
from . import registry
from .analysis_hca_routine import AnalysisHCARoutine


class AnalysisFeatureScoreRoutine(AnalysisHCARoutine):
	"""
	routines to calculate feature scores to differentiate OPUs; most methods of
	this class should be run after AnalysisHCARoutine.run_hca() has been called,
	and the results become available
	"""
	score_meth = registry.get("feature_score")


	def rank_features(self, method=score_meth.default_key) -> numpy.ndarray:
		"""
		rank features with label info acquired from HCA clustering

		return a index arrays, each elements represents the index of a feature,
		ranked in descending order; i.e. the first element is the index of the
		most important feature, etc.
		"""
		score_meth = self.score_meth.get(method)
		self.feature_score_meth = score_meth

		ret = score_meth.feature_score(self.dataset.intens, self.hca_labels)
		self.feature_rank = ret
		return ret


	def plot_opu_feature_score(self, png, *, dpi=300):
		if not png:
			return
		
		# create layout
		layout = self.__create_layout()
		figure = layout["figure"]

		# prepare the rank matrix
		n_wavenum = self.dataset.n_wavenum # n_wavenum is number of features
		feat_rank = numpy.empty(n_wavenum, dtype=int)
		feat_rank[self.feature_rank] = numpy.arange(n_wavenum)

		# plot heatmap
		wavenum_low = self.dataset.wavenum_low
		wavenum_high = self.dataset.wavenum_high
		ax = layout["axes"]
		X = numpy.linspace(wavenum_low, wavenum_high, n_wavenum + 1)
		Y = numpy.arange(2)
		ax.pcolor(X.reshape(1, -1), Y.reshape(-1, 1),
			feat_rank.reshape(-1, n_wavenum), cmap="viridis_r", zorder=2
		)
		# plot mask
		mask_y = 1.0 - (feat_rank / feat_rank.max())
		interp_mask_y = numpy.interp(X, self.dataset.wavenum, mask_y)

		ax.fill_between(X, interp_mask_y, 1.0,
			edgecolor="none", facecolor="#f0f0f8", zorder=3
		)

		# misc
		ax.set_xlim(wavenum_low, wavenum_high)
		ax.set_ylim(0, 1)
		title = "Feature rank by %s" % self.feature_score_meth.name_str
		ax.set_title(title, fontsize = 12)

		# save and clean up
		figure.savefig(png, dpi = dpi)
		matplotlib.pyplot.close()
		return


	def __create_layout(self) -> dict:
		lc = mpllayout.LayoutCreator(
			left_margin		= 0.2,
			right_margin	= 0.2,
			top_margin		= 0.5,
			bottom_margin	= 0.4,
		)

		axes = lc.add_frame("axes")
		axes.set_anchor("bottomleft")
		axes.set_size(5.0, 1.0)

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
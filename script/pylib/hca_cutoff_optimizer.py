#!/usr/bin/env python3

import abc
import functools
import numpy
from . import registry


class HCACutoffOptimizer(object):
	@abc.abstractmethod
	def __call__(self, model, dist_mat, d, cutoff_list, *ka, **kw) -> float:
		pass


registry.new(registry_name = "hca_cutoff_optimizer",
	value_type = HCACutoffOptimizer)


@registry.get("hca_cutoff_optimizer").register("aic")
class AIC(HCACutoffOptimizer):
	def __call__(self, model, data, dist, cutoff_list, *ka, **kw) -> float:
		sigma = numpy.median(data.std(axis = 0))
		aic_list = [self._calc_aic(model, data, dist, i, sigma)\
			for i in cutoff_list]
		# find the cutoff with least aic
		ret = cutoff_list[numpy.argmin(aic_list)]
		return ret

	def _calc_aic(self, model, data, dist, cutoff, sigma):
		# adjust parameter and fit model
		model.set_params(distance_threshold = cutoff)
		model.fit(dist)
		# calculate aic
		d = data.shape[1]
		uniq_labels = numpy.unique(model.labels_)
		ret = 0
		for l in uniq_labels:
			cluster_points = data[model.labels_ == l]
			cluster_points -= cluster_points.mean(axis = 0, keepdims = True)
			ret += ((cluster_points / sigma) ** 2).sum()
		ret += 2 * d * model.n_clusters_
		return ret


@registry.get("hca_cutoff_optimizer").register("bic")
class BIC(HCACutoffOptimizer):
	def __call__(self, model, data, dist, cutoff_list, *ka, **kw) -> float:
		sigma = numpy.median(data.std(axis = 0))
		bic_list = [self._calc_bic(model, data, dist, i, sigma)\
			for i in cutoff_list]
		# find the cutoff with least bic
		ret = cutoff_list[numpy.argmin(bic_list)]
		return ret

	def _calc_bic(self, model, data, dist, cutoff, sigma):
		# adjust parameter and fit model
		model.set_params(distance_threshold = cutoff)
		model.fit(dist)
		# calculate bic
		n, d = data.shape
		uniq_labels = numpy.unique(model.labels_)
		ret = 0
		for l in uniq_labels:
			cluster_points = data[model.labels_ == l]
			cluster_points -= cluster_points.mean(axis = 0, keepdims = True)
			ret += ((cluster_points / sigma) ** 2).sum()
		ret += n * model.n_clusters_ * numpy.log(d)
		return ret

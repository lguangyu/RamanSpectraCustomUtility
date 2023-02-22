#!/usr/bin/env python3

import abc
import functools

import numpy
import sklearn.metrics

# custom lib
from . import registry


class ClusterMetric(abc.ABC):
	@abc.abstractmethod
	def __call__(self, X, Y=None, *ka, **kw) -> numpy.ndarray:
		pass

	@property
	@abc.abstractmethod
	def name_str(self) -> str:
		pass

	def to_plot_data(self, dist: numpy.ndarray) -> numpy.ndarray:
		return dist

	@property
	def vmax(self):
		return None

	@property
	def vmin(self) -> float:
		return 0

	@property
	def cmap(self):
		return "BuPu"


registry.new(registry_name="cluster_metric", value_type=ClusterMetric)


@registry.get("cluster_metric").register("euclidean")
class EuclideanDist(ClusterMetric):
	@functools.wraps(sklearn.metrics.pairwise.euclidean_distances)
	def __call__(self, *ka, **kw):
		return sklearn.metrics.pairwise.euclidean_distances(*ka, **kw)

	@property
	def name_str(self):
		return "Euclidean distance"


@registry.get("cluster_metric").register("cosine", as_default=True)
class CosineDist(ClusterMetric):
	@functools.wraps(sklearn.metrics.pairwise.cosine_distances)
	def __call__(self, *ka, **kw):
		return sklearn.metrics.pairwise.cosine_distances(*ka, **kw)

	@property
	def name_str(self):
		return "cosine similarity"

	def to_plot_data(self, dist):
		return 1 - dist

	@property
	def vmin(self):
		return -1

	@property
	def vmax(self):
		return 1

	@property
	def cmap(self):
		return "RdYlBu_r"


@registry.get("cluster_metric").register("sqrt_cosine")
class SqrtCosineDist(ClusterMetric):
	@functools.wraps(sklearn.metrics.pairwise.cosine_distances)
	def __call__(self, *ka, **kw):
		return numpy.sqrt(sklearn.metrics.pairwise.cosine_distances(*ka, **kw))

	@property
	def name_str(self):
		return "Sqrt. cosine distance"

	@property
	def vmax(self):
		return 1.4145

	@property
	def cmap(self):
		return "RdYlBu"

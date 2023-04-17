#!/usr/bin/env python3

import abc

from skfeature.function.similarity_based.fisher_score import fisher_score
from skfeature.function.similarity_based.lap_score import lap_score
from skfeature.function.similarity_based.trace_ratio import trace_ratio

# custom lib
from . import registry


class FeatureScoreMethod(abc.ABC):
	@abc.abstractmethod
	def feature_score(self, X, Y):
		pass

	@property
	@abc.abstractmethod
	def name_str(self) -> str:
		pass

	def __call__(self, *ka, **kw):
		return self.feature_score(*ka, **kw)


registry.new(registry_name="feature_score", value_type=FeatureScoreMethod)


@(registry.get("feature_score")).register("fisher_score")
class FisherScore(FeatureScoreMethod):
	def feature_score(self, X, Y):
		return fisher_score(X, Y, mode="index")

	@property
	def name_str(self):
		return "Fisher score"


@(registry.get("feature_score")).register("lap_score", as_default=True)
class LaplacianScore(FeatureScoreMethod):
	def feature_score(self, X, Y):
		return lap_score(X, Y, mode="index")

	@property
	def name_str(self):
		return "Laplacian score"


@(registry.get("feature_score")).register("trace_ratio")
class TraceRatio(FeatureScoreMethod):
	def feature_score(self, X, Y):
		return trace_ratio(X, Y, mode="index")

	@property
	def name_str(self):
		return "trace-ratio"

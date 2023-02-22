#!/usr/bin/env python3

import abc

import numpy

# custom lib
from . import registry


class NormMethod(object):
	@abc.abstractmethod
	def __call__(self, X, *ka, **kw) -> numpy.ndarray:
		pass

	@property
	@abc.abstractmethod
	def name_str(self) -> str:
		pass


registry.new(registry_name="normalize", value_type=NormMethod)


@registry.get("normalize").register("none", as_default=True)
class NormMeth_None(NormMethod):
	def __call__(self, X):
		return X

	@property
	def name_str(self):
		return "none"


@registry.get("normalize").register("l1")
class NormMeth_L1(NormMethod):
	def __call__(self, X):
		norm = numpy.linalg.norm(X, ord=1, axis=1, keepdims=True)
		return X / norm

	@property
	def name_str(self):
		return "l1"


@registry.get("normalize").register("l2")
class NormMeth_L2(NormMethod):
	def __call__(self, X):
		norm = numpy.linalg.norm(X, ord=2, axis=1, keepdims=True)
		return X / norm

	@property
	def name_str(self):
		return "l2"


@registry.get("normalize").register("minmax")
class NormMeth_Mixmax(NormMethod):
	def __call__(self, X):
		mins = X.min(axis=1, keepdims=True)
		span = X.max(axis=1, keepdims=True) - mins
		return (X - mins) / span

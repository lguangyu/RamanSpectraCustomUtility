#!/usr/bin/env python3

import abc
import functools
import numpy
from . import registry


class Normalize1DMethod(object):
	@abc.abstractmethod
	def __call__(self, vec, *ka, **kw) -> numpy.ndarray:
		pass
	@property
	@abc.abstractmethod
	def name_str(self):
		pass
	@property
	def desc_str(self):
		return ""

	def with_transform_vec_type(func):
		def wrap(self, vec, *ka, **kw) -> numpy.ndarray:
			vec = numpy.asarray(vec)
			if vec.ndim != 1:
				raise ValueError("input data must be in 1-d array")
			return func(self, vec, *ka, **kw)
		return wrap


registry.new(registry_name = "normalize_1d", value_type = Normalize1DMethod)


@registry.get("normalize_1d").register("none", as_default = True)
class L1Normalize(Normalize1DMethod):
	@Normalize1DMethod.with_transform_vec_type
	def __call__(self, vec, *ka, **kw) -> numpy.ndarray:
		return vec
	@property
	def name_str(self):
		return "none"
	@property
	def desc_str(self):
		return "do not perform normalization"


@registry.get("normalize_1d").register("l1")
class L1Normalize(Normalize1DMethod):
	@Normalize1DMethod.with_transform_vec_type
	def __call__(self, vec, *ka, **kw) -> numpy.ndarray:
		return vec / numpy.linalg.norm(vec, ord = 1)
	@property
	def name_str(self):
		return "l1-normalize"


@registry.get("normalize_1d").register("l2")
class L2Normalize(Normalize1DMethod):
	@Normalize1DMethod.with_transform_vec_type
	def __call__(self, vec, *ka, **kw) -> numpy.ndarray:
		return vec / numpy.linalg.norm(vec, ord = 2)
	@property
	def name_str(self):
		return "l2-normalize"


@registry.get("normalize_1d").register("max-min")
class MaxMinNormalize(Normalize1DMethod):
	@Normalize1DMethod.with_transform_vec_type
	def __call__(self, vec, *ka, **kw) -> numpy.ndarray:
		ret = vec - vec.min()
		ret /= ret.max()
		return ret
	@property
	def name_str(self):
		return "max-min-normalize"
	@property
	def desc_str(self):
		return "linearly scale to max=1 and min=0"

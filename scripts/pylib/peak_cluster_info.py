#!/usr/bin/env python3

import functools
import numpy
from . import file_util


def read_peak_cluster_info_file(file) -> numpy.ndarray:
	return numpy.loadtxt(file, delimiter = "\t", dtype = float, comments = "#")


def save_peak_cluster_info_file(file, model):
	weights = model.weights_.reshape(-1, 1)
	means = model.means_.reshape(-1, 1)
	covariances = model.covariances_.reshape(-1, 1)
	sort_idx = (means[:, 0]).argsort()
	data = numpy.hstack([weights[sort_idx], means[sort_idx],
		covariances[sort_idx]])
	header = ("\t").join(["weight", "mean", "sigma"])
	numpy.savetxt(file, data, fmt = "%f", delimiter = "\t", header = header,
		comments = "#")
	return


class RangeBasedCluster(object):
	def __init__(self, left_bound, right_bound, *ka, **kw):
		super(RangeBasedCluster, self).__init__(*ka, **kw)
		self.left_bound = left_bound
		self.right_bound = right_bound
		return

	def is_intersect(self, other):
		assert isinstance(other, RangeBasedCluster)
		return ((self.left_bound <= other.right_bound) and\
			(other.left_bound <= self.right_bound))

	def safe_union(self, other) -> bool:
		# return True when union is successful, otherwise false
		if self.is_intersect(other):
			self.left_bound = min(self.left_bound, other.left_bound)
			self.right_bound = max(self.right_bound, other.right_bound)
			return True
		else:
			return

@functools.wraps(read_peak_cluster_info_file)
def read_ranged_peak_cluster_info_file(*ka, **kw):
	# currently these two are the same
	return read_peak_cluster_info_file(*ka, **kw)


def save_ranged_peak_cluster_info_file(file, ranges):
	header = ("\t").join(["left", "right"])
	numpy.savetxt(file, ranges, fmt = "%f", delimiter = "\t", header = header,
		comments = "#")
	return

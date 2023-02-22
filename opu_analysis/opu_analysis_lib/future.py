#!/usr/bin/env python3

import inspect

import sklearn.cluster


def sklearn_cluster_AgglomerativeClustering(*ka, metric=None, **kw):
	cls = sklearn.cluster.AgglomerativeClustering
	if "metric" in inspect.signature(cls.__init__).parameters:
		new = cls(*ka, metric=metric, **kw)
	else:
		new = cls(*ka, affinity=metric, **kw)
	return new

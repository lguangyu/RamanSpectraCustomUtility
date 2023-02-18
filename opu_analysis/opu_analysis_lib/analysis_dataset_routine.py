#!/usr/bin/env python3

import numpy
# custom lib
from . import util
from .spectra_dataset import SpectraDataset


class AnalysisDatasetRoutine(object):
	"""
	routines laoding dataset contig file and prepare spectra data for analysis
	"""
	def __init__(self, dataset: SpectraDataset, *ka,
			biosample=None, biosample_color=None, **kw):
		super().__init__(*ka, **kw)
		self.dataset = dataset
		# assure biosample and biosample_color are assigned after dataset in
		# __init__, otherwise attribute error will occur
		self.biosample = biosample
		self.biosample_color = biosample_color
		return

	@property
	def biosample(self):
		"""
		set/get biosample information corresponding to the spectra in <dataset>;
		value can either be None, a string or a list
		* if None, assumes all spectra from <dataset> are from a same biosample,
			and a default sample name will be used (default)
		* if a string, like None, except that using the string as sample name
		* or a list (of strings), to specify the name of each spectra in
			<dataset>; in addition, the list length must match number of spectra
			in <dataset>
		"""
		return self._biosample
	@biosample.setter
	def biosample(self, value):
		# transform scalar value here if used
		if value is None:
			value = numpy.repeat(["sample"], self.dataset.n_spectra)
		elif isinstance(value, str):
			value = numpy.repeat([value], self.dataset.n_spectra)
		# check biosample is of the same length as n_spectra
		if len(value) != self.dataset.n_spectra:
			raise ValueError("biosample must be None, str, or list with a "
				"length of dataset.n_spectra")
		self._biosample = value
		return

	@property
	def biosample_color(self):
		"""
		set/get biosample color information corresponding to the spectra in
		<dataset>; value can either be None, a string or a list
		* if None, assumes all spectra from <dataset> use the default color
		* if a string, like None, except that using the string as color
		* or a list (of strings), to specify the color of each spectra in
			<dataset>; in addition, the list length must match number of spectra
			in <dataset>
		"""
		return self._biosample_color
	@biosample_color.setter
	def biosample_color(self, value):
		# transform scalar value here if used
		if value is None:
			value = numpy.repeat(["#000000"], self.dataset.n_spectra)
		elif isinstance(value, str):
			value = numpy.repeat([value], self.dataset.n_spectra)
		# check biosample is of the same length as n_spectra
		if len(value) != self.dataset.n_spectra:
			raise ValueError("biosample_color must be None, str, or list with "
				"a length of dataset.n_spectra")
		self._biosample_color = value
		return

	@classmethod
	def load_json_config(cls, cfg_file, *, reconcile_param=None, **kw):
		"""
		load dataset config json file
		the config should be in a list structure with dictionary elements; each
		element could have at least below key/value pairs:
		file: the dataset file in tsv format (required)
		name: the dataset name (optional), the file name is used if omitted
		color: the dataset color used in plot, default to black (optional)
		"""
		cfg = util.load_json(cfg_file)
		if reconcile_param is None: reconcile_param = dict()
		dataset_list = [SpectraDataset.from_file(c["file"], **reconcile_param) \
			for c in cfg]
		dataset = SpectraDataset.concatenate(*dataset_list)
		# construct the biosample and biosample color lists based on the config
		biosample = list()
		biosample_color = list()
		for c, d in zip(cfg, dataset_list):
			biosample.extend([c["name"]] * d.n_spectra)
			biosample_color.extend([c.get("color", "#000000")] * d.n_spectra)

		new = cls(dataset, biosample=biosample, biosample_color=biosample_color,
			**kw)
		return new
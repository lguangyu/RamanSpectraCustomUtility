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
	def from_config(cls, cfg: list, *, reconcile_param=None, **kw):
		"""
		prepare dataset by config

		the config should be in a list structure with dictionary elements; each
		element could have at least below key/value pairs:
		file: the dataset file in tsv format (required)
		name: the dataset name (optional), the file name is used if omitted
		color: the dataset color used in plot, default to black (optional)
		"""
		if reconcile_param is None:
			reconcile_param = dict()
		dataset_list = list()
		for c in cfg:
			if isinstance(c["file"], str):
				dataset_list.append(SpectraDataset.from_file(c["file"],
					name=c["name"], **reconcile_param
					# without setting spectra_names_override it will deduct from
					# name, i.e. spectra_names_override=c["name"]
				))
			elif isinstance(c["file"], list):
				dataset_list.extend([SpectraDataset.from_file(f,
					name=c["name"], spectra_names_override=f, **reconcile_param
				) for f in c["file"]])
			else:
				raise ValueError("'file' field of the dataset config json must "
					"be str or list, got '%s'" % type(c["file"]).__name__)
		dataset = SpectraDataset.concatenate(*dataset_list)
		# construct the biosample and biosample color lists
		biosample = list()
		for d in dataset_list:
			biosample.extend([d.name] * d.n_spectra)
		# for biosample_color, using the biosample list and the original cfg
		biosample_color_map = {c["name"]: c.get("color", "#000000")
			for c in cfg}
		biosample_color = [biosample_color_map[k] for k in biosample]

		new = cls(dataset, biosample=biosample, biosample_color=biosample_color,
			**kw)
		return new

	@classmethod
	def from_config_json(cls, cfg_file, *, reconcile_param=None, **kw):
		"""
		prepare dataset by config defined in json

		see from_config() for details about a proper format of config
		"""
		cfg = util.load_json(cfg_file)
		return cls.from_config(cfg, reconcile_param=reconcile_param, **kw)

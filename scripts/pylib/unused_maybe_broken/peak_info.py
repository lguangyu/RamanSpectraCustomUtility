#!/usr/bin/env python3

import numpy
from . import file_util


class PeakInfo(object):
	def __init__(self, *ka, title = None, indices = None, shifts = None,
			intensities = None, **kw):
		super(PeakInfo, self).__init__(*ka, **kw)
		self.title = title
		self.indices = list() if indices is None else indices
		self.shifts = list() if shifts is None else shifts
		self.intensities = list() if intensities is None else intensities
		assert len(self.indices) == len(self.shifts) == len(self.intensities)
		return

	def add_peak(self, *, index: int, shift: float, intensity: float):
		self.indices.append(int(index))
		self.shifts.append(float(shift))
		self.intensities.append(float(intensity))
		return

	@staticmethod
	def breakup_commas(s: str, *, dtype = float):
		if not s:
			return numpy.empty(0, dtype = dtype)
		else:
			return numpy.asarray(s.split(","), dtype = dtype)

	@classmethod
	def from_str(cls, line: str, *ka, **kw):
		new = cls(*ka, **kw)
		title, index, shift, intens = line.rstrip("\r\n").split("\t")
		new = cls(*ka, title = title,
			indices = cls.breakup_commas(index, dtype = int),
			shifts = cls.breakup_commas(shift, dtype = float),
			intensities = cls.breakup_commas(intens, dtype = float))
		return new

	def to_str(self):
		index = (",").join([str(i) for i in self.indices])
		shift = (",").join([str(i) for i in self.shifts])
		intens = (",").join([str(i) for i in self.intensities])
		return ("\t").join([self.title, index, shift, intens])

	def filter_peak_by_shift_range(self, left, right):
		ret = type(self)(title = self.title)
		assert len(ret.indices) == 0
		for i, s, t in zip(self.indices, self.shifts, self.intensities):
			if (s >= left) and (s <= right):
				ret.add_peak(index = i, shift = s, intensity = t)
		return ret

	def is_empty(self):
		#return len(self.indices) == 0
		return not bool(self.indices)



def read_peak_info_file(file) -> dict:
	ret = dict()
	with file_util.get_fp(file, "r") as fp:
		for line in fp:
			if line.startswith("#"):
				continue
			peak_info = PeakInfo.from_str(line)
			ret[peak_info.title] = peak_info
	return ret

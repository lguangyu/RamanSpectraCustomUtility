#!/usr/bin/env python3

import numpy
import sys
# custom lib
from . import util
from . import json_io


class RamanSpectra(json_io.JSONSerializable):
	def __init__(self, title: str, wavnum, inten, *ka, **kw):
		super().__init__(*ka, **kw)
		self.title = title
		self.set_data(wavnum, inten)
		return

	def set_data(self, wavnum, inten):
		self.wavnum	= numpy.asarray(wavnum, dtype = float)
		self.inten	= numpy.asarray(inten, dtype = float)
		self.check_wavenum_inten_integrity()
		return

	def check_wavenum_inten_integrity(self):
		if len(self.wavnum) != len(self.inten):
			raise ValueError("wavenumber and intensity lengths mismatch")
		if numpy.any(numpy.diff(self.wavnum) <= 0):
			raise ValueError("wavenumber must be monotonically increasing")
		return

	def copy(self, *ka, **kw):
		new = type(self)(
			title	= self.title,
			wavnum	= self.wavnum.copy(),
			inten	= self.inten.copy(),
		)
		return new

	@property
	def max_intensity(self):
		return self.inten.max()
	@property
	def min_intensity(self):
		return self.inten.max()

	def filter_by_wavnum_range(self, min_wn, max_wn, *, inplace = False):
		mask = (self.wavnum >= min_wn) & (self.wavnum <= max_wn)
		ret = self if inplace else self.copy()
		ret.set_data(ret.wavnum[mask], ret.inten[mask])
		return ret

	def bin(self, min_wn, max_wn, bin_size = None, *, inplace = False):
		# first, filter range
		ret = self.filter_by_wavnum_range(min_wn, max_wn, inplace = inplace)

		# check if we need to do bin, None = no bin needed
		if bin_size is None:
			return ret

		# ensure bin_size is positive
		bin_size = util.PosFloat(bin_size)

		# find the binning windows
		bin_bounds	= numpy.arange(min_wn, max_wn, bin_size, dtype = float)
		bin_wavnum	= (bin_bounds[:-1] + bin_bounds[1:]) / 2
		bin_inten	= numpy.empty(len(bin_wavnum), dtype = float)

		# calculate bin-intensity as the average
		for i in range(len(bin_wavnum)):
			b0, b1 = bin_bounds[i], bin_bounds[i + 1]
			mask = (b0 <= ret.wavnum) & (ret.wavnum < b1)
			if not mask.any():
				print("warning: bin window %.3f-%.3f contains no data; "
					"treat as 0 intensity" % (b0, b1), file = sys.stderr)
				bin_inten[i] = 0
			else:
				bin_inten[i] = ret.inten[mask].mean()
		ret.set_data(bin_wavnum, bin_inten)

		return ret

	def json_serialize(self):
		return dict(
			title	= self.title,
			wavnum	= self.wavnum.tolist(),
			inten	= self.inten.tolist()
		)

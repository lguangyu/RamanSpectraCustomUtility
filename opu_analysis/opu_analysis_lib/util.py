#!/usr/bin/python3

import io
import json
import sys
import typing


class PosInt(int):
	def __new__(cls, *ka, **kw):
		new = super().__new__(cls, *ka, **kw)
		if new <= 0:
			raise ValueError("%s cannot be 0 or negative, got '%d'"
				% (cls.__name__, new))
		return new


class NonNegInt(int):
	def __new__(cls, *ka, **kw):
		new = super().__new__(cls, *ka, **kw)
		if new < 0:
			raise ValueError("%s cannot be negative, got '%d'"
				% (cls.__name__, new))
		return new


class PosFloat(float):
	def __new__(cls, *ka, **kw):
		new = super().__new__(cls, *ka, **kw)
		if new <= 0:
			raise ValueError("%s cannot be 0 or negative, got '%f'"
				% (cls.__name__, new))
		return new


class NonNegFloat(float):
	def __new__(cls, *ka, **kw):
		new = super().__new__(cls, *ka, **kw)
		if new < 0:
			raise ValueError("%s cannot be negative, got '%f'"
				% (cls.__name__, new))
		return new


class Fraction(float):
	def __new__(cls, *ka, **kw):
		new = super().__new__(cls, *ka, **kw)
		if (new < 0) or (new > 1):
			raise ValueError("%s cannot be less than 0 or greater than 1, got "
				"'%f'" % (cls.__name__, new))
		return new


def get_fp(f, *ka, factory=open, **kw):
	"""
	wrapper of file handle open function (default: builtin.open) to make it safe
	to call upon already-opened file handle as the first argument;
	many non-system file open functions already have this functionality, e.g.
	numpy.loadtxt()
	"""
	if isinstance(f, io.IOBase):
		# this does not check the opened mode of the file handle, can cause I/O
		# error if mode is wrong (e.g. attempt to write on handle of mode 'r')
		ret = f
	elif isinstance(f, str):
		ret = factory(f, *ka, **kw)
	else:
		raise TypeError("get_fp: first argument must be instance of io.IOBase "
			"or str, got '%s'" % type(f).__name__)
	return ret


def load_json(f, *ka, **kw):
	with get_fp(f, "r") as fp:
		ret = json.load(fp, *ka, **kw)
	return ret


class CyclicIndexedList(list):
	"""
	a list subclass that allow index values to go infinity
	"""

	def __getitem__(self, index):
		return super().__getitem__(index % len(self))


def drop_replicate(seq):
	"""
	remove replicate in iterable seq while preseveing the original order by
	encountering
	"""
	assert set().add(1) is None, "this algorithm assumes set.add() always "\
		"returns None; should this fail, place with alternative ways"
	seen = set()
	return [i for i in seq if not (i in seen or seen.add(i))]

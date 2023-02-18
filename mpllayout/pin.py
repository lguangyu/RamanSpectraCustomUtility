#!/usr/bin/env python3

import math
from . import PlaceableElementBase


class PinBase(PlaceableElementBase):
	def __init__(self, *ka, **kw):
		super().__init__(*ka, **kw)
		# init placement
		self.clear_placement()
		self.clear_placement_ref()
		return

	def is_placeable(self):
		# a point is placeable if it has a reference point set
		return (self.figpos_ref is not None)

	def is_placed(self):
		return (self.figpos is not None)

	def get_placement(self) -> float:
		return self.figpos

	def set_placement(self, value: float):
		self.figpos = value
		return

	def clear_placement(self):
		self.set_placement(None)
		return

	def get_placement_ref(self):
		return self.get_placement_ref_pin(), self.get_placement_ref_offset()

	def get_placement_ref_pin(self):
		return self.figpos_ref

	def get_placement_ref_offset(self):
		return self.figpos_offset

	def set_placement_ref(self, ref, offset: float):
		self.figpos_ref = ref
		self.figpos_offset = float(offset)
		return

	def clear_placement_ref(self):
		self.set_placement_ref(None, 0.0)
		return

	def _get_dependency(self):
		# a point only allows one dependency (at most)
		return self.get_placement_ref_pin()

	def get_dependencies(self):
		dep = self._get_dependency()
		return tuple() if dep is None else (dep, )

	def solve_placement(self):
		# when calling this method, assumes not self.is_placed() (by base class)
		# and further assumes self.is_placeable() 
		if not self.figpos_ref.is_placed():
			raise type(self).PlacementUnsolvableError("dependency '%s' ("
				"required by '%s') cannot be solved"\
				% (self.figpos_ref.global_name, self.global_name))
		# else solve position based on the dependency
		self.set_placement(self.figpos_ref.figpos + self.figpos_offset)
		return

	def verify_placement(self):
		if self.is_placeable():
			if (not self.is_placed()) or (not self.figpos_ref.is_placed()):
				raise type(self).PlacementError("verify_placement() must be "
					"called after placement being solved")
			# check if figpos diff is the same as offset
			offset = self.get_placement() - self.figpos_ref.get_placement()
			if not math.isclose(offset, self.figpos_offset):
				raise type(self).IncomplyingPlacementError(
					"resolved %s and %s do not comply with offset '%.5f'; "
					"usually due to assigned conflicting positional relations"\
					% (str(self), str(self.figpos_ref), self.figpos_offset))
		return

	def __str__(self) -> str:
		return "%s (%.6f)" % (self.global_name, self.get_placement())


class RulerPin(PinBase):
	def __init__(self, *ka, ruler_pos: float, **kw):
		super().__init__(*ka, **kw)
		# RulerPin requires explicit parent, not defaulting to None
		if self.parent is None:
			raise ValueError("parent of RulerPin must be not None")
		# relative position wihtin parent Ruler
		self.ruler_pos = ruler_pos
		return

	def _get_dependency(self):
		# if placement ref pin is set, parent is not a dependency
		ret = super()._get_dependency()
		if ret is None:
			ret = self.parent
		return ret

#!/usr/bin/env python3

import abc
import math
from . import PlaceableElementBase, RulerPin


class RulerBase(PlaceableElementBase):
	def __init__(self, *ka, **kw):
		super().__init__(*ka, **kw)
		self.pins_dict = dict()
		self.init_pins()
		return

	def create_pin(self, name, *ka, pin_type = RulerPin, **kw) -> RulerPin:
		"""
		create a RulerPin instance owned by 'self'. should be used as the only
		factory of RulerPin subclasses. *ka and **kw are fowarded to
		pin_type.__init__(). return the created RulerPin instance. the created
		pin will be stored in self.pins_dict dict.

		the created RulerPin object can be accessed as an attribute of the parent
		Ruler by self.<name>_pin
		"""
		if name in self.pins_dict:
			raise ValueError("pin '%s' already exists" % name)
		attr_name = name + "_pin"
		if hasattr(self, attr_name):
			raise ValueError("attr '%s' conflicts with existing attribute name "
				"in class '%s'" % (attr_name, type(self).__name__))
		# create pin: (signature) parent, name, *ka, **kw
		pin = pin_type(name, *ka, parent = self, **kw)
		self.pins_dict[name] = pin
		setattr(self, attr_name, pin)
		return pin

	def get_pin(self, name: str):
		"""
		get the child pin by <name>
		"""
		return self.pins_dict[name]

	def iter_pins(self):
		"""
		return an iterator to all children pins
		"""
		return self.pins_dict.values()

	def iter_pin_names(self):
		"""
		return an iterator to names of all children pins
		"""
		return self.pins_dict.keys()

	@abc.abstractmethod
	def init_pins(self):
		"""
		implemented by subclasses to define and init pins. must use create_pin()
		method to create any pin (not manually call RulerPin class!).
		"""
		return

	@abc.abstractmethod
	def calc_ruler_length(self):
		"""
		calculate ruler length using pins that are already placed. this might be
		infeasible if not enought pins are already placed. in such cases, must
		return None
		"""
		return

	def get_ruler_length(self, *, allow_calculated = False):
		"""
		ruler_length is the unit length of ruler position. this method will
		first check if it has already been set (by set_ruler_length()), and
		return it if so; otherwise it depends on the value of allow_calculated.
		if True, it can try to calculate it using method calc_ruler_length(),
		and return whatever the method results in (will be None if the length
		cannot be calculated). if allow_calculated = False, the method will
		return None directly set_ruler_length() has not been used for a non-None
		value.
		"""
		ret = self._ruler_length
		if (ret is None) and allow_calculated:
			ret = self.calc_ruler_length()
		return ret

	def set_ruler_length(self, value: float):
		if (value is not None) and (value < 0):
			raise ValueError("ruler length cannot be negative (%f)" % value)
		self._ruler_length = value
		return

	def clear_ruler_length(self):
		return self.set_ruler_length(None)

	def is_placed(self):
		# a Ruler is placed only if all RulerPin's are placed
		return all([p.is_placed() for p in self.iter_pins()])

	def get_placement(self):
		return set([p.get_placement() for p in self.iter_pins()])

	def set_placement(self):
		raise NotImplementedError("%s does not support this method; use "
			"solve_placement() instead" % type(self).__name__)
		return

	def clear_placement(self):
		for p in self.iter_pins():
			p.clear_placement()
		return

	def get_dependencies(self):
		# a pin is a dependency of the parent Ruler only if it has dependency set
		ret = [p for p in self.iter_pins() if p.is_placeable()]
		return ret


class LinearRuler(RulerBase):
	def __init__(self, *ka, **kw):
		super().__init__(*ka, **kw)
		self._ruler_length = None
		return

	def calc_ruler_length(self):
		placed_pins = self._get_anchorable_pins("placed")
		if len(placed_pins) < 2:
			ret = 0
		else:
			rp1, rp2, *_ = placed_pins
			if rp1.ruler_pos == rp2.ruler_pos:
				ret = 0
			else:
				ret = (rp1.get_placement() - rp2.get_placement())\
					/ (rp1.ruler_pos - rp2.ruler_pos)
		return ret

	def init_pins(self):
		self.create_pin("pmax", ruler_pos = 1.0)
		self.create_pin("pmid", ruler_pos = 0.5)
		self.create_pin("pmin", ruler_pos = 0.0)
		return

	def is_placeable(self):
		# LinearRuler requires at least one point to be placeable (set with
		# referennce) and an extra point, or ulength set.
		count = len(self._get_anchorable_pins("both"))
		if self.get_ruler_length(allow_calculated = False) is not None:
			count += 1
		return (count >= 2)

	def _get_anchorable_pins(self, filt = "both") -> set:
		"""
		get pins that can potentially be used as anchorage when solving for
		ruler placement. 'anchorable pins' refer to those either already has
		placement (with non-None on-figure position) or has placement
		reference set. argument 'filt' specifies what type of anchorage to
		choose, or both.

		ARGUMENT
		--------
		filt: str (default: 'both')
			which pins to filter, choose from 'placed', 'placeable' or 'both'
		"""
		if filt == "both":
			cond = lambda p: (p.is_placed() or p.is_placeable())
		elif filt == "placed":
			cond = lambda p: p.is_placed()
		elif filt == "placeable":
			cond = lambda p: p.is_placeable()
		else:
			raise ValueError("unrecognized filt value: '%s'" % filt)
		return set(filter(cond, self.iter_pins()))

	def solve_placement(self):
		# solve a LinearRuler differs based on if ulength is set. if ulength is
		# set we will try to use that first, plus the first pin location found
		if not self.is_placeable():
			raise type(self).PlacementUnsolvableError("insufficient information"
				" to solve '%s'" % self.global_name)
		rlen = self.get_ruler_length(allow_calculated = True)
		if rlen is None:
			raise type(self).DependencyUnplacedError("solve_placement() called "
				"before dependency placements being solved. if this error "
				"message raises from a manual invocation of solve_placement(), "
				"try call solve_placement_resursive() instead")
		# place pins with rlen (ruler length) and one placed point
		# here we don't need to check if self._get_anchorable_pins() returns an
		# empty set; if so, self.is_placeable() should alread raise an error
		rp, *_ = self._get_anchorable_pins("placed")
		for p in self.iter_pins():
			if p.is_placed():
				continue
			p.set_placement(rp.get_placement()\
				+ (p.ruler_pos - rp.ruler_pos) * rlen)
		return

	def verify_placement(self):
		# TODO: need to think if these are enough.
		for p in self.iter_pins():
			p.verify_placement()
		# check if pmid is right in the middle between pmax and pmin
		if not math.isclose(self.pmax_pin.get_placement()\
			+ self.pmin_pin.get_placement(), self.pmid_pin.get_placement() * 2):
			raise type(self).IncomplyingPlacementError("resolved %s, %s and %s "
				"does not comply with in-ruler positions"\
				% (str(self.pmin_pin), str(self.pmid_pin), str(self.pmax_pin)))
		if self.pmax_pin.get_placement() < self.pmin_pin.get_placement():
			raise type(self).IncomplyingPlacementError("resolved %s position "
				"less than %s implies negative ruler length"\
				% (str(self.pmax_pin), str(self.pmin_pin)))
		return

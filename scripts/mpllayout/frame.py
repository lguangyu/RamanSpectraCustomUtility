#!/usr/bin/env python3

import abc
from . import PlaceableElementBase, RulerPin, LinearRuler


class FrameBase(PlaceableElementBase):
	"""
	FrameBase is the base class for all Frame-type elements. Frame is a special
	type of placeable elements with a user-friendly interface using anchors.
	anchors are high-level wraps of low-level elements, such as RulerPin's. to
	specify a Frame relative to another Frame, one need to specify some of its
	anchors in relative to the other Frame's anchor. anchors are also labelled
	withe human-readability such as "top-left" to refer to the point on the of
	the top-left corner of a rectangle.
	"""
	@abc.abstractmethod
	def get_ruler_pins_by_anchor_name(self, anchor: str): pass

	@abc.abstractmethod
	def set_anchor(self, anchor, ref_frame, ref_anchor, offsets): pass

	@abc.abstractmethod
	def get_anchor(self, anchor): pass

	@abc.abstractmethod
	def clear_anchor(self, anchor): pass

	@abc.abstractmethod
	def iter_anchor_names(self): pass

	@abc.abstractmethod
	def get_ruler_pins_by_anchor_name(self, anchor): pass

	@abc.abstractmethod
	def get_extent(self): pass


class Frame2DBase(FrameBase):
	"""
	a frame has two dimensions, horizontal and vertical. each dimension is
	represented by a ruler instance. solving placement of a Frame2DBase is
	essentially resolving the placement of both rulers. therefore, many of the
	placement-related methods in this class are summarized from or forwarded to
	those dependency rulers.
	"""
	def __init__(self, *ka, **kw):
		super().__init__(*ka, **kw)
		self._h_ruler = LinearRuler("h_ruler", parent = self)
		self._v_ruler = LinearRuler("v_ruler", parent = self)
		return

	############################################################################
	# many of below functions are simple summary or forwarding of corresponding
	# methods to dependency elements
	def get_extent(self):
		return \
			self._h_ruler.pmin_pin.get_placement(),\
			self._h_ruler.pmax_pin.get_placement(),\
			self._v_ruler.pmin_pin.get_placement(),\
			self._v_ruler.pmax_pin.get_placement()

	def is_placeable(self):
		return all([dep.is_placeable() for dep in self.get_dependencies()])

	def is_placed(self):
		return all([dep.is_placed() for dep in self.get_dependencies()])

	def get_placement(self, anchor):
		p1, p2 = self.get_ruler_pins_by_anchor_name(anchor)
		return p1.get_placement(), p2.get_placement()

	def set_placement(self, anchor, position):
		"""
		directly set a anchor on-figure position; doing this will erase the
		previously set anchor placement ref
		"""
		if not ((isinstance(position, tuple) or isinstance(offsets, list))\
			and (len(position) == 2)):
			raise TypeError("position must be a tuple or list of length 2")
		pins = self.get_ruler_pins_by_anchor_name(anchor)
		for pin, pos in zip(pins, position):
			pin.clear_placement_ref()
			pin.set_placement(pos)
		return

	def clear_placement(self, anchor = None):
		"""
		clear anchor placement; if anchor = None, clear all.
		"""
		_it = self.iter_anchor_names() if anchor is None else (anchor, )
		for p1, p2 in map(lambda a: self.get_ruler_pins_by_anchor_name(a), _it):
			p1.clear_placement()
			p2.clear_placement()
		return

	def get_anchor(self, anchor):
		p1, p2 = self.get_ruler_pins_by_anchor_name(anchor)
		return p1.get_placement_ref(), p2.get_placement_ref()

	def set_anchor(self, anchor, ref_frame = None, ref_anchor = None,
			offsets = None):
		"""
		set self's <anchor> in reference to ref_frame's <ref_anchor> with given
		offsets in horizontal/vertical coordinates.
		if ref_anchor is None, it is assumed to be the same as <anchor>
		"""
		# check ref_frame type and resolve ref_frame (if is None)
		if ref_frame is None:
			if self.parent is None:
				raise ValueError("ref_frame = None assumes using parent as the "
					"reference frame. however the parent of %s is None"\
					% self.global_name)
			ref_frame = self.parent
		elif not isinstance(ref_frame, Frame2DBase):
			raise TypeError("ref_frame must be instance of Frame2DBase or None,"
				" not '%s'" % type(ref_frame).__name__)
		if offsets is None:
			offsets = (0.0, 0.0)
		if not ((isinstance(offsets, tuple) or isinstance(offsets, list))\
			and (len(offsets) == 2)):
			raise TypeError("offsets must be a tuple or list of length 2")
		if ref_anchor is None:
			ref_anchor = anchor
		for pin, pref, offset in zip(self.get_ruler_pins_by_anchor_name(anchor),
				ref_frame.get_ruler_pins_by_anchor_name(ref_anchor), offsets):
			pin.set_placement_ref(pref, offset = float(offset))
		return

	def clear_anchor(self, anchor):
		for p in self.get_ruler_pins_by_anchor_name(anchor):
			p.clear_placement_ref()
		return

	def get_dependencies(self):
		return self._h_ruler, self._v_ruler

	def solve_placement(self):
		for dep in self.get_dependencies():
			dep.solve_placement()
		return

	def verify_placement(self):
		for dep in self.get_dependencies():
			dep.verify_placement()
		return


class RectangularFrame(Frame2DBase):
	"""
	a RectangularFrame is almost a fundamental Frame2DBase. it has anchors
	defined by a 3x3 grid, namely (bottom, center, top) x (left, center, right).
	note anchor 'bottomcenter' is simplified as 'bottom', similar to other three
	edge centers. the frame center ('centercenter') is simplified as 'center'.
	anchor name should be case insentitive.

	RectangularFrame also provides the functionality to directly set 'width' or
	'height' (or combined as 'size'), which is by setting the ruler length of
	horizontal or vertical ruler respectively.
	"""

	__anchors_getter = {
		# bottom anchors
		"bottomleft":	lambda o: (o._h_ruler.pmin_pin, o._v_ruler.pmin_pin),
		"bottom":		lambda o: (o._h_ruler.pmid_pin, o._v_ruler.pmin_pin),
		"bottomright":	lambda o: (o._h_ruler.pmax_pin, o._v_ruler.pmin_pin),
		# vertical center anchors
		"left":			lambda o: (o._h_ruler.pmin_pin, o._v_ruler.pmid_pin),
		"center":		lambda o: (o._h_ruler.pmid_pin, o._v_ruler.pmid_pin),
		"right":		lambda o: (o._h_ruler.pmax_pin, o._v_ruler.pmid_pin),
		# top anchors
		"topleft":		lambda o: (o._h_ruler.pmin_pin, o._v_ruler.pmax_pin),
		"top":			lambda o: (o._h_ruler.pmid_pin, o._v_ruler.pmax_pin),
		"topright":		lambda o: (o._h_ruler.pmax_pin, o._v_ruler.pmax_pin),
	}

	def iter_anchor_names(self):
		return self.__anchors_getter.keys()

	def get_ruler_pins_by_anchor_name(self, anchor) -> (RulerPin, RulerPin):
		anchor = anchor.lower() # make case insensitive
		if anchor not in self.__anchors_getter:
			raise ValueError("%s has no anchor '%s'"\
				% (type(self).__name__, anchor))
		return self.__anchors_getter[anchor](self)

	def get_width(self):
		return self._h_ruler.get_ruler_length(allow_calculated = True)

	def set_width(self, value):
		self._h_ruler.set_ruler_length(value)
		return

	def get_height(self):
		return self._v_ruler.get_ruler_length(allow_calculated = True)

	def set_height(self, value):
		self._v_ruler.set_ruler_length(value)
		return

	def get_size(self):
		return self.get_width(), self.get_height()

	def set_size(self, width, height):
		self.set_width(width)
		self.set_height(height)
		return

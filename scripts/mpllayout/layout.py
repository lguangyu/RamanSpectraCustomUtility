#!/usr/bin/env python3

import abc
import importlib
import warnings
from . import FrameBase, RectangularFrame


class LayoutElementFrameBase(abc.ABC):
	@abc.abstractmethod
	def create_artist(self, figure, *, left_margin, right_margin, bottom_margin,
			top_margin):
		"""
		create the matplotlib artist object. this method should be overridden by
		subclasses for different artist object factory, while the calling
		signature must be kept identical. this function is internally called by
		LayoutCreator.create_figure_layout().

		ARGUMENT
		--------
		figure:
			an instance of matplotlib.figure.Figure to create the artist on
		left_margin, right_margin, bottom_margin, top_margin (float):
			all are keyword-only arguments. these provides extra information to
			so that the margins can be calculated. the margins are filled out by
			LayoutCreator.create_figure_layout() internally.
		"""
		return


class LayoutAxesFrame(RectangularFrame, LayoutElementFrameBase):
	def __init__(self, *ka, axes_class = None, **kw):
		super().__init__(*ka, **kw)
		self.axes_class = axes_class
		return

	def create_artist(self, figure, *, left_margin, right_margin, bottom_margin,
			top_margin):
		fl, fr, fb, ft = self.parent.get_extent()
		al, ar, ab, at = self.get_extent()
		# calculate figure dimensions
		fwidth	= left_margin + (fr - fl) + right_margin
		fheight	= bottom_margin + (ft - fb) + top_margin
		# calcualte axes dimensions
		left	= (al - fl + left_margin) / fwidth
		bottom	= (ab - fb + bottom_margin) / fheight
		width	= (ar - al) / fwidth
		height	= (at - ab) / fheight
		# create axes
		ret = figure.add_axes([left, bottom, width, height],
			axes_class = self.axes_class)
		return ret


class LayoutCreator(object):
	"""
	management of axes layouts (as frames) on figure and can format a matplotlib
	Figure object with the configured layout. layout is created by user-defined
	relationships between axes on a figure in real-world units (inches). this
	method is intended to precisely create figure layouts with multiple axes of
	desired sizes and relations.

	ATTRIBUTES
	----------
	origin:
		the reference origin of the plottable area (known as canvas).
	width, height:
		the horizontal and vertical span (in inches) of the plottable area. if
		None, it will be automatically determined.
	left_margin, right_margin, top_margin, bottom_margin:
		the extra white margins *OUTSIDE* of the plottable area, i.e. the final
		figure width will be left_margin + width + right_margin. in inches.

	SYNOPSIS
	--------
	1. CREATE A NEW CREATOR OBJECT
	>>> lc = LayoutCreator("my-layout")

	the argument "my-layout" is optional. extra arguments include:
	origin (str): ["bottomleft"]
		a string that specifies which anchor point is used by other axes to
		locate themselves. this is by default "bottomleft" as the point with
		coordinate (0, 0) natively in matplotlib Figure objects.
	canvas_width, canvas_height (float): [None]
		specifies the width and height of the canvas frame. if left as None, the
		width and height will be determined automatically by fitting to the
		smallest rectangular box that can contain all axes frames, if such can
		physically be feasible. note some carefully tailored cases may break the
		feasibility, for example by delibrately place all axes below the bottom
		edge of the canvas frame. in such cases the width and height will be not
		reliable in automatic mode.
	left_margin, right_margin, top_margin, bottom_margin (float): [0.5]
		the margins define extra white spaces between the canvas frame edges and
		figure edges. if set as 0, the out-most axes may have some spines
		in-line with figure edges.

	2. CREATE AND SPECIFY THE FIRST AXES FRAME
	>>> ax1 = lc.add_frame("my-main-plot")
	>>> ax1.set_anchor("bottomleft", offsets = (0.5, 0.2))
	>>> ax1.set_size(4, 2.5)

	lc.add_frame() creates a frame and adds it to lc's axes dict. the newly
	created frame will be returned.
	ax1.set_anchor() is the most useful method to specify how a frame should be
	placed relative to another frame. the calling signature is:

	.set_anchor(anchor, ref_frame = None, ref_anchor = None, offsets = (0, 0))
	where:
	anchor (str):
		a required string argument that specifies which anchor point of the
		aligning frame should be used
	ref_frame (frame object): [None]
		an optional argument specifies which frame will be used as the reference
		frame. if left as None, the parent frame of the aligning frame will be
		used. by default, parent of all axes frames is the canvas frame of the
		current LayoutCreator object.
	ref_anchor (str): [None]
		an optional argument species which anchor point of the reference frame
		will aligned. if left as None, assumes the same as <anchor>.
	offsets (tuple): [(0, 0)]
		an optional argument species the offset of the aligning anchor in
		relative to the reference anchor, with 'right' and 'up' defined as
		positive direction in horizontal and vertical dimensions respectively.

	in above cases, the call to
	>>> ax1.set_anchor("bottomleft", offsets = (0.5, 0.2))

	is essentially the same as
	>>> ax1.set_anchor("bottomleft", lc.canvas_frame, "bottomleft",
	>>>		offsets = (0.5, 0.2))

	i.e. the "bottomleft" anchor of ax1 should be aligned to the point which is
	0.5-inch to the right and 0.2-inch to the top of the "bottomleft" anchor of
	the canvas frame, 

	then the call to ax1.set_size() specifies the width and height of the ax1
	frame in inches. so far, with one anchor point fully specified and width and
	height determined, the ax1 frame is fully determined.

	3. ADD A SECOND AXES FRAME
	>>> ax2 = lc.add_frame("my-side-plot")
	>>> ax2.set_anchor("bottomleft", ax1, "right", offsets = (0.5, 0.0))
	>>> ax2.set_anchor("topright", ax1, offsets = (2.5, 0.0))

	now we use two anchor points rather than an anchor and the size to define
	a frame. the first ax2.set_anchor() call alignes ax2's "bottomleft" anchor
	to the previous frame ax1's "right" (center of right edge) with x-offset
	of +0.5 inch and y-offset of 0.0 inch. the second ax2.set_anchor() aligns
	ax2's "topright" anchor to ax1's "topright" with a (+2.5, 0.0) offset. now
	with "bottomleft" and "topright" anchors set, the ax2 frame's placement is
	also fully determined. more importantly, ax2 is ensured to be aligned to
	ax1's top half, namely its top edge will be exactly the same as ax1's and
	bottom edge as ax1's vertical center. this is achieved with specifying the
	anchor positions only, without explicit calculating the frame coordinates
	by the user.

	4. CREATE THE LAYOUT ON FIGURE
	>>> layout = lc.create_figure_layout()

	now it's time to sketch the configured layout on matplotlib Figure. this can
	be done by call create_figure_layout() method of the LayoutCreator object.
	this method takes two arguments:
	figure (matplotlib.figure.Figure): [None]
		an instance of Figure class to create the axes on. if left as None, a
		new Figure instance will be created from scratch and configured.
	force (bool): [False] [keyword-only]
		by default (False), create_figure_layout() method will check if the
		destiny Figure object is clean, to prevent messing up things already
		configured/existed on the figure. if <figure> is not clean, a
		NonEmptyFigureError will be raised. however, by setting force = True,
		the destiny Figure object will be modified anyway whether it's clean or
		not.

	method create_figure_layout() returns a dict contains the Figure object and
	all axes that are created. the figure can be accessed via key "figure" and
	other the axes can be accessed by the name used in respective add_frame()
	calls, such as:
	>>> layout["figure"] # the input/created Figure instance
	>>> layout["my-main-plot"].plot(...)
	>>> layout["my-side-plot"].plot(...)
	"""
	class NonEmptyFigureError(RuntimeError): pass
	class ElementNotPlacedError(RuntimeError): pass

	def __init__(self, name = "unnamed-layout", *ka,
			origin = "bottomleft", canvas_width = None, canvas_height = None,
			left_margin = 0.5, right_margin = 0.5, top_margin = 0.5,
			bottom_margin = 0.5, **kw):
		super().__init__(*ka, **kw)
		self.name			= name
		self.origin			= origin
		self.left_margin	= left_margin
		self.right_margin	= right_margin
		self.top_margin		= top_margin
		self.bottom_margin	= bottom_margin
		self.canvas_frame.set_size(canvas_width, canvas_height)
		self.__frames = dict() # the dict holds all created frames
		return

	@property
	def canvas_frame(self):
		"""
		the plottable frame used to represent the area that all axes placement
		refers to.
		"""
		if not hasattr(self, "_canvas_frame"):
			# "canvas" as we call it
			self._canvas_frame = RectangularFrame("canvas")
		return self._canvas_frame

	def get_canvas_width(self):
		return self.canvas_frame.get_width()

	def get_canvas_height(self):
		return self.canvas_frame.get_height()

	def get_canvas_size(self):
		return self.canvas_frame.get_size()

	def set_canvas_width(self, value):
		return self.canvas_frame.set_width(value)

	def set_canvas_height(self, value):
		return self.canvas_frame.set_height(value)

	def set_canvas_size(self, width, height):
		return self.canvas_frame.set_size(width, height)

	@property
	def matplotlib(self):
		"""
		lazy import matplotlib as an attribute; can get access to this module as
		self.matplotlib
		"""
		if not hasattr(self, "_matplotlib"):
			module = importlib.import_module("matplotlib")
			# this should have imported matplotlib.figure and matplotlib.axes
			importlib.import_module("matplotlib.pyplot", package = module)
			self._matplotlib = module
		return self._matplotlib

	############################################################################
	# matplotlib.figure.Figure-related methods
	@staticmethod
	def _is_figure_empty(figure):
		"""
		return True if a matplotlib.figure.Figure object is created anew (empty)
		"""
		# newly created figures should only have one patch (background patch?)
		return len(figure.get_children()) == 1

	def _get_figure(self, figure, force):
		# create the figure if input <figure> is None
		if figure is None:
			figure = self.matplotlib.figure.Figure()
		if not (self._is_figure_empty(figure) or force):
			raise type(self).NonEmptyFigureError("input figure is not empty; "
				"use a figure object created from anew or set force=True")
		return figure

	def _update_figure_size(self, figure):
		if not self.canvas_frame.is_placed():
			raise type(self).ElementNotPlacedError("solve layout placement "
				"using place_all_frames() first")
		cwidth, cheight = self.canvas_frame.get_size()
		figure.set_size_inches(self.left_margin + cwidth + self.right_margin,
			self.bottom_margin + cheight, self.top_margin)
		return

	def create_figure_layout(self, figure = None, *, force = False) -> dict:
		"""
		create the layout on <figure>; if figure = None, a new figure will be
		created from matplotlib.figure.Figure class.
		"""
		ret = dict()
		figure = self._get_figure(figure, force)
		ret["figure"] = figure
		# place all elements
		self.place_all_frames()
		# resize figure
		self._update_figure_size(figure)
		# create all frames
		for frame in self.iter_frames():
			artist = frame.create_artist(figure,
				left_margin = self.left_margin,
				right_margin = self.right_margin,
				bottom_margin = self.bottom_margin,
				top_margin = self.top_margin,
			)
			ret[frame.name] = artist
		return ret

	def place_all_frames(self):
		"""
		solve placement of all frames added to this LayoutCreator
		"""
		# this is not the interfaced required by PlaceableElementBase
		# need to set the reference point first
		self.canvas_frame.clear_placement()
		# set origin at position (0, 0)
		self.canvas_frame.set_placement(self.origin, (0.0, 0.0))
		# clear all frame's existing placements, then re-solve, finally verify
		for frame in self.iter_frames():
			frame.clear_placement() # clear existing placement
		for frame in self.iter_frames():
			frame.solve_placement_resursive()
		for frame in self.iter_frames():
			frame.verify_placement()
		# figure out if the canvas frame rulers are placed; if not, solve by
		# figuring out the canvas width/height
		self._solve_canvas_ruler_placement("width")
		self._solve_canvas_ruler_placement("height")
		return

	@staticmethod
	def _calc_inclusive_rlen(pin, inc_min, inc_max) -> float:
		"""
		calculate the rlen needed to have all frames clip-on. this method is
		intended for internal use only.

		ARGUMENT
		--------
		ref_rpos:
			the RulerPin that used as reference (i.e. origin), must have figure
			figure coord 0.
		inc_min, inc_max:
			the min/max figure coord needed to be included, i.e. if place a
			ruler with the resolved length at the <pin> with respect to its
			ruler position, pmin_pin position will be no higher than inc_min,
			and pmax_pin position will be no lower than inc_max. the only
			exception that may violate this rule is if the ref <pin> is pmin_pin
			or pmax_pin,
		"""
		assert pin.get_placement() == 0 # we want exact 0, not something close
		rpos = pin.ruler_pos
		# calculate from the inc_max->pin
		rlen_u = 0 if rpos == 1.0 else inc_max / (1.0 - rpos)
		# calculate from the pin->inc_min
		rlen_l = 0 if rpos == 0.0 else (-inc_min) / rpos
		return max(rlen_u, rlen_l, 0)

	def _solve_canvas_ruler_placement(self, dim):
		if dim == "width":
			ruler, idx = self.canvas_frame._h_ruler, 0
		elif dim == "height":
			ruler, idx = self.canvas_frame._v_ruler, 1
		else:
			raise ValueError("dim must be 'width' or 'height', got '%s'" % dim)
		#
		if ruler.is_placed():
			# nothing need to be done if is already placed
			pass
		elif ruler.is_placeable():
			# placement is manually set/overridden
			ruler.solve_placement_resursive()
		else:
			# now we need to calculate the ruler length
			rlen = self._calc_inclusive_rlen(
				self.canvas_frame.get_ruler_pins_by_anchor_name(self.origin)[idx],
				min(map(lambda f: f.get_extent()[idx * 2], self.iter_frames())),
				max(map(lambda f: f.get_extent()[idx * 2 + 1], self.iter_frames())),
			)
			if rlen == 0:
				warnings.warn("canvas %s cannot be determined automatically. "
					"try manually set using set_canvas_%s() or "
					"set_canvas_size() instead" % (dim, dim))
			# if reached here, the ruler length must never been set, we can set
			# it first, run the placemenet then clear the ruler length
			ruler.set_ruler_length(rlen)
			ruler.solve_placement_resursive()
			ruler.clear_ruler_length()
		# finally verify placement
		ruler.verify_placement()
		return

	############################################################################
	# children frame management methods
	def add_frame(self, name, *ka, frame_class = LayoutAxesFrame, **kw):
		"""
		factory function of layout frames, create and return a new frame with
		type <frame_class>.

		ARGUMENTS
		---------
		name (str):
			the name of created frame (required)
		frame_class (LayoutElementFrameBase subclass):
			the class of frame that will be created, LayoutElementFrameBase by
			default
		"""
		if name in self.__frames:
			raise ValueError("frame '%s' already exists, use another name"\
				% name)
		if name == "figure":
			raise ValueError("name 'figure' is preserved, use another name")
		if not issubclass(frame_class, LayoutElementFrameBase):
			raise TypeError("frame_class must be a subclass of "
				"LayoutElementFrameBase, not '%s'" % frame_class.__name__)
		frame = frame_class(name, *ka, parent = self.canvas_frame, **kw)
		self.__frames[name] = frame
		return frame

	def get_frame(self, name):
		"""
		get the frame with <name>
		"""
		return self.__frames.get(name, None)

	def iter_frames(self):
		"""
		return the iterator of currently added frames
		"""
		return self.__frames.values()

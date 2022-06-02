#!/usr/bin/env python3

import abc


class ElementBase(object):
	def __init__(self, name: str, *ka, parent = None, **kw):
		super().__init__(*ka, **kw)
		self.set_name(name)
		self.set_parent(parent)
		return

	############################################################################
	# parent
	@property
	def parent(self):
		return self._parent
	def set_parent(self, parent):
		if not (isinstance(parent, ElementBase) or parent is None):
			raise TypeError("parent must be instance of ElementBase or None, "
				"not '%s'" % type(parent).__name__)
		self._parent = parent
		return

	############################################################################
	# element name management and global name which recurse into the parent(s)
	@property
	def name(self):
		return self._name
	def set_name(self, name):
		if not isinstance(name, str):
			raise TypeError("name must be instance of str, not '%s'"\
				% type(name).__name__)
		if "/" in name:
			raise ValueError("character '/' not allowed in name string")
		self._name = name
		return

	@property
	def global_name(self) -> str:
		"""
		whenever parent is not None, this method will recursively trace back the
		parents and concatenate all their names as a slash ("/")-separated name.
		this could provide unique names to identify different elements under
		different parents' possessions.
		"""
		ret = self.name
		if self.parent is not None:
			ret = self.parent.global_name + "/" + ret
		return ret

	############################################################################
	# unique-id-based methods and properties
	@property
	def uid(self) -> int:
		"""
		alias to id(self). uid is used by '==' operator and hash calls.
		"""
		return id(self)

	def __eq__(self, other) -> bool:
		ret = (self.uid == other.uid) if isinstance(other, ElementBase)\
			else NotImplemented
		return ret

	def __hash__(self) -> int:
		return self.uid


class PlaceableElementBase(abc.ABC, ElementBase):
	"""
	PlaceableElementBase is the base class for all elements that can be put on
	the figure canvas. the interface defined in this class is used to solve the
	interaction between different elements, especially the dependency chain in
	solving the placement of elements.
	"""
	class PlacementError(RuntimeError): pass
	class CircularDependencyError(PlacementError): pass
	class DependencyUnplacedError(PlacementError): pass
	class IncomplyingPlacementError(PlacementError): pass
	class PlacementUnsolvableError(PlacementError): pass

	@abc.abstractmethod
	def is_placeable(self) -> bool:
		"""
		return True if sufficient information is set so that placement of the
		calling element can potentially be solved. note this method does not
		recurse into the dependencies, nor imply the dependencies can be solved
		as well.
		"""
		return

	@abc.abstractmethod
	def is_placed(self) -> bool:
		"""
		return True if the placement has already been solved with this element
		"""
		return

	@abc.abstractmethod
	def get_placement(self, *ka, **kw):
		"""
		get current element placement info; for complex or high-level element,
		the return value may be related to the elements it possesses

		calling signature may vary based on subclass implementations
		"""
		return

	@abc.abstractmethod
	def set_placement(self, *ka, **kw):
		"""
		set current element placement info; if such method is ambiguous for an
		element type, should raise NotImplementedError; for such element types,
		solve_placement() should be the only valid way to find the placements

		calling signature may vary based on subclass implementations
		"""
		return

	@abc.abstractmethod
	def clear_placement(self, *ka, **kw):
		"""
		clear placement to not-placed status

		calling signature may vary based on subclass implementations
		"""
		return

	@abc.abstractmethod
	def get_dependencies(self) -> "iterable":
		"""
		return an iterable object containing all dependencies directly needed to
		solve the calling element
		"""
		return

	@abc.abstractmethod
	def solve_placement(self):
		"""
		solve the placement of the calling element without recursing into the
		dependencies. this function is intended to be called internally by
		solve_placement_resursive() and is assumed to be called after all direct
		dependencies have been placed (solved). it should raise a
		PlacementUnsolvableError if failed to solve the placement. calling this
		method can assume self.is_placed() == False, but even if
		self.is_placed() == True should not raise any error.
		"""
		return

	def solve_placement_resursive(self, *, dep_path: list = None):
		"""
		solve the placement of calling element and its dependencies recursively.
		dep_path a list that tracks the dependencies to detect any circular
		dependencies, and raise a CircularDependencyError if needed.
		"""
		if dep_path is None:
			dep_path = list()
		# first check circular dependency
		if self in dep_path:
			raise type(self).CircularDependencyError("circular dependency "
				"found: " + (("=>").join(map(lambda x: x.global_name,
					dep_path + [self]))))
		# then check if self is already placed; if yes, no need to do anything
		if not self.is_placed():
			dep_path.append(self)
			# solve dependencies first
			for dep in self.get_dependencies():
				dep.solve_placement_resursive(dep_path = dep_path)
			# here we check self.is_placed() again as in some cases dependencies
			# may help solve self placement
			if not self.is_placed():
				# then solve locally
				self.solve_placement()
			# reaching here means self has been solved
			# now to remove self from the dep_path
			dep_path.pop()
		return

	@abc.abstractmethod
	def verify_placement(self):
		"""
		check if the placement is complying with a placement rule or relation,
		if set. raise IncomplyingPlacementError if incompliance is found.
		"""
		return

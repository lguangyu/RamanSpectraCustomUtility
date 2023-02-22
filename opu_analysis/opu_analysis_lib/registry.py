#!/usr/bin/env python3

_REGISTRY_STUB = dict()


def new(*ka, reg_type=None, **kw):
	if reg_type is None:
		reg_type = Registry
	elif not issubclass(reg_type, Registry):
		raise TypeError("reg_type must be (subclass of) Registry")
	new_reg = reg_type(*ka, **kw)
	key = new_reg.registry_name
	if key in _REGISTRY_STUB:
		raise ValueError("registry name '%s' already exists" % key)
	_REGISTRY_STUB[key] = new_reg
	return new_reg


def get(registry_name):
	return _REGISTRY_STUB[registry_name]


class Registry(dict):
	def __init__(self, *ka, registry_name: str, value_type=object):
		if (not isinstance(value_type, object)):
			raise TypeError("value_type must be a class type, not '%s'"
				% type(value_type).__name__)
		super().__init__(*ka)
		self.registry_name = registry_name
		self.value_type = value_type
		self.default_key = None
		return

	@property
	def default_key(self):
		return self._default_key

	@default_key.setter
	def default_key(self, key: str = None):
		if ((key is not None) and (not isinstance(key, str))):
			raise TypeError("key must be str, not '%s'" % type(key).__name__)
		self._default_key = key
		return

	def register(self, key, *, as_default=False):
		if key in self:
			raise ValueError("key '%s' already exists" % key)
		if as_default or not len(self):
			self.default_key = key

		def decorator(obj):
			if not issubclass(obj, self.value_type):
				raise TypeError("decorated object must be subclass of '%s'"
					% self.value_type.__name__)
			self[key] = obj
			return obj
		return decorator

	def list_keys(self):
		return sorted(self.keys())

	def get(self, key, *ka, **kw):
		cls = self[key]
		return cls(*ka, **kw)

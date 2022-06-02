#!/usr/bin/env python3

import abc
import json
# custom lib
from . import util


class JSONSerializable(object):
	@abc.abstractmethod
	def json_serialize(self):
		pass


class GeneralJSONDecoder(json.JSONDecoder):
	pass


class GeneralJSONEncoder(json.JSONEncoder):
	def default(self, o):
		if isinstance(o, JSONSerializable):
			ret = o.json_serialize()
		else:
			ret = super().default(o)
		return ret


def load_json(f, *ka, **kw):
	with util.get_fp(f, "r") as fp:
		ret = json.load(fp, *ka, cls = GeneralJSONDecoder, **kw)
	return ret


def dump_json(o, f, *ka, **kw):
	with util.get_fp(f, "w") as fp:
		json.dump(o, fp, *ka, cls = GeneralJSONEncoder, **kw)
	return

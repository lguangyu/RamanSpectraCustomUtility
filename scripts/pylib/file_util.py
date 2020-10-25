#!/usr/bin/env python3

import io


def get_fp(file, mode = "r", *ka, factory = open, **kw):
	if isinstance(file, io.IOBase):
		if file.mode != mode:
			raise ValueError("expected mode '%s', got '%s'"\
				% (mode, file.mode))
		else:
			return file
	elif isinstance(file, str):
		return open(file, mode, *ka, **kw)
	else:
		raise TypeError("file must be io.IOBase or str, not '%s'"\
			% type(file).__name__)
	return

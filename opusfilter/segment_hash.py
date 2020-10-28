"""Hashing parallel segments"""

import pyhash
import regex


from . import ConfigurationError


class SegmentHasher:
    """Hasher one or more text segments"""

    re_not_letter = regex.compile(r'[^\p{L}]')

    def __init__(self, compare='all', hash='xx_64', hashseed=0, lowercase=False, letters_only=False):
        self.compare = None
        if compare != 'all':
            if not isinstance(compare, list) or not all(isinstance(x, int) for x in compare):
                raise ConfigurationError(
                    "The compare parameter for hashing has to be 'all' or "
                    "a list of input file indices")
            self.compare = sorted(compare)
        hashname = hash
        if hashname and not hasattr(pyhash, hashname):
            raise ConfigurationError(
                "Algorithm '{}' not available from from pyhash".format(hashname))
        self.hashfunc = getattr(pyhash, hashname)(seed=hashseed) if hashname else lambda x: x
        self.lowercase = lowercase

    def apply(self, lines):
        if self.compare is None:
            inputstr = ''.join(lines)
        else:
            try:
                inputstr = ''.join(lines[idx] for idx in self.compare)
            except KeyError:
                raise ConfigurationError(
                    "The input indices {} in the compare parameter do not match input of lenght {}",
                    self.compare, len(lines))
        return self.hashfunc(inputstr)

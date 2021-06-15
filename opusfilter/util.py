"""Utility functions"""

import bz2
import gzip
import io
import logging
import lzma
import os

import requests
from tqdm import tqdm
import ruamel.yaml


logger = logging.getLogger(__name__)


def file_open(filename, mode='r', encoding='utf8'):
    """Open file with implicit gzip/bz2 support

    Uses text mode by default regardless of the compression.

    In write mode, creates the output directory if it does not exist.

    """
    if 'w' in mode and not os.path.isdir(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    if filename.endswith('.bz2'):
        if mode in {'r', 'w', 'x', 'a'}:
            mode += 't'
        return bz2.open(filename, mode=mode, encoding=encoding)
    if filename.endswith('.xz'):
        if mode in {'r', 'w', 'x', 'a'}:
            mode += 't'
        return lzma.open(filename, mode=mode, encoding=encoding)
    if filename.endswith('.gz'):
        if mode in {'r', 'w', 'x', 'a'}:
            mode += 't'
        return gzip.open(filename, mode=mode, encoding=encoding)
    return open(filename, mode=mode, encoding=encoding)


def file_download(url, localfile=None, chunk_size=None):
    """Download file from URL to a local file"""
    if localfile is None:
        localfile = url.split('/')[-1]
    if chunk_size is None:
        chunk_size = 1024 * 1024
    with requests.get(url, stream=True) as req:
        req.raise_for_status()
        try:
            total_length = int(req.headers.get('content-length'))
        except (TypeError, ValueError):
            total_length = None
        pbar = tqdm(miniters=1, total=total_length, unit="B", unit_scale=True,
                    unit_divisor=1024, desc=localfile)
        with open(localfile, 'wb') as fobj:
            for chunk in req.iter_content(chunk_size=chunk_size):
                pbar.update(len(chunk))
                fobj.write(chunk)
                fobj.flush()
    return localfile


yaml = ruamel.yaml.YAML()


def yaml_dumps(obj):
    """Return a string containing YAML output from input object"""
    with io.StringIO() as iostream:
        yaml.dump(obj, iostream)
        iostream.seek(0)
        return iostream.read()

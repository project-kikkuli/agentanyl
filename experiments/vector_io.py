"""Restricted reader for the released vector-only PyTorch ZIP checkpoints."""
import io
import pickle
import zipfile
from collections import OrderedDict

import numpy as np


class _Int64DType:
    """Pickle placeholder accepting only NumPy's little-endian int64 dtype."""
    def __setstate__(self, state):
        expected = (3, '<', None, None, None, -1, -1, 0)
        if state != expected:
            raise ValueError('unsupported NumPy dtype metadata')
        self.byteorder = '<'


def _numpy_dtype(name, align, copy):
    if name != 'i8' or align is not False or copy is not True:
        raise ValueError('unsupported NumPy dtype')
    return _Int64DType()


def _numpy_scalar(dtype, data):
    if not isinstance(dtype, _Int64DType) or not isinstance(data, bytes) or len(data) != 8:
        raise ValueError('unsupported NumPy scalar')
    return int.from_bytes(data, 'little', signed=True)


def _latin1_encode(value, encoding):
    if not isinstance(value, str) or encoding != 'latin1' or len(value) > 16:
        raise ValueError('unsupported pickle byte encoding')
    return value.encode('latin1')


def read_vectors(path):
    """Read tensor storages and the released checkpoint's int64 layer field.

    Pickle globals are mapped to local, inert handlers. No module imports or
    general-purpose object reconstruction are permitted.
    """
    def rebuild(storage, offset, shape, stride, requires_grad, hooks):
        if requires_grad or len(shape) != 1 or stride != (1,):
            raise ValueError('only contiguous one-dimensional vectors are accepted')
        if offset < 0 or offset + shape[0] > len(storage):
            raise ValueError('tensor outside storage')
        return storage[offset:offset + shape[0]].copy()

    with zipfile.ZipFile(path) as archive:
        metadata = [x for x in archive.namelist() if x.endswith('/data.pkl')]
        if len(metadata) != 1:
            raise ValueError('ambiguous tensor archive')
        prefix = metadata[0][:-len('data.pkl')]
        if archive.read(prefix + 'byteorder') != b'little':
            raise ValueError('unsupported byte order')

        class Reader(pickle.Unpickler):
            def find_class(self, module, name):
                allowed = {
                    ('torch._utils', '_rebuild_tensor_v2'): rebuild,
                    ('torch', 'FloatStorage'): 'FloatStorage',
                    ('collections', 'OrderedDict'): OrderedDict,
                    ('numpy.core.multiarray', 'scalar'): _numpy_scalar,
                    ('numpy._core.multiarray', 'scalar'): _numpy_scalar,
                    ('numpy', 'dtype'): _numpy_dtype,
                    ('_codecs', 'encode'): _latin1_encode,
                }
                if (module, name) not in allowed:
                    raise ValueError(f'unexpected pickle global {module}.{name}')
                return allowed[module, name]

            def persistent_load(self, value):
                if not isinstance(value, tuple) or len(value) != 5:
                    raise ValueError('unsupported storage reference')
                kind, storage_type, key, location, size = value
                if kind != 'storage' or storage_type != 'FloatStorage' or not str(key).isdigit() or location != 'cpu':
                    raise ValueError('unsupported storage')
                if not isinstance(size, int) or size < 0 or size > 100000:
                    raise ValueError('invalid storage size')
                data = archive.read(prefix + 'data/' + key)
                if len(data) != size * 4:
                    raise ValueError('invalid storage size')
                return np.frombuffer(data, dtype='<f4')

        result = Reader(io.BytesIO(archive.read(metadata[0]))).load()
    if not isinstance(result, dict):
        raise ValueError('checkpoint must contain a mapping')
    for name, value in result.items():
        if isinstance(value, np.ndarray) and (value.shape != (3584,) or not np.all(np.isfinite(value))):
            raise ValueError(f'invalid vector {name}')
    return result

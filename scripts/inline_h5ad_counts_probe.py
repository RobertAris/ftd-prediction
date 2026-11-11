import os
import sys
sys.excepthook = lambda kind, value, trace: print(dict(probe_failed=True))
import h5py
import numpy as np
assert int(os.environ.get(next(iter(dict(LSB_JOBID=0))), 0)) > 0
files = [h5py.File(path) for path in sys.argv[1:]]
assert all(isinstance(file.get(field, getlink=True), h5py.HardLink) for file in files for field in dict(X=0, raw=0) if field in file)
assert all(isinstance(file[outer].get(inner, getlink=True), h5py.HardLink) for file in files for outer in dict(raw=0) for inner in dict(X=0) if outer in file and inner in file[outer])
paths = list(dict(X=0)) + [chr(47).join(dict(raw=0, X=0))]
matrices = [(number, path, file[path]) for number, file in enumerate(files, 1) for path in paths if path in file]
assert all(isinstance(node.get(field, getlink=True), h5py.HardLink) for number, path, node in matrices for field in dict(data=0) if isinstance(node, h5py.Group))
datasets = [(number, path, node[field]) for number, path, node in matrices for field in dict(data=0) if isinstance(node, h5py.Group)] + [(number, path, node) for number, path, node in matrices if isinstance(node, h5py.Dataset)]
assert len(datasets) > 0 and all(node.ndim in (1, 2) for number, path, node in datasets)
assert all(not node.external and not node.is_virtual and node.dtype.kind in dict(i=0, u=0, f=0) for number, path, node in datasets)
samples = [(number, path, node[:100000]) for number, path, node in datasets if node.ndim == 1] + [(number, path, node[:1, :100000].ravel()) for number, path, node in datasets if node.ndim == 2]
[print(dict(input_number=number, matrix=path, sampled_values=int(values.size), sample_informative=bool(values.size), sample_finite=bool(np.isfinite(values).all()), sample_nonnegative=bool((values >= 0).all()), sample_integer=bool(np.equal(values, np.floor(values)).all()))) for number, path, values in samples]
[file.close() for file in files]

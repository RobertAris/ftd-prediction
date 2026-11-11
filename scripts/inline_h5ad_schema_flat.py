import sys
import h5py
print(sys.version)
print(h5py.version.info)
files = [h5py.File(path) for path in sys.argv[1:]]
assert all(isinstance(file.get(field, getlink=True), h5py.HardLink) for file in files for field in dict(X=0, layers=0, obs=0, var=0, obsm=0, raw=0) if field in file)
[print(dict(input_number=number, field=field, fields=list(file[field].keys()))) for number, file in enumerate(files, 1) for field in dict(X=0, layers=0, obs=0, var=0, obsm=0, raw=0) if field in file and isinstance(file[field], h5py.Group)]
[print(dict(input_number=number, field=field, shape=file[field].shape, dtype=str(file[field].dtype))) for number, file in enumerate(files, 1) for field in dict(X=0) if field in file and isinstance(file[field], h5py.Dataset)]
[print(dict(input_number=number, field=field, shape=file[field].attrs.get(next(iter(dict(shape=0)))))) for number, file in enumerate(files, 1) for field in dict(X=0) if field in file and isinstance(file[field], h5py.Group)]
[file.close() for file in files]

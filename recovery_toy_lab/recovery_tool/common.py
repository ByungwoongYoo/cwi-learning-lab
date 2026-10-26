"""Small, closed-format JSON/NPZ persistence; no pickle deserialization."""
from __future__ import annotations
import hashlib, io, json, os, tempfile, zipfile
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
MAX_ARCHIVE = 4 * 1024 * 1024
class FormatError(ValueError): pass

def digest(raw): return hashlib.sha256(raw).hexdigest()
def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def json_write(path,value): Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
def _pairs(pairs):
    d={}
    for k,v in pairs:
        if k in d: raise FormatError('Duplicate JSON key')
        d[k]=v
    return d

def json_read(path,limit=131072):
    p=Path(path)
    if p.stat().st_size > limit: raise FormatError('JSON exceeds size bound')
    try:
        return json.loads(p.read_text(),object_pairs_hook=_pairs,parse_constant=lambda x: (_ for _ in ()).throw(FormatError('Nonfinite JSON')))
    except (ValueError,UnicodeError,OSError) as e: raise FormatError('Invalid JSON') from e

def exact_keys(d,keys):
    if type(d) is not dict or set(d)!=set(keys): raise FormatError('Unexpected schema keys')

def checked_npz(path,sha,specs):
    p=Path(path)
    if p.stat().st_size>MAX_ARCHIVE: raise FormatError('Archive exceeds size bound')
    raw=p.read_bytes()
    if digest(raw)!=sha: raise FormatError('Archive SHA256 mismatch')
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            infos=z.infolist()
            if len(infos)!=len(specs) or len(infos)>32: raise FormatError('Archive member count mismatch')
            if {i.filename for i in infos}!={k+'.npy' for k in specs}: raise FormatError('Unexpected archive member')
            if len({i.filename for i in infos})!=len(infos): raise FormatError('Duplicate archive member')
            if sum(i.file_size for i in infos)>MAX_ARCHIVE: raise FormatError('Decoded archive too large')
            for info in infos:
                if info.flag_bits&1 or info.file_size>MAX_ARCHIVE: raise FormatError('Unsupported member')
                with z.open(info) as f:
                    ver=np.lib.format.read_magic(f)
                    if ver not in ((1,0),(2,0)): raise FormatError('Unsupported NPY version')
                    reader=np.lib.format.read_array_header_1_0 if ver==(1,0) else np.lib.format.read_array_header_2_0
                    shape,fortran,dtype=reader(f,max_header_size=4096)
                    expected_shape,expected_dtype=specs[info.filename[:-4]]
                    if tuple(shape)!=tuple(expected_shape) or dtype!=np.dtype(expected_dtype) or fortran or dtype.hasobject:
                        raise FormatError('Array shape/dtype mismatch')
        with np.load(io.BytesIO(raw),allow_pickle=False,max_header_size=4096) as data:
            arrays={k:data[k].copy() for k in specs}
        if any(a.dtype.kind=='f' and not np.isfinite(a).all() for a in arrays.values()): raise FormatError('Nonfinite array')
        return arrays
    except FormatError: raise
    except (OSError,ValueError,KeyError,EOFError,zipfile.BadZipFile) as e: raise FormatError('Invalid NPZ archive') from e

CONFIG = {'schema_version':1,'architecture':[7,16,1],'threads':1,'model_seed':1701,'sampler_seed':1702,'sample_seed':20261026,'epochs':4,'batch_size':256,'learning_rate':0.03,'momentum':0.9,'weight_decay':0.0,'score_threshold':0.5,'dataset':'synthetic-balanced-v1'}
def frozen_config(): return dict(CONFIG),digest(canonical(CONFIG))

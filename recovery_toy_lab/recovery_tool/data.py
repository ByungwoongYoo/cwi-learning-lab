"""Fixed synthetic benchmark: no downloaded images or external datasets."""
from pathlib import Path
import numpy as np
from .common import frozen_config,canonical,digest,json_read,json_write,checked_npz,exact_keys,FormatError
SPECS={'x':((8192,7),'float64'),'y':((8192,1),'float64'),'indices':((8192,),'int64')}
def generated():
    c,_=frozen_config();rng=np.random.default_rng(c['sample_seed'])
    x=rng.standard_normal((8192,7));z=x[:,0]+.5*x[:,1]-.25*x[:,2]
    order=np.argsort(z,kind='stable');y=np.zeros((8192,1),dtype=np.float64);y[order[4096:]]=1
    return {'x':x,'y':y,'indices':np.arange(8192,dtype=np.int64)}
def expected_meta(a,archive):
    _,ch=frozen_config()
    return {'schema_version':1,'config_sha256':ch,'archive_sha256':archive,'sample_indices_sha256':digest(a['indices'].tobytes()),'generator':'numpy-pcg64-normal-linear-median-v1','rows':8192,'positive':4096,'negative':4096,'scope':'SYNTHETIC_TRAIN_ONLY'}
def prepare(folder):
    a=generated();folder=Path(folder);folder.mkdir(exist_ok=False)
    np.savez(folder/'data.npz',**a);m=expected_meta(a,digest((folder/'data.npz').read_bytes()));json_write(folder/'manifest.json',m);return m
def load_data(folder):
    folder=Path(folder);m=json_read(folder/'manifest.json');e=generated()
    exact_keys(m,expected_meta(e,''))
    a=checked_npz(folder/'data.npz',m['archive_sha256'],SPECS)
    if canonical(m)!=canonical(expected_meta(e,m['archive_sha256'])) or any(not np.array_equal(a[k],e[k]) for k in e): raise FormatError('Synthetic generator/data mismatch')
    return a,m

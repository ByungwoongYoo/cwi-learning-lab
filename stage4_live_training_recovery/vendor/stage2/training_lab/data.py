"""Read only eight fixed public TRAIN TIFFs; sample once, fit scaler on sample."""
import io, json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter
from .common import ROOT, FormatError, digest, json_read, json_write, checked_npz, frozen_config, exact_keys
DATA_SPECS={'x':((8192,7),'float64'),'y':((8192,1),'float64'),'mean':((7,),'float64'),'scale':((7,),'float64'),'indices':((8192,2),'int64')}

def source_hashes():
    folder=ROOT/'examples/public_train'
    receipt=json_read(folder/'fetch_receipt.json')
    names=[f'original_{o:03d}_{e}{s}.tiff' for o in (4,6) for e in ('40kV','90kV') for s in ('','_mask')]
    if {p.name for p in folder.glob('*.tiff')}!=set(names): raise FormatError('Unexpected source TIFF scope')
    hashes={}
    for name in names:
        p=folder/name
        if p.stat().st_size>8*1024*1024: raise FormatError('Source exceeds byte bound')
        sha=digest(p.read_bytes())
        if sha!=receipt['members'][name]['sha256']: raise FormatError('Source hash mismatch')
        hashes[name]=sha
    freeze=json_read(ROOT/'docs/config_freeze.json')
    if hashes!=freeze['source_sha256']: raise FormatError('Frozen source mismatch')
    return hashes

def _image(path,mask=False):
    # Only called after fixed allowlist SHA verification, never accepts uploads/URLs.
    with Image.open(path) as im:
        if im.size!=(956,760) or im.n_frames!=(2 if mask else 1): raise FormatError('Source TIFF schema')
        if mask: im.seek(1)
        a=np.array(im)
    if mask:
        if a.dtype!=np.uint8 or not np.isin(a,[0,1]).all(): raise FormatError('Mask values')
    elif a.dtype!=np.float32 or not np.isfinite(a).all(): raise FormatError('Image values')
    return a

def prepare(out):
    c,ch=frozen_config(); hashes=source_hashes();rng=np.random.default_rng(c['sample_seed'])
    xx=[];yy=[];indices=[];counts={}
    for oid in c['original_ids']:
        folder=ROOT/'examples/public_train'
        low=_image(folder/f'original_{oid:03d}_40kV.tiff');high=_image(folder/f'original_{oid:03d}_90kV.tiff')
        gt=_image(folder/f'original_{oid:03d}_40kV_mask.tiff',True)
        gt90=_image(folder/f'original_{oid:03d}_90kV_mask.tiff',True)
        if not np.array_equal(gt,gt90): raise FormatError('Paired labels disagree')
        idx=np.sort(rng.choice(low.size,c['pixels_per_example'],replace=False))
        # Reused independently authored Stage1 feature formula; source author code not used.
        planes=[low,low-uniform_filter(low,9,mode='reflect'),low-uniform_filter(low,31,mode='reflect'),high,high-uniform_filter(high,9,mode='reflect'),high-uniform_filter(high,31,mode='reflect'),low-high]
        x=np.stack([p.ravel()[idx] for p in planes],axis=1).astype(np.float64)
        y=gt.ravel()[idx].astype(np.float64)[:,None]
        xx.append(x);yy.append(y);indices.append(np.column_stack([np.full(len(idx),oid,dtype=np.int64),idx]))
        counts[str(oid)]={'samples':len(idx),'positive_labels':int(y.sum()),'partition':'train'}
    raw=np.concatenate(xx); y=np.concatenate(yy);mean=raw.mean(0);scale=raw.std(0);scale=np.maximum(scale,1e-12)
    arrays={'x':(raw-mean)/scale,'y':y,'mean':mean,'scale':scale,'indices':np.concatenate(indices)}
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    np.savez(out/'data.npz',**arrays)
    m={'schema_version':1,'config_sha256':ch,'source_sha256':hashes,'archive_sha256':digest((out/'data.npz').read_bytes()),'sample_indices_sha256':digest(arrays['indices'].tobytes()),'counts':counts,'status':'TRAIN_RESUBSTITUTION_ONLY_NOT_GENERALIZATION'}
    json_write(out/'manifest.json',m)
    return m

def load_data(folder):
    c,ch=frozen_config();m=json_read(Path(folder)/'manifest.json')
    exact_keys(m,['schema_version','config_sha256','source_sha256','archive_sha256','sample_indices_sha256','counts','status'])
    if type(m['schema_version']) is not int or m['schema_version']!=1 or m['config_sha256']!=ch or m['source_sha256']!=source_hashes(): raise FormatError('Dataset binding mismatch')
    a=checked_npz(Path(folder)/'data.npz',m['archive_sha256'],DATA_SPECS)
    if digest(a['indices'].tobytes())!=m['sample_indices_sha256'] or not np.isin(a['y'],[0,1]).all() or (a['scale']<=0).any(): raise FormatError('Dataset values mismatch')
    expected=np.random.default_rng(c['sample_seed'])
    for j,oid in enumerate(c['original_ids']):
        rows=a['indices'][j*4096:(j+1)*4096]
        idx=np.sort(expected.choice(956*760,4096,replace=False))
        if not (rows[:,0]==oid).all() or not np.array_equal(rows[:,1],idx): raise FormatError('Sample indices mismatch')
    expected_counts={str(oid):{'samples':4096,'positive_labels':int(a['y'][j*4096:(j+1)*4096].sum()),'partition':'train'} for j,oid in enumerate(c['original_ids'])}
    from .common import canonical
    if canonical(m['counts'])!=canonical(expected_counts) or m['status']!='TRAIN_RESUBSTITUTION_ONLY_NOT_GENERALIZATION': raise FormatError('Dataset count/status mismatch')
    return a,m

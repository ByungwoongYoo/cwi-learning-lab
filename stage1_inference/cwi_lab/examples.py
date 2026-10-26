"""Deterministic synthetic inputs and read-only bundled TRAIN examples."""
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image
from .inference import read_tiff,InputError
ROOT=Path(__file__).resolve().parents[1]
CHOICES=['Synthetic • pipeline check, not an X-ray','CWI TRAIN • normal • Original 006','CWI TRAIN • bone • Original 004','Upload your own public/authorized TIFF pair']

def synthetic():
    yy,xx=np.mgrid[:256,:320];body=np.exp(-(((xx-160)/100)**2+((yy-128)/65)**2)*2)
    rng=np.random.default_rng(20261026)
    low=(0.6*body+rng.normal(0,.004,body.shape)-.03).astype(np.float32)
    high=(.31*body+rng.normal(0,.002,body.shape)-.015).astype(np.float32)
    return low,high,None,{'source':'SYNTHETIC_NOT_REAL_XRAY','known_normal':False,'partition':'none','note':'Artificial arrays only test software wiring. They prove no detection performance.'}

def load_example(choice):
    if choice==CHOICES[0]:return synthetic()
    oid={CHOICES[1]:6,CHOICES[2]:4}.get(choice)
    if oid is None:raise InputError('Select a built-in example or provide TIFF uploads.')
    folder=ROOT/'examples/public_train';receipt_path=folder/'fetch_receipt.json'
    if not receipt_path.exists():raise InputError('Public TRAIN examples are not bundled; use synthetic input.')
    receipt=json.loads(receipt_path.read_text())
    def checked(name):
        p=folder/name
        if not p.is_file():raise InputError('This public training example is unavailable; use synthetic input.')
        raw=p.read_bytes();entry=receipt['members'].get(name,{})
        if hashlib.sha256(raw).hexdigest()!=entry.get('sha256'):raise InputError('Bundled example checksum mismatch.')
        return raw
    low=read_tiff(checked(f'original_{oid:03d}_40kV.tiff'))
    high=read_tiff(checked(f'original_{oid:03d}_90kV.tiff'))
    # Trusted fixed allowlist file only; user masks are never opened here.
    import io
    with Image.open(io.BytesIO(checked(f'original_{oid:03d}_40kV_mask.tiff'))) as im:
        if im.n_frames!=2 or im.size!=(956,760):raise InputError('Invalid bundled review mask.')
        im.seek(1);gt=np.array(im)
    if gt.dtype!=np.uint8 or not np.isin(gt,[0,1]).all():raise InputError('Invalid bundled review mask values.')
    return low,high,gt.astype(bool),{'source':'CWI_PUBLIC_CC_BY_4_TRAIN','Original_ID':oid,'known_normal':oid==6,'partition':'train','note':'Used in fitting. This is a false-alarm review example, never a held-out performance estimate.'}

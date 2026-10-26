import io
import numpy as np
import pytest
from PIL import Image
from cwi_lab.inference import *
from cwi_lab.examples import synthetic,load_example,CHOICES

def tiff(a):
    b=io.BytesIO();Image.fromarray(a).save(b,format='TIFF');return b.getvalue()

def test_signed_float_round_trip():
    a=np.linspace(-.5,.7,1024,dtype=np.float32).reshape(32,32)
    assert np.array_equal(read_tiff(tiff(a)),a)
    assert read_tiff(tiff(a)).min()<0

@pytest.mark.parametrize('bad',[b'',b'fake',b'X'*(MAX_BYTES+1),'file:///etc/passwd','https://example.com/image.tiff'])
def test_reject_bad_bytes(bad):
    with pytest.raises(InputError):read_tiff(bad)

@pytest.mark.parametrize('a',[np.zeros((16,16),dtype=np.uint8),np.zeros((16,16,3),dtype=np.uint8),np.zeros((8,8),dtype=np.float32),np.full((16,16),np.nan,dtype=np.float32),np.full((16,16),np.inf,dtype=np.float32),np.full((16,16),10001,dtype=np.float32)])
def test_reject_invalid_images(a):
    with pytest.raises(InputError):read_tiff(tiff(a))

def test_multi_page_rejected():
    b=io.BytesIO();im=Image.fromarray(np.zeros((16,16),np.float32));im.save(b,format='TIFF',save_all=True,append_images=[im])
    with pytest.raises(InputError):read_tiff(b.getvalue())

def test_pair_and_model_errors():
    a=np.zeros((16,16),np.float32)
    for kw in [dict(model='dual'),dict(model='not-a-model'),dict(high=np.zeros((16,17),np.float32)),dict(model='low',threshold=float('nan')),dict(model='low',min_area=0)]:
        with pytest.raises(InputError):infer(a,**kw)

def test_feature_parity():
    l,h,_,_=synthetic();x=features(l,h)
    expected=np.stack([l,l-uniform_filter(l,9,mode='reflect'),l-uniform_filter(l,31,mode='reflect'),h,h-uniform_filter(h,9,mode='reflect'),h-uniform_filter(h,31,mode='reflect'),l-h],axis=-1).reshape(-1,7)
    assert np.array_equal(x,expected)
    for name,n in [('low',3),('dual',7)]:
        m=load_models()[name]
        z=((expected[:,:n]-m['mean'])/m['scale'])@np.asarray(m['coef'])+m['intercept']
        assert np.array_equal(infer(l,h,name)['scores'].ravel(),1/(1+np.exp(-np.clip(z,-700,700))))

def test_eight_connected_area_boundary():
    s=np.zeros((32,32));s[1:6,1:6]=.5;s[15:19,15:21]=1
    mask,c=clean_mask(s)
    assert mask.sum()==25 and len(c)==1
    s=np.eye(32)
    assert clean_mask(s)[0].sum()==32

def test_exploratory_and_review_separation():
    l,h,gt,meta=synthetic();before=infer(l,h)
    assert before['summary']['setting_status']=='ORIGINAL_FROZEN_SETTINGS'
    assert infer(l,h,threshold=.7)['summary']['setting_status']=='EXPLORATORY_NOT_VALIDATED'
    r=review(before['mask'],np.zeros_like(l,dtype=bool))
    assert r['FP_pixels']==before['mask'].sum()
    assert np.array_equal(before['mask'],infer(l,h)['mask'])
    assert review(before['mask'])['status']=='NO_GROUND_TRUTH'

def test_model_tamper_rejected(monkeypatch,tmp_path):
    p=tmp_path/'models.json';p.write_text('{}')
    monkeypatch.setattr('cwi_lab.inference.MODEL_PATH',p)
    with pytest.raises(RuntimeError):load_models()

def test_public_train_examples():
    for choice in CHOICES[1:3]:
        l,h,gt,m=load_example(choice)
        assert l.shape==h.shape==gt.shape==(760,956) and m['partition']=='train'
        r=infer(l,h)
        assert r['summary']['whole_image_inference'] and not r['summary']['ground_truth_used_in_inference']

@pytest.mark.parametrize('v',[float('nan'),float('inf'),None,'bad',25.5,True])
def test_min_area_invalid_numeric(v):
    a=np.zeros((16,16),np.float32)
    with pytest.raises(InputError):infer(a,model='low',min_area=v)

@pytest.mark.parametrize('v',[None,'bad',float('inf'),0.,1.])
def test_threshold_invalid_numeric(v):
    a=np.zeros((16,16),np.float32)
    with pytest.raises(InputError):infer(a,model='low',threshold=v)

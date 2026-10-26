"""Frozen JSON linear inference, independent of sklearn and training code."""
from __future__ import annotations
import hashlib
import io
import json
import numbers
from pathlib import Path
import warnings
import numpy as np
from PIL import Image, UnidentifiedImageError
from scipy.ndimage import uniform_filter, label

MAX_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 1_048_576
MAX_DIM = 1536
MODEL_SHA256 = '0b6bce8f46b677c834048994c3ea72cba26e4d6a4172cd88018190ed7cbb9416'
MODEL_PATH = Path(__file__).with_name('models.json')

class InputError(ValueError):
    """A safe, user-facing validation failure."""

def validate_array(a: np.ndarray) -> np.ndarray:
    if not isinstance(a, np.ndarray) or a.ndim != 2:
        raise InputError('A single 2-D grayscale image is required.')
    h, w = a.shape
    if min(h,w) < 16 or max(h,w) > MAX_DIM or h*w > MAX_PIXELS:
        raise InputError('Dimensions must be 16–1536 per side and at most 1,048,576 pixels.')
    if a.dtype != np.float32:
        raise InputError('Only signed float32 TIFF data are accepted; do not convert to 8-bit.')
    if not np.isfinite(a).all() or np.abs(a).max() > 10_000:
        raise InputError('NaN, infinity, or absolute values above 10,000 are not accepted.')
    return a

def read_tiff(data: bytes) -> np.ndarray:
    """Accept bounded bytes, not paths/URLs. Validate before decoding pixels."""
    if not isinstance(data, bytes) or not data or len(data) > MAX_BYTES:
        raise InputError('Provide a TIFF of at most 8 MiB.')
    if data[:4] not in (b'II*\x00', b'MM\x00*'):
        raise InputError('Only classic TIFF files are supported.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as im:
                w,h = im.size
                if im.format != 'TIFF' or min(w,h)<16 or max(w,h)>MAX_DIM or w*h>MAX_PIXELS:
                    raise InputError('Unsupported TIFF dimensions or format.')
                # Do not enumerate arbitrary multipage files; only check page 1 exists.
                try:
                    im.seek(1)
                except EOFError:
                    im.seek(0)
                else:
                    raise InputError('Multi-page TIFF is not accepted for image inputs.')
                if im.mode != 'F':
                    raise InputError('Expected float32 TIFF (Pillow mode F), not an 8-bit preview.')
                a = np.array(im, dtype=np.float32)
    except InputError:
        raise
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise InputError('Unreadable or unsupported TIFF.') from exc
    return validate_array(a)

def features(low: np.ndarray, high: np.ndarray | None = None) -> np.ndarray:
    validate_array(low)
    a = [low, low-uniform_filter(low,9,mode='reflect'), low-uniform_filter(low,31,mode='reflect')]
    if high is not None:
        validate_array(high)
        if low.shape != high.shape:
            raise InputError('40 kV and 90 kV images must have identical dimensions.')
        a += [high, high-uniform_filter(high,9,mode='reflect'), high-uniform_filter(high,31,mode='reflect'), low-high]
    return np.stack(a,axis=-1).reshape(-1,len(a))

def load_models() -> dict:
    raw=MODEL_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != MODEL_SHA256:
        raise RuntimeError('Frozen model checksum mismatch. Refusing inference.')
    return json.loads(raw)

def clean_mask(scores: np.ndarray, threshold: float = 0.5, min_area: int = 25) -> tuple[np.ndarray, list[dict]]:
    if not isinstance(threshold,numbers.Real) or not np.isfinite(threshold) or not 0.01 <= threshold <= 0.99:
        raise InputError('Threshold must be between 0.01 and 0.99.')
    if isinstance(min_area,bool) or not isinstance(min_area,numbers.Real) or not np.isfinite(min_area) or int(min_area)!=min_area or not 1<=min_area<=10_000:
        raise InputError('Minimum area must be an integer from 1 to 10,000.')
    if scores.ndim!=2 or not np.isfinite(scores).all():
        raise InputError('Finite 2-D scores required.')
    cc,n=label(scores>=threshold,structure=np.ones((3,3),dtype=np.uint8))
    sizes=np.bincount(cc.ravel());keep=sizes>=min_area;keep[0]=False
    mask=keep[cc]
    # Components are ranked by area, but capped in JSON/UI to prevent oversized output.
    ids=np.flatnonzero(keep);ids=ids[np.argsort(-sizes[ids],kind='stable')]
    components=[{'component_id':int(i),'area_pixels':int(sizes[i])} for i in ids[:100]]
    return mask,components

def infer(low: np.ndarray, high: np.ndarray | None = None, model: str = 'dual', threshold: float = 0.5, min_area: int = 25) -> dict:
    if model not in ('low','dual'):
        raise InputError('Model must be low or dual.')
    if model=='dual' and high is None:
        raise InputError('The dual model requires a matching 90 kV image.')
    # Validate a supplied pair even when using the low-energy comparator.
    if high is not None and (validate_array(high).shape != validate_array(low).shape):
        raise InputError('40 kV and 90 kV images must have identical dimensions.')
    x=features(low,high if model=='dual' else None)
    m=load_models()[model]
    z=((x-np.asarray(m['mean']))/np.asarray(m['scale'])) @ np.asarray(m['coef']) + m['intercept']
    scores=(1/(1+np.exp(-np.clip(z,-700,700)))).reshape(low.shape)
    mask,components=clean_mask(scores,threshold,min_area)
    component_total=int(label(mask,structure=np.ones((3,3),dtype=np.uint8))[1])
    exploratory=threshold != 0.5 or min_area != 25
    summary={'model':model,'model_sha256':MODEL_SHA256,'shape':list(low.shape),'input_dtype':'float32',
             'score_kind':'uncalibrated logistic score, NOT a probability of safety or disease',
             'threshold':float(threshold),'minimum_component_area':int(min_area),'connectivity':8,
             'whole_image_inference':True,'ground_truth_used_in_inference':False,
             'image_alarm':bool(mask.any()),'alarm_pixels':int(mask.sum()),
             'components_shown':len(components),'component_count':component_total,'components_truncated':component_total>100,
             'setting_status':'EXPLORATORY_NOT_VALIDATED' if exploratory else 'ORIGINAL_FROZEN_SETTINGS',
             'performance_status':'FAILED_DEVELOPMENT_VALIDATION',
             'known_failure':'Both models alarmed on all 10/10 normal validation images; two correlated fillets only.',
             'deployment_status':'NOT_DEPLOYED','authorship':'AI_ASSISTED','user_learning':'NOT_ASSESSED'}
    return {'scores':scores,'mask':mask,'components':components,'summary':summary}

def review(mask: np.ndarray, ground_truth: np.ndarray | None = None, known_normal: bool = False) -> dict:
    """Post-inference review only; never used to select a region or threshold."""
    if ground_truth is None:
        return {'status':'KNOWN_NORMAL_FALSE_ALARM_REVIEW' if known_normal else 'NO_GROUND_TRUTH',
                'false_alarm':bool(mask.any()) if known_normal else None,
                'note':'An alarm on known-normal input is a false alarm; no alarm does not prove safety.'}
    if ground_truth.shape != mask.shape or ground_truth.dtype!=np.bool_:
        raise InputError('Review mask must be boolean and match the image.')
    tp=int((mask & ground_truth).sum());fp=int((mask & ~ground_truth).sum());fn=int((~mask & ground_truth).sum())
    return {'status':'TRAIN_EXAMPLE_ONLY_NOT_EVALUATION','TP_pixels':tp,'FP_pixels':fp,'FN_pixels':fn,
            'dice':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            'note':'Descriptive training-example review; not generalization evidence.'}

def display_gray(a: np.ndarray) -> np.ndarray:
    lo,hi=np.percentile(a,[1,99])
    return (np.clip((a-lo)/max(float(hi-lo),1e-6),0,1)*255).astype(np.uint8)

def overlay(low: np.ndarray, mask: np.ndarray, gt: np.ndarray | None = None) -> np.ndarray:
    rgb=np.repeat(display_gray(low)[...,None],3,axis=2).astype(float)
    red=mask if gt is None else mask & ~gt
    rgb[red]=rgb[red]*0.35+np.array([255,73,100])*0.65
    if gt is not None:
        hit=mask & gt;miss=~mask & gt
        rgb[hit]=rgb[hit]*0.35+np.array([78,222,159])*0.65
        rgb[miss]=rgb[miss]*0.35+np.array([69,193,255])*0.65
    return rgb.astype(np.uint8)

"""Local CLI. Paths are only accepted from the local command-line operator."""
import argparse,json,pathlib
import numpy as np
from PIL import Image
from .examples import CHOICES,load_example
from .inference import infer,read_tiff,review,overlay,MAX_BYTES,InputError

def main():
    p=argparse.ArgumentParser(description='Frozen CWI learning lab (NOT production inspection)')
    p.add_argument('--example',choices=['synthetic','normal','bone'],default='synthetic')
    p.add_argument('--low',type=pathlib.Path);p.add_argument('--high',type=pathlib.Path)
    p.add_argument('--model',choices=['low','dual'],default='dual')
    p.add_argument('--threshold',type=float,default=.5);p.add_argument('--min-area',type=int,default=25)
    p.add_argument('--out',type=pathlib.Path,default=pathlib.Path('artifacts/local_run'))
    args=p.parse_args()
    try:
        if args.high and not args.low:raise InputError('--high requires --low.')
        if args.low:
            def read(path):
                with path.open('rb') as f:return read_tiff(f.read(MAX_BYTES+1))
            low=read(args.low);high=read(args.high) if args.high else None;gt=None
            meta={'source':'LOCAL_AUTHORIZED_UPLOAD','known_normal':False,'partition':'unknown'}
        else:
            low,high,gt,meta=load_example(CHOICES[['synthetic','normal','bone'].index(args.example)])
        r=infer(low,high,args.model,args.threshold,args.min_area)
        summary={**r['summary'],'input':meta,'review':review(r['mask'],gt,meta['known_normal']),'components':r['components']}
        args.out.mkdir(parents=True,exist_ok=True)
        (args.out/'summary.json').write_text(json.dumps(summary,indent=2))
        Image.fromarray(overlay(low,r['mask'],gt)).save(args.out/'review.png')
        Image.fromarray(r['mask'].astype(np.uint8)*255).save(args.out/'alarm_mask.png')
        print(json.dumps(summary,indent=2))
    except (InputError,OSError) as exc:p.error(str(exc))
if __name__=='__main__':main()

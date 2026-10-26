"""Launch and check real HTTP in one local process; never exposes a tunnel."""
import os,sys,json,pathlib,io
os.environ['GRADIO_ANALYTICS_ENABLED']='False'
os.environ.setdefault('MPLCONFIGDIR','/tmp/cwi_mpl')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/cwi_cache')
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import httpx
from app import demo,CSS,run_review
from cwi_lab.examples import CHOICES
from PIL import Image
import numpy as np
receipt={'status':'STARTED','scope':'Real local HTTP server; NOT browser/UI QA, NOT remote deployment'}
try:
    demo.queue(max_size=8,default_concurrency_limit=1).launch(server_name='127.0.0.1',server_port=7861,share=False,max_file_size='8mb',prevent_thread_lock=True,css=CSS,quiet=True)
    for route in ['/','/config','/gradio_api/info']:
        r=httpx.get('http://127.0.0.1:7861'+route,timeout=15)
        assert r.status_code==200,(route,r.status_code)
        receipt[route]={'status':r.status_code,'bytes':len(r.content)}
    fn=next(fn for fn in demo.fns.values() if fn.fn==run_review)
    base='http://127.0.0.1:7861'
    def predict(values):
        r=httpx.post(base+'/gradio_api/run/predict',json={'fn_index':fn._id,'data':values,'session_hash':'local-http-smoke'},timeout=30)
        assert r.status_code==200,(r.status_code,r.text[:500])
        return r.json()['data'][2]
    summary=predict([CHOICES[0],None,None,'dual',False,.5,25])
    assert summary['input']['source']=='SYNTHETIC_NOT_REAL_XRAY'
    receipt['http_inference']={'status':'PASS','alarm_pixels':summary['alarm_pixels'],'model':summary['model']}
    b=io.BytesIO();Image.fromarray(np.full((16,16),-.1,np.float32)).save(b,format='TIFF')
    upload=httpx.post(base+'/gradio_api/upload',files={'files':('signed.tiff',b.getvalue(),'image/tiff')},timeout=15)
    assert upload.status_code==200,(upload.status_code,upload.text)
    path=upload.json()[0]
    summary=predict([CHOICES[3],{'path':path,'url':base+'/gradio_api/file='+path,'meta':{'_type':'gradio.FileData'}},None,'low',False,.5,25])
    assert summary['input']['source']=='USER_AUTHORIZED_UPLOAD'
    receipt['http_upload_inference']={'status':'PASS','shape':summary['shape'],'model':summary['model']}
    receipt['status']='PASS' 
finally:
    demo.close()
    out=pathlib.Path(__file__).resolve().parents[1]/'artifacts/http_smoke.json'
    out.write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))

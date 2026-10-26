"""One bounded CPU profile: one TRAIN pair, warmup + exactly five repetitions.
No training, threshold search, held-out evaluation, or GPU operation.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import sys,pathlib,time,json,platform,statistics,importlib.metadata
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from cwi_lab.examples import load_example,CHOICES
from cwi_lab.inference import infer,MODEL_SHA256

def main():
    low,high,_,meta=load_example(CHOICES[1])
    receipt={'scope':'CPU whole-image frozen inference; not training, GPU, new validation or tuning',
             'input':{'Original_ID':meta['Original_ID'],'partition':'train','shape':list(low.shape)},
             'model_sha256':MODEL_SHA256,'warmup_calls_per_model':1,'timed_calls_per_model':5,
             'timed_region':'infer(): feature construction, model JSON/hash, standardization/logistic, 8conn cleanup, summary; excludes TIFF decode, UI, plotting and network',
             'clock':'time.perf_counter_ns','python':platform.python_version(),'platform':platform.platform(),
             'versions':{x:importlib.metadata.version(x) for x in ['numpy','scipy']},
             'configured_thread_limits':{k:os.environ[k] for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']},'results':{}}
    for model in ['low','dual']:
        infer(low,high,model)
        elapsed=[]
        for _ in range(5):
            start=time.perf_counter_ns();r=infer(low,high,model);elapsed.append((time.perf_counter_ns()-start)/1e6)
        receipt['results'][model]={'milliseconds':elapsed,'median_ms':statistics.median(elapsed),'min_ms':min(elapsed),'max_ms':max(elapsed),'alarm_pixels':r['summary']['alarm_pixels']}
    try:
        from threadpoolctl import threadpool_info
        receipt['observed_threadpools']=[{k:x.get(k) for k in ['user_api','internal_api','num_threads','prefix','version','threading_layer','architecture']} for x in threadpool_info()]
    except ImportError:receipt['observed_threadpools']='not measured; optional threadpoolctl is absent'
    receipt['limitations']='Single machine, one TRAIN input, five repetitions; not production latency, throughput, speedup, accuracy or hardware-generalization evidence.'
    out=pathlib.Path(__file__).resolve().parents[1]/'artifacts/cpu_profile.json';out.write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()

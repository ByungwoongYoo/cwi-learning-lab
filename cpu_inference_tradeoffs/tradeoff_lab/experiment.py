import argparse, csv, hashlib, io, json, os, platform, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
def sha(b): return hashlib.sha256(b).hexdigest()
def save_json(p,x): p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8')
def plan():
    p=ROOT/'PLAN.json'; expected=(ROOT/'PLAN.sha256').read_text().split()[0]
    if sha(p.read_bytes())!=expected: raise ValueError('Frozen plan hash mismatch')
    return json.loads(p.read_text())
def dataset(n,seed):
    g=np.random.default_rng(seed); x=g.standard_normal((n,64)).astype('float32')
    score=x[:,0]+.8*x[:,1]-.7*x[:,2]+.6*x[:,3]*x[:,4]+.5*np.sin(x[:,5]*2)+.2*x[:,6]
    return torch.from_numpy(x),torch.from_numpy((score>0).astype('int64'))
def model(): return nn.Sequential(nn.Linear(64,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,2))
def setup(c):
    torch.set_num_threads(c['threads']); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True); torch.manual_seed(c['seed'])
    if c['backend'] not in torch.backends.quantized.supported_engines: raise RuntimeError('Required quantized backend unavailable')
    torch.backends.quantized.engine=c['backend']
def metrics(logits,y):
    p=torch.softmax(logits,dim=1)
    return {'accuracy':float((p.argmax(1)==y).double().mean()),'cross_entropy':float(nn.functional.cross_entropy(logits,y))}
def batched(m,x,b): return torch.cat([m(x[i:i+b]) for i in range(0,len(x),b)])
def export_weights(m,path,quantized):
    arrays={}; meta={}
    for name,layer in m.named_modules():
        if isinstance(layer, nn.Linear): w=layer.weight.detach(); bias=layer.bias.detach()
        elif isinstance(layer,torch.ao.nn.quantized.dynamic.Linear): w=layer.weight(); bias=layer.bias()
        else: continue
        if w.is_quantized:
            arrays[name+'_weight']=w.int_repr().numpy(); meta[name]={'qscheme':str(w.qscheme()),'dtype':str(w.dtype)}
            if w.qscheme() in (torch.per_tensor_affine,torch.per_tensor_symmetric):
                arrays[name+'_scale']=np.array(w.q_scale(),dtype=np.float64);arrays[name+'_zero']=np.array(w.q_zero_point(),dtype=np.int64)
            else: raise ValueError('Unsupported export quantization scheme')
        else: arrays[name+'_weight']=w.numpy();meta[name]={'dtype':str(w.dtype)}
        arrays[name+'_bias']=bias.detach().numpy()
    np.savez(path,**arrays)
    # NPZ is a safe inspection export, NOT a loadable quantized model / runtime footprint.
    return {'npz_bytes':path.stat().st_size,'array_payload_bytes':sum(a.nbytes for a in arrays.values()),'layers':meta,'sha256':sha(path.read_bytes())}
def run(out):
    c=plan(); setup(c)
    out=Path(out)
    if out.exists() or out.is_symlink(): raise ValueError('Output exists; choose a fresh directory')
    if any(part.is_symlink() for part in [out,*out.parents]): raise ValueError('Symlink path refused')
    out.mkdir(parents=True)
    save_json(out/'plan_used.json',c)
    splits={k:dataset(c['sizes'][k],seed) for k,seed in c['data_seeds'].items()}
    split_receipts={k:{'rows':len(x),'positives':int(y.sum()),'x_sha256':sha(x.numpy().tobytes()),'y_sha256':sha(y.numpy().tobytes())} for k,(x,y) in splits.items()}
    save_json(out/'split_receipts.json',split_receipts)
    m=model();opt=torch.optim.Adam(m.parameters(),lr=c['learning_rate']);x,y=splits['train']; losses=[]
    g=torch.Generator().manual_seed(c['seed']+1);start=time.perf_counter()
    m.train()
    for epoch in range(c['epochs']):
        order=torch.randperm(len(x),generator=g)
        for ix in order.split(c['train_batch']):
            opt.zero_grad(set_to_none=True);loss=nn.functional.cross_entropy(m(x[ix]),y[ix]);loss.backward();opt.step()
        losses.append(float(loss.detach()))
    train_s=time.perf_counter()-start;m.eval();start=time.perf_counter()
    q=torch.ao.quantization.quantize_dynamic(m,{nn.Linear},dtype=torch.qint8,inplace=False).eval()
    quant_s=time.perf_counter()-start
    if sum(isinstance(l,torch.ao.nn.quantized.dynamic.Linear) for l in q.modules())!=3: raise AssertionError('All three Linear layers must quantize')
    env={'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform(),'machine':platform.machine(),'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'backend':torch.backends.quantized.engine,'supported_backends':torch.backends.quantized.supported_engines,'cpu_count':os.cpu_count(),'torch_config':torch.__config__.show(),'source_api_doc':torch.ao.quantization.quantize_dynamic.__doc__}
    save_json(out/'environment.json',env)
    weights={'fp32':export_weights(m,out/'fp32_weights.npz',False),'int8':export_weights(q,out/'int8_weights.npz',True)}
    # Same torch.save state_dict method to BytesIO; sizes only, pickle artifacts not distributed.
    for name,mod in [('fp32',m),('int8',q)]:
        bio=io.BytesIO();torch.save(mod.state_dict(),bio);weights[name]['torch_state_dict_bytes_in_memory']=bio.tell()
    quality=[];raw=[];summary=[]
    with torch.inference_mode():
        cx,cy=splits['calibration'];cl=m(cx[:256]);cq=q(cx[:256]);assert cl.shape==(256,2) and cq.shape==(256,2) and torch.isfinite(cl).all() and torch.isfinite(cq).all()
        save_json(out/'calibration_smoke.json',{'rows_checked':256,'finite':True,'tuned':False,'purpose':'shape and finite smoke; dynamic quantization does not fit observers on this split'})
        tx,ty=splits['test']
        for b in c['batch_sizes']:
            f=batched(m,tx,b);z=batched(q,tx,b);fm=metrics(f,ty);qm=metrics(z,ty)
            diff=(f.softmax(1)-z.softmax(1)).abs();drop=fm['accuracy']-qm['accuracy']
            quality.append({'batch':b,'fp32':fm,'int8':qm,'accuracy_drop':drop,'mean_probability_difference':float(diff.mean()),'max_probability_difference':float(diff.max()),'prediction_disagreements':int((f.argmax(1)!=z.argmax(1)).sum()),'predeclared_quality_gate_pass':drop<=c['accuracy_drop_tolerance'] and float(diff.max())<=c['max_probability_difference_tolerance']})
            np.savez(out/f'predictions_batch{b}.npz',fp32=f.numpy(),int8=z.numpy(),labels=ty.numpy())
            inp=tx[:b].contiguous()
            for _ in range(c['warmup']):m(inp);q(inp)
            for block in range(c['blocks']):
                order=['fp32','int8'] if block%2==0 else ['int8','fp32']
                for rank,name in enumerate(order):
                    mod=m if name=='fp32' else q
                    t0=time.perf_counter_ns()
                    for _ in range(c['calls_per_block']):mod(inp)
                    elapsed=time.perf_counter_ns()-t0
                    raw.append({'batch':b,'block':block,'order':rank,'variant':name,'elapsed_ns':elapsed,'calls':c['calls_per_block'],'ns_per_call':elapsed/c['calls_per_block']})
            a=np.array([r['ns_per_call'] for r in raw if r['batch']==b and r['variant']=='fp32']);d=np.array([r['ns_per_call'] for r in raw if r['batch']==b and r['variant']=='int8']);ratios=a/d
            summary.append({'batch':b,'fp32_block_mean_ns_p50':float(np.median(a)),'fp32_block_mean_ns_p95':float(np.quantile(a,.95)),'int8_block_mean_ns_p50':float(np.median(d)),'int8_block_mean_ns_p95':float(np.quantile(d,.95)),'paired_fp32_over_int8_median':float(np.median(ratios)),'paired_ratio_p05':float(np.quantile(ratios,.05)),'paired_ratio_p95':float(np.quantile(ratios,.95)),'paired_ratio_std':float(np.std(ratios,ddof=1)),'int8_faster_blocks':int((ratios>1).sum()),'blocks':len(a)})
    with (out/'timing_raw.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(raw[0]));w.writeheader();w.writerows(raw)
    result={'schema':1,'plan_sha256':sha((ROOT/'PLAN.json').read_bytes()),'parameters':sum(p.numel() for p in m.parameters()),'train_seconds':train_s,'quantize_seconds':quant_s,'epochs':c['epochs'],'last_batch_train_losses':losses,'weights':weights,'quality':quality,'timing':summary,'limitations':['Synthetic task only; no external generalization or production SLA.','Timing p50/p95 describe block-mean latency, NOT individual request tail latency.','Only one host, one process, one CPU thread; no network/I/O/concurrency.','Dynamic activation quantization is batch dependent.','Safe NPZ export is inspection format, not executable packed-model size or process RSS.','Torch state_dict sizes measured in memory only; no pickle files delivered.','AB/BA alternation mitigates drift but cannot eliminate host noise.']}
    save_json(out/'result.json',result)
    print(json.dumps({'result':str(out/'result.json'),'timing':summary,'quality':quality},indent=2))
def main():
    p=argparse.ArgumentParser(description='Frozen synthetic CPU FP32 vs dynamic INT8 comparison. Requires PyTorch/NumPy.');p.add_argument('--output',required=True)
    a=p.parse_args()
    try:run(a.output)
    except (ValueError,RuntimeError,AssertionError) as e: print(f'ERROR: {e}',file=sys.stderr);sys.exit(2)
if __name__=='__main__':main()

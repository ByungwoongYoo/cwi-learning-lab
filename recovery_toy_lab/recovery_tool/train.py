"""Real PyTorch CPU training and complete, non-pickle mid-epoch checkpoints."""
from __future__ import annotations
import argparse, os, platform, resource, shutil, tempfile, time
from pathlib import Path
import numpy as np
import torch
from .common import ROOT, FormatError, canonical, digest, json_read, json_write, checked_npz, frozen_config, exact_keys
from .data import load_data
PARAM_SHAPES=[(16,7),(16,),(1,16),(1,)]
META_KEYS=['schema_version','format','config_sha256','data_sha256','sample_indices_sha256','archive_sha256','runtime','optimizer','progress','sampler_rng','losses','initial_loss']

def runtime(): return {'torch':torch.__version__,'numpy':np.__version__,'python':platform.python_version()}
def setup(c):
    torch.set_num_threads(c['threads']);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(c['model_seed'])

def make_model(): return torch.nn.Sequential(torch.nn.Linear(7,16,dtype=torch.float64),torch.nn.Tanh(),torch.nn.Dropout(0.1),torch.nn.Linear(16,1,dtype=torch.float64))
def optimizer_config(c): return {'name':'SGD','learning_rate':c['learning_rate'],'momentum':c['momentum'],'weight_decay':c['weight_decay'],'dampening':0.0,'nesterov':False}
def specs():
    s={**{f'p{i}':(shape,'float64') for i,shape in enumerate(PARAM_SHAPES)},**{f'v{i}':(shape,'float64') for i,shape in enumerate(PARAM_SHAPES)}}
    s['torch_rng']=((torch.get_rng_state().numel(),),'uint8');s['permutation']=((8192,),'int64')
    return s

def _integer(v,lo,hi): return type(v) is int and lo<=v<=hi

def read_checkpoint(folder,c,ch,dm):
    folder=Path(folder);m=json_read(folder/'manifest.json');exact_keys(m,META_KEYS)
    if type(m['schema_version']) is not int or m['schema_version']!=1 or m['format']!='RECOVERY_TOY_JSON_NPZ_V1': raise FormatError('Checkpoint schema/version mismatch')
    if m['config_sha256']!=ch or m['data_sha256']!=dm['archive_sha256'] or m['sample_indices_sha256']!=dm['sample_indices_sha256']: raise FormatError('Checkpoint binding mismatch')
    if m['runtime']!=runtime() or canonical(m['optimizer'])!=canonical(optimizer_config(c)): raise FormatError('Runtime/optimizer mismatch')
    p=m['progress'];exact_keys(p,['epoch','cursor','step'])
    if not _integer(p['step'],1,128) or not _integer(p['epoch'],0,4) or not _integer(p['cursor'],0,8191): raise FormatError('Invalid progress')
    if p['epoch']!=p['step']//32 or p['cursor']!=(p['step']%32)*256: raise FormatError('Progress consistency mismatch')
    if type(m['losses']) is not list or len(m['losses'])!=p['step'] or any(type(x) not in (int,float) or not np.isfinite(x) or x<0 for x in m['losses']): raise FormatError('Invalid losses')
    if type(m['initial_loss']) not in (int,float) or not np.isfinite(m['initial_loss']) or m['initial_loss']<0: raise FormatError('Invalid initial loss')
    r=m['sampler_rng'];exact_keys(r,['bit_generator','state','has_uint32','uinteger']);exact_keys(r['state'],['state','inc'])
    if r['bit_generator']!='PCG64' or not _integer(r['state']['state'],0,2**128-1) or not _integer(r['state']['inc'],1,2**128-1) or r['state']['inc']%2!=1 or not _integer(r['has_uint32'],0,1) or not _integer(r['uinteger'],0,2**32-1): raise FormatError('Invalid sampler RNG')
    a=checked_npz(folder/'state.npz',m['archive_sha256'],specs())
    if not np.array_equal(np.sort(a['permutation']),np.arange(8192)): raise FormatError('Invalid permutation')
    # Validate RNG payload on isolated generators, without mutating live global state.
    try:
        probe=np.random.default_rng();probe.bit_generator.state=r
        torch.Generator(device='cpu').set_state(torch.from_numpy(a['torch_rng']))
    except (ValueError,RuntimeError,TypeError) as e: raise FormatError('Invalid RNG payload') from e
    return m,a

def save_checkpoint(folder,c,ch,dm,model,opt,rng,permutation,progress,losses,initial_loss):
    folder=Path(folder)
    if folder.exists(): raise FileExistsError('Refusing to overwrite checkpoint directory')
    folder.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(tempfile.mkdtemp(prefix='.checkpoint-',dir=folder.parent))
    try:
        arrays={'torch_rng':torch.get_rng_state().numpy().copy(),'permutation':permutation.copy()}
        for i,p in enumerate(model.parameters()):
            arrays[f'p{i}']=p.detach().numpy().copy()
            arrays[f'v{i}']=opt.state[p]['momentum_buffer'].detach().numpy().copy()
        np.savez(tmp/'state.npz',**arrays)
        m={'schema_version':1,'format':'RECOVERY_TOY_JSON_NPZ_V1','config_sha256':ch,'data_sha256':dm['archive_sha256'],'sample_indices_sha256':dm['sample_indices_sha256'],'archive_sha256':digest((tmp/'state.npz').read_bytes()),'runtime':runtime(),'optimizer':optimizer_config(c),'progress':progress,'sampler_rng':rng.bit_generator.state,'losses':losses,'initial_loss':initial_loss}
        json_write(tmp/'manifest.json',m)
        read_checkpoint(tmp,c,ch,dm) # self-check before atomic directory publication
        os.replace(tmp,folder)
    finally:
        if tmp.exists(): shutil.rmtree(tmp)

def evaluate(model,x,y):
    model.eval()
    with torch.no_grad():
        logits=model(x);loss=float(torch.nn.functional.binary_cross_entropy_with_logits(logits,y));scores=torch.sigmoid(logits).numpy().copy()
    model.train()
    return loss,scores

def run(data_dir,out,stop_step=128,resume=None):
    overall_wall=time.perf_counter();overall_cpu=time.process_time()
    c,ch=frozen_config()
    if not _integer(stop_step,1,128): raise ValueError('Stop step must be 1..128')
    setup(c);a,dm=load_data(data_dir)
    x=torch.from_numpy(a['x']);y=torch.from_numpy(a['y'])
    model=make_model();opt=torch.optim.SGD(model.parameters(),lr=c['learning_rate'],momentum=c['momentum'],weight_decay=c['weight_decay'],foreach=False)
    rng=np.random.default_rng(c['sampler_seed']);permutation=rng.permutation(len(x));p={'epoch':0,'cursor':0,'step':0};losses=[]
    initial_loss,_=evaluate(model,x,y)
    if resume:
        meta,state=read_checkpoint(resume,c,ch,dm)
        with torch.no_grad():
            for i,param in enumerate(model.parameters()):
                param.copy_(torch.from_numpy(state[f'p{i}']))
                opt.state[param]['momentum_buffer']=torch.from_numpy(state[f'v{i}']).clone()
        rng.bit_generator.state=meta['sampler_rng'];torch.set_rng_state(torch.from_numpy(state['torch_rng']))
        permutation=state['permutation'];p=meta['progress'];losses=meta['losses'];initial_loss=meta['initial_loss']
    if stop_step<=p['step']: raise ValueError('Stop step must exceed checkpoint step')
    start_step=p['step'];w=time.perf_counter();cpu=time.process_time()
    while p['step']<stop_step:
        idx=permutation[p['cursor']:p['cursor']+c['batch_size']]
        opt.zero_grad(set_to_none=True)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x[idx]),y[idx])
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
        loss.backward()
        if any(t.grad is None or not torch.isfinite(t.grad).all() for t in model.parameters()): raise RuntimeError('Nonfinite/missing gradients')
        opt.step();losses.append(float(loss.detach()));p['step']+=1;p['cursor']+=len(idx)
        if p['cursor']==len(x):
            p['epoch']+=1;p['cursor']=0
            if p['epoch']<c['epochs']: permutation=rng.permutation(len(x))
    training_cpu=time.process_time()-cpu;training_wall=time.perf_counter()-w
    if any(not torch.isfinite(t).all() for t in model.parameters()): raise RuntimeError('Nonfinite model')
    final_loss,scores=evaluate(model,x,y)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    save_checkpoint(out/'checkpoint',c,ch,dm,model,opt,rng,permutation,p,losses,initial_loss)
    np.savez(out/'predictions.npz',scores=scores)
    predicted=scores>=c['score_threshold'];labels=a['y'].astype(bool)
    summary={'status':'TECHNICAL_TRAINING_SMOKE_ONLY','metrics_scope':'8192 synthetic balanced training rows; no generalization claim','source_examples':0,'train_samples':len(x),'positive_labels':int(y.sum()),'architecture':[7,16,1],'parameter_count':sum(t.numel() for t in model.parameters()),'device':'cpu','runtime':runtime(),'pid':os.getpid(),'resumed_from_step':start_step,'optimizer_steps_this_process':p['step']-start_step,'optimizer_steps_total':p['step'],'initial_train_loss':initial_loss,'final_train_loss':final_loss,'all_training_losses_finite':bool(np.isfinite(losses).all()),'train_confusion':{'TP':int((predicted&labels).sum()),'FP':int((predicted&~labels).sum()),'FN':int((~predicted&labels).sum()),'TN':int((~predicted&~labels).sum())},'training_wall_seconds':training_wall,'training_process_cpu_seconds':training_cpu,'process_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'run_wall_seconds':time.perf_counter()-overall_wall,'run_process_cpu_seconds':time.process_time()-overall_cpu,'prediction_sha256':digest(scores.tobytes()),'authorship':'AI_ASSISTED','user_competency':'NOT_ASSESSED'}
    json_write(out/'summary.json',summary)
    return summary

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',required=True);parser.add_argument('--out',required=True);parser.add_argument('--stop-step',type=int,default=128);parser.add_argument('--resume');args=parser.parse_args()
    import json
    print(json.dumps(run(args.data,args.out,args.stop_step,args.resume),sort_keys=True))
if __name__=='__main__': main()

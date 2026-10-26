"""Bounded, single-writer checkpoint generations. No deletion, pickle or networking."""
from pathlib import Path
import os,re,stat,sys,shutil,time,uuid,itertools
PROJECT=Path(__file__).resolve().parents[1]

from recovery_tool.common import frozen_config,FormatError,json_read
from recovery_tool.train import read_checkpoint
from recovery_tool.data import load_data
RUNS=Path(os.environ.get('RECOVERY_TOOL_RUNS',str(Path.cwd()))).absolute()
NAME=re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,47}\Z')
GEN=re.compile(r'g([0-9]{6})\Z')
MAX_ENTRIES=100
class StoreError(ValueError): pass

def checked_path(path):
    p=Path(path).absolute()
    if '..' in p.parts: raise StoreError('Parent traversal forbidden')
    try: rel=p.relative_to(RUNS)
    except ValueError: raise StoreError('Path outside project runs')
    for node in [*reversed(RUNS.parents),RUNS,*[RUNS.joinpath(*rel.parts[:i]) for i in range(1,len(rel.parts)+1)]]:
        if node.is_symlink(): raise StoreError('Symbolic link forbidden')
    return p

def run_root(name,create=False):
    if type(name) is not str or not NAME.fullmatch(name): raise StoreError('Invalid run name')
    checked_path(RUNS/name)
    if create:
        RUNS.mkdir(exist_ok=True)
        (RUNS/name).mkdir(exist_ok=False)
    elif not (RUNS/name).is_dir(): raise StoreError('Run does not exist')
    return RUNS/name

def bounded_files(folder):
    p=checked_path(folder)
    if not p.is_dir(): raise StoreError('Checkpoint is not a directory')
    entries=list(itertools.islice(p.iterdir(),3))
    if len(entries)!=2 or {q.name for q in entries}!={'manifest.json','state.npz'}: raise StoreError('Unexpected checkpoint members')
    for q,limit in [(p/'manifest.json',131072),(p/'state.npz',4*1024*1024)]:
        checked_path(q)
        st=q.lstat()
        if not stat.S_ISREG(st.st_mode) or st.st_nlink!=1: raise StoreError('Only regular, non-hardlinked files allowed')
        if not 0<st.st_size<=limit: raise StoreError('Checkpoint member size invalid')
    return p

def bindings(root):
    for name,limit in [('manifest.json',131072),('data.npz',4*1024*1024)]:
        p=checked_path(root/'data'/name);st=p.lstat()
        if not stat.S_ISREG(st.st_mode) or st.st_nlink!=1 or not 0<st.st_size<=limit: raise StoreError('Invalid data member')
    c,ch=frozen_config();_,dm=load_data(checked_path(root/'data'))
    return c,ch,dm

def validate(folder,root,expected=None):
    p=bounded_files(folder);c,ch,dm=bindings(root)
    m,a=read_checkpoint(p,c,ch,dm)
    if expected is not None and m['progress']['step']!=expected: raise StoreError('Generation and saved step differ')
    return m,a

def inspect(root):
    root=checked_path(root);base=checked_path(root/'generations')
    if not base.is_dir(): raise StoreError('Generation directory missing')
    nodes=list(itertools.islice(base.iterdir(),MAX_ENTRIES+1))
    if len(nodes)>MAX_ENTRIES: raise StoreError('Generation entry bound exceeded')
    valid=[];invalid=[]
    for p in sorted(nodes):
        match=GEN.fullmatch(p.name)
        if not match:
            invalid.append({'name':p.name,'reason':'Noncanonical or incomplete generation name'});continue
        try:
            step=int(match[1]);validate(p,root,step)
            valid.append({'name':p.name,'step':step})
        except (StoreError,FormatError,OSError,ValueError,RuntimeError) as e:
            invalid.append({'name':p.name,'reason':type(e).__name__+': '+str(e)})
    valid.sort(key=lambda q:q['step'])
    return {'valid':valid,'invalid':invalid,'selected':valid[-1] if valid else None,'cleanup':'DRY_RUN_ONLY_NO_FILES_DELETED'}

def latest(root):
    report=inspect(root)
    if report['selected'] is None: raise StoreError('NO_VALID_CHECKPOINT: refusing silent restart')
    return root/'generations'/report['selected']['name'],report

def hook(label):
    if os.environ.get('RECOVERY_FAULT_STAGE')==label:
        marker=checked_path(os.environ['RECOVERY_FAULT_MARKER']);marker.write_text(label)
        while True: time.sleep(.02)

def commit(root,source):
    root=checked_path(root);source=bounded_files(source);m,_=validate(source,root)
    step=m['progress']['step'];base=checked_path(root/'generations')
    if not base.is_dir(): raise StoreError('Generation directory missing')
    if len(list(itertools.islice(base.iterdir(),MAX_ENTRIES)))>=MAX_ENTRIES: raise StoreError('Generation entry bound exceeded')
    destination=checked_path(base/f'g{step:06d}')
    if destination.exists(): raise StoreError('Duplicate generation; never overwrite')
    tmp=checked_path(base/('.pending-'+uuid.uuid4().hex));tmp.mkdir()
    # Failed or interrupted staging is retained for inspection, never automatically removed.
    shutil.copyfile(source/'state.npz',tmp/'state.npz');hook('array_closed')
    shutil.copyfile(source/'manifest.json',tmp/'manifest.json');hook('manifest_closed')
    validate(tmp,root,step);hook('validated')
    if destination.exists(): raise StoreError('Duplicate generation; concurrent writers unsupported')
    os.replace(tmp,destination);hook('published')
    return destination

"""Optional build-time fetch. Fixed TRAIN allowlist, strict ranges, 30 MiB cap.
Never imported or called by the web app. No user-provided URL or path is accepted.
"""
import hashlib,json,pathlib,struct,urllib.request,zlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
ALLOW=ROOT/'examples/allowlist.json'
EXPECTED_ALLOWLIST='19094a15a645d0b971d125b0fcaa9ed7ccc56ff0750a0dad3f8de6c8577ac246'
CAP=30*1024*1024

def main():
    assert hashlib.sha256(ALLOW.read_bytes()).hexdigest()==EXPECTED_ALLOWLIST
    m=json.loads(ALLOW.read_text()); receipt={'received':0,'reserved':0,'requests':[],'members':{}}
    dest=ROOT/'examples/public_train';dest.mkdir(exist_ok=True)
    rp=dest/'fetch_receipt.json'
    if rp.exists():
        receipt=json.loads(rp.read_text())
    def save(): rp.write_text(json.dumps(receipt,indent=2))
    def fetch(start,n,name):
        if n<=0 or receipt['reserved']+n>CAP: raise RuntimeError('30 MiB cap exceeded')
        receipt['reserved']+=n;save()
        req=urllib.request.Request(m['source_url'],headers={'Range':f'bytes={start}-{start+n-1}','Accept-Encoding':'identity'})
        with urllib.request.urlopen(req,timeout=45) as r:
            if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{start+n-1}/{m["source_archive_bytes"]}':
                raise RuntimeError('Exact range not honored; response body not read')
            b=r.read(n+1)
        receipt['received']+=len(b);receipt['requests'].append({'start':start,'length':n,'member':name});save()
        if len(b)!=n:raise RuntimeError('Length mismatch')
        return b
    mapping={v:(r,k) for r in m['lineages'] for k,v in r['members'].items()}
    for row in m['members']:
        name=row['path'];r,k=mapping[name]
        assert row['partition']=='train' and r['partition']=='train' and '/Dataset #1/' in name and '/train/' in name and '/test/' not in name
        out=dest/f"original_{r['Original_ID']:03d}_{k}.tiff"
        if out.exists():
            b=out.read_bytes();assert len(b)==row['size'] and zlib.crc32(b)==row['crc32'];continue
        h=fetch(row['header_offset'],30,name);fields=struct.unpack('<4s5H3L2H',h)
        assert fields[0]==b'PK\x03\x04' and fields[3]==row['compression'];nl,el=fields[-2:]
        b=fetch(row['header_offset']+30,nl+el+row['compressed_size'],name)
        assert b[:nl].decode('utf-8' if row['flags']&2048 else 'cp437')==name
        d=zlib.decompressobj(-15);raw=d.decompress(b[nl+el:],row['size']+1)
        assert d.eof and not d.unused_data and not d.unconsumed_tail
        assert len(raw)==row['size'] and zlib.crc32(raw)==row['crc32']
        out.write_bytes(raw)
        receipt['members'][out.name]={'source_member':name,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'partition':'train','Original_ID':r['Original_ID']}
        save();print(out.name,len(raw),flush=True)
    receipt['complete']=True;save()
if __name__=='__main__':main()

import asyncio,io
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
import gradio as gr
from gradio import utils
from gradio.blocks import Block
from app import demo,run_review
from cwi_lab.examples import CHOICES

FN=next(fn for fn in demo.fns.values() if fn.fn==run_review)

def inputs(value):return [CHOICES[3],value,None,'low',False,.5,25]

def filedata(path,**kw):return {'path':str(path),'meta':{'_type':'gradio.FileData'},**kw}

@pytest.mark.parametrize('payload',[filedata('https://audit.invalid/never-requested.tiff'),filedata('http://127.0.0.1/private'),filedata('/etc/passwd'),filedata('file:///etc/passwd'),filedata('//audit.invalid/a'),{'nested':filedata('https://audit.invalid/a')}])
def test_reject_before_any_cache_move(monkeypatch,payload):
    calls=[]
    async def forbidden(*args,**kwargs):calls.append(args);raise AssertionError('Cache movement must not occur')
    monkeypatch.setattr(Block,'async_move_resource_to_block_cache',forbidden)
    with pytest.raises(gr.Error):asyncio.run(demo.preprocess_data(FN,inputs(payload),None))
    assert not calls

def test_nested_url_in_non_file_control_rejected_before_cache(monkeypatch):
    calls=[]
    async def forbidden(*args,**kwargs):calls.append(args);raise AssertionError('Unexpected cache call')
    monkeypatch.setattr(Block,'async_move_resource_to_block_cache',forbidden)
    x=inputs(None);x[0]=filedata('https://audit.invalid/a')
    with pytest.raises(gr.Error):asyncio.run(demo.preprocess_data(FN,x,None))
    assert not calls

def test_local_upload_roundtrip():
    folder=Path(utils.get_upload_folder())/'cwi_guard_test';folder.mkdir(parents=True,exist_ok=True)
    path=folder/'signed.tiff'
    try:
        Image.fromarray(np.full((16,16),-.1,np.float32)).save(path)
        processed=asyncio.run(demo.preprocess_data(FN,inputs(filedata(path,url='http://localhost:7860/gradio_api/file='+str(path))),None))
        assert isinstance(processed[1],bytes)
        assert run_review(*processed)[2]['input']['source']=='USER_AUTHORIZED_UPLOAD'
    finally:path.unlink(missing_ok=True)

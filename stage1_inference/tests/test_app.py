import socket
import pytest
from app import run_review
from cwi_lab.examples import CHOICES

def test_actual_callback_and_repeat():
    for _ in range(2):
        original,output,summary,rows,status=run_review(CHOICES[0],None,None,'dual',False,.99,999)
        assert original.shape==(256,320) and output.shape==(256,320,3)
        assert summary['threshold']==.5 and summary['minimum_component_area']==25
        assert summary['input']['source']=='SYNTHETIC_NOT_REAL_XRAY'
        assert 'ALARM' in status

def test_upload_failure_clears_prior_outputs():
    original,output,summary,rows,status=run_review(CHOICES[3],b'bad',None,'dual',False,.5,25)
    assert original is None and output is None and rows==[]
    assert summary['status']=='INPUT_REJECTED'

def test_no_outbound_network_during_inference(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Outbound network attempted')
    monkeypatch.setattr(socket.socket,'connect',forbidden)
    _,_,summary,_,_=run_review(CHOICES[0],None,None,'dual',False,.5,25)
    assert summary['performance_status']=='FAILED_DEVELOPMENT_VALIDATION'

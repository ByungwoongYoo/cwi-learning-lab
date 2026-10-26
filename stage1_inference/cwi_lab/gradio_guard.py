"""App-scoped guard for Gradio 6.30 pre-cache input handling.
Reject URL/path injection before Gradio can fetch/cache a FileData value.
This deliberately supports only this app's scalar controls and upload inputs.
"""
from pathlib import Path
import numbers
import gradio as gr
from gradio import utils
from gradio.data_classes import FileData
from .inference import MAX_BYTES

class LocalUploadBlocks(gr.Blocks):
    async def preprocess_data(self, block_fn, inputs, state, explicit_call=False):
        self.validate_inputs(block_fn,inputs)
        inputs=list(inputs)
        for index,(block,value) in enumerate(zip(block_fn.inputs,inputs)):
            if isinstance(block,gr.File):
                if value is None:continue
                if isinstance(value,FileData):value=value.model_dump()
                if not isinstance(value,dict) or not isinstance(value.get('path'),str):
                    raise gr.Error('Only a local file uploaded through this app is accepted.')
                path=value['path']
                # Reject URI schemes, scheme-relative locations and embedded traversal before resolving.
                if '://' in path or path.startswith('//') or '\x00' in path or value.get('is_stream'):
                    raise gr.Error('Remote URLs and streams are not accepted. Upload a TIFF file.')
                try:
                    root=Path(utils.get_upload_folder()).resolve(strict=True)
                    candidate=Path(path).resolve(strict=True)
                    if not candidate.is_relative_to(root) or not candidate.is_file():
                        raise ValueError('not an upload')
                    if candidate.stat().st_size>MAX_BYTES:raise ValueError('size cap')
                except (OSError,ValueError,RuntimeError):
                    raise gr.Error('The file must be an app upload of at most 8 MiB.') from None
                # URL metadata varies by Gradio frontend/root. It is never needed to
                # read an already-verified local upload: strip it and ALL extra nested
                # client fields rather than forwarding them to cache traversal.
                inputs[index]={'path':str(candidate),'meta':{'_type':'gradio.FileData'}}
            else:
                # All this app's non-file controls are scalars. Prevent nested FileData
                # in a dropdown/slider from reaching Gradio's recursive cache traversal.
                if value is not None and not isinstance(value,(str,bool,numbers.Real)):
                    raise gr.Error('Invalid non-file control value.')
        return await super().preprocess_data(block_fn,inputs,state,explicit_call=explicit_call)

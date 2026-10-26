"""Private/local-by-default Gradio app. No model downloads or outbound inference."""
import os
os.environ['GRADIO_ANALYTICS_ENABLED']='False'
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import gradio as gr
from cwi_lab.examples import CHOICES,load_example
from cwi_lab.gradio_guard import LocalUploadBlocks
from cwi_lab.inference import infer,read_tiff,review,overlay,display_gray,InputError

WARNING='학습용 실패 분석 데모 · FAILED BASELINE · 의료/생산 검사에 사용 금지'
CSS='''
.gradio-container {max-width:1240px !important; margin:auto;}
#hero {background:linear-gradient(120deg,#102936,#154d52);padding:26px;border-radius:16px;color:#f4fbfb;}
#hero h1,#hero p {color:#f4fbfb;} #warning {border-left:5px solid #e1a328;padding:12px;background:#fff7e7;color:#442f0c;}
'''

def run_review(choice, low_bytes, high_bytes, model, exploratory, threshold, min_area):
    """HTTP uploads are bytes, never arbitrary server paths or URLs."""
    try:
        if choice==CHOICES[3]:
            low=read_tiff(low_bytes)
            high=read_tiff(high_bytes) if high_bytes is not None else None
            gt=None;meta={'source':'USER_AUTHORIZED_UPLOAD','partition':'unknown','known_normal':False,
                          'note':'No ground truth supplied; red pixels cannot be called false positives.'}
        else:
            low,high,gt,meta=load_example(choice)
        t=threshold if exploratory else .5;a=min_area if exploratory else 25
        result=infer(low,high,model,t,a)
        summary={**result['summary'],'input':meta,'review':review(result['mask'],gt,meta['known_normal'])}
        alarm='ALARM — 경보 있음' if summary['image_alarm'] else 'NO ALARM — 안전 판정 아님'
        status=f'### {alarm}\n**{summary["setting_status"]}** · {summary["alarm_pixels"]:,} alarm pixels\n\n{meta["note"]}'
        rows=[[r['component_id'],r['area_pixels']] for r in result['components']]
        return display_gray(low),overlay(low,result['mask'],gt),summary,rows,status
    except InputError as exc:
        # Clear all prior result outputs on failure; do not leave a stale alarm visible.
        return None,None,{'error':str(exc),'status':'INPUT_REJECTED'},[],f'### 입력 거절\n{exc}'

def make_app():
    with LocalUploadBlocks(title='CWI Learning Lab',analytics_enabled=False,delete_cache=(3600,3600)) as demo:
        gr.Markdown('# CWI Learning Lab\n입력 → 고정 모델 추론 → 오경보 검토\n\nAI_ASSISTED · USER LEARNING: NOT_ASSESSED · NOT_DEPLOYED',elem_id='hero')
        gr.Markdown('**'+WARNING+'**\n\n두 모델 모두 기존 검증 정상 영상 **10/10에서 경보**. 성능 향상이 아닌 소프트웨어/오류 분석 학습 프로젝트입니다.',elem_id='warning')
        with gr.Row():
            with gr.Column(scale=1):
                choice=gr.Dropdown(CHOICES,value=CHOICES[0],label='1 · 입력 선택')
                gr.Markdown('CWI 예제는 공개 **훈련 데이터**입니다. 개인정보·KAMP 비공개 자료는 업로드하지 마세요. 업로드는 이 서버로 전송되지만 외부 추론 서비스에는 보내지 않습니다.')
                low=gr.File(label='40 kV • float32 TIFF • ≤8 MiB',file_types=['.tif','.tiff'],type='binary')
                high=gr.File(label='90 kV • matching float32 TIFF • ≤8 MiB',file_types=['.tif','.tiff'],type='binary')
                model=gr.Radio(['dual','low'],value='dual',label='2 · 고정 모델')
                with gr.Accordion('탐색 설정 — 검증 성능이 아닙니다',open=False):
                    exploratory=gr.Checkbox(value=False,label='EXPLORATORY mode 활성화')
                    threshold=gr.Slider(.01,.99,value=.5,step=.01,label='Uncalibrated score threshold')
                    min_area=gr.Slider(1,1000,value=25,step=1,label='8-connected component minimum pixels')
                    gr.Markdown('기본값 0.5 / 25는 기존 동결 설정. 변경 결과로 개선됐다고 주장하지 않습니다.')
                run=gr.Button('3 · 추론하고 오경보 검토',variant='primary')
                clear=gr.ClearButton([low,high])
            with gr.Column(scale=2):
                status=gr.Markdown('### 먼저 예제를 실행해 보세요\n실제 모델 계산이 실행됩니다. 합성 예제는 실제 X-ray가 아닙니다.')
                with gr.Row():
                    original=gr.Image(label='40 kV display • percentile scaling for display ONLY',interactive=False)
                    output=gr.Image(label='Review overlay • whole-image inference',interactive=False)
                gr.Markdown('**색상**: 정답 없음 → 빨강=모델 경보. CWI 훈련 예제 → 빨강=FP, 초록=TP, 파랑=FN. 표시 대비만 조절하며 모델 입력은 signed float32 그대로 유지합니다.')
                components=gr.Dataframe(headers=['Component ID','Area (pixels)'],datatype=['number','number'],interactive=False,label='Largest retained components (max 100)')
                result=gr.JSON(label='Inference + review receipt')
        run.click(run_review,[choice,low,high,model,exploratory,threshold,min_area],[original,output,result,components,status],api_name=False,concurrency_limit=1)
        clear.click(lambda:(None,None,{},[],'### 결과를 지웠습니다'),outputs=[original,output,result,components,status],api_name=False)
        gr.Markdown('Public source: [Vladyslav Andriiashen / CWI, 2024](https://doi.org/10.5281/zenodo.10579608), CC BY 4.0. No source-author endorsement. [License](https://creativecommons.org/licenses/by/4.0/). No validation/test payload is bundled.')
    return demo

demo=make_app()
if __name__=='__main__':
    # Local-only by default. SPACE_ID is supplied only by a future authorized HF Space.
    host='0.0.0.0' if os.environ.get('SPACE_ID') else '127.0.0.1'
    demo.queue(max_size=8,default_concurrency_limit=1).launch(server_name=host,server_port=7860,share=False,max_file_size='8mb',css=CSS)

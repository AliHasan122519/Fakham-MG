"""Same 6 sentences, same seeds: quality + speed + memory for one engine mode.
Usage: MODE=fp32|bf16_cap8|bf16_uncapped SVS_HOME=<home with refs/my_voice.wav + .txt> SVS_MODEL_DIR=<models> python compare_modes.py"""
import sys,os,time,gc,json,ctypes,numpy as np,psutil
mode=os.environ["MODE"]
if mode=="fp32": os.environ["SVS_BF16"]="0"
if mode=="bf16_uncapped": os.environ["ONEDNN_PRIMITIVE_CACHE_CAPACITY"]="1024"
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..')); HOME=os.environ.setdefault('SVS_HOME','/tmp/refdiag/homeUI'); os.environ.setdefault('SVS_MODEL_DIR','/tmp/refdiag/home7/models')
from svs.engine import ENGINE
from svs import asr, speaker, refaudio, render, qc
from pathlib import Path
P=psutil.Process(); gb=lambda: round(P.memory_info().rss/2**30,3)
ref=HOME+"/refs/my_voice.wav"; rt=open(HOME+"/refs/my_voice.txt",encoding="utf-8").read().strip()
r16=refaudio.audio_resample16(refaudio.decode(Path(ref))[0])
S=["خرج الصياد في الفجر إلى البحر.","كان الموج هادئًا والسماء صافية والريح خفيفة جدًا في ذلك الصباح الجميل.","وعاد.","افتتحت المدرسة أبوابها لطلابها الجدد!","ملأت الفرحة الوجوه كلها، وبدأ العام الدراسي بروح جميلة وأمل كبير.","تقع المدينة على سفح الجبل الأخضر."]
libc=ctypes.CDLL("libc.so.6")
rows=[]
for i,t in enumerate(S):
    b=gb(); t0=time.time(); w,sr=ENGINE.synth(t,ref,rt,engine="najdi",speed=1.0,nfe_step=16,seed=100+i); dt=time.time()-t0
    a=gb(); gc.collect(); libc.malloc_trim(0); r=gb()
    w=np.asarray(w,np.float32); w16=refaudio.audio_resample16(w)
    heard=asr.transcribe(w16)["text"]
    rows.append({"i":i,"chars":len(t),"audio_s":round(len(w)/sr,2),"synth_s":round(dt,1),"rtf":round(dt/(len(w)/sr),2),"ram_before":b,"ram_after":a,"ram_released":r,
                 "asr":round(asr.similarity(t,heard),3),"spk":round(speaker.similarity(r16,w16),3),"f0_st":None if render._f0_median_st(w) is None else round(render._f0_median_st(w),2)})
    print(json.dumps(rows[-1],ensure_ascii=False),flush=True)
print("SUMMARY",mode,json.dumps({"rtf_mean":round(float(np.mean([r["rtf"] for r in rows])),2),"asr_mean":round(float(np.mean([r["asr"] for r in rows])),3),
  "spk_mean":round(float(np.mean([r["spk"] for r in rows])),3),"ram_start":rows[0]["ram_before"],"ram_end_released":rows[-1]["ram_released"],"ram_peak":max(r["ram_after"] for r in rows)}))

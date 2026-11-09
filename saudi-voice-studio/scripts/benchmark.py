#!/usr/bin/env python
"""Measured benchmark of the REAL pipeline on this machine (GPU or CPU). Run in Colab on a T4 and send me the printed JSON.

    python scripts/benchmark.py            # ~1-3 min on a T4, ~8 min on CPU

Measures: device/precision, single-sentence synthesis (warm), a full multi-sentence take (text -> final master), regenerating one
sentence + merge, and the quality indicators (ASR match, speaker similarity, QC status) so speed is never reported without quality.
Writes benchmark.json into SVS_HOME. No network use beyond model downloads that already happened.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MPLBACKEND", "Agg")

TEXT = ("في قلب الصحراء الواسعة، تروى حكايات الأجداد جيلًا بعد جيل. كانت القوافل تعبر الرمال الذهبية تحت شمس لا ترحم. "
        "وكان الدليل العجوز يعرف كل نجمة وكل كثيب. لكن في تلك الليلة هبّت عاصفة لم يشهدوا مثلها من قبل! "
        "ارتفع الغبار حتى حجب السماء، وضاع الطريق. تساءل الجميع بخوف: هل سننجو؟ وقف الدليل ثابتًا وقال بهدوء: اتبعوني ولا تتفرقوا. "
        "وفي الصباح الباكر انقشع الغبار، وظهرت الواحة أمامهم كحلم جميل. ركض الأطفال نحو الماء بفرح لا يوصف. "
        "وجلس الشيوخ تحت النخيل يحمدون الله على السلامة. وهكذا انتهت الرحلة وبقيت الحكاية.")


class Sampler:
    """Samples this process' RSS and CPU% once per second (and GPU memory/utilisation when a GPU exists)."""
    def __init__(self):
        import threading
        self.rss, self.cpu, self.gpu_mem, self.gpu_util = [], [], [], []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        import subprocess
        try:
            import psutil
            p = psutil.Process()
            p.cpu_percent(None)
        except Exception:
            p = None
        while not self._stop.wait(1.0):
            if p is not None:
                self.rss.append(p.memory_info().rss / 2 ** 30)
                self.cpu.append(p.cpu_percent(None))
            try:
                o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=3).stdout.strip().split(",")
                self.gpu_mem.append(float(o[0]) / 1024)
                self.gpu_util.append(float(o[1]))
            except Exception:
                pass

    def start(self):
        self._t.start()
        return self

    def report(self, since: int = 0) -> dict:
        import numpy as np
        r = {}
        if self.rss:
            r["ram_peak_gb"] = round(max(self.rss[since:] or self.rss), 2)
            r["cpu_pct_mean"] = round(float(np.mean(self.cpu[since:] or self.cpu)), 0)
        if self.gpu_mem:
            r["gpu_mem_peak_gb"] = round(max(self.gpu_mem[since:] or self.gpu_mem), 2)
            r["gpu_util_pct_mean"] = round(float(np.mean(self.gpu_util[since:] or self.gpu_util)), 0)
        return r

    def mark(self) -> int:
        return len(self.rss)


def main() -> int:
    import numpy as np
    import torch

    sam = Sampler().start()

    from svs import auto, config, render, voices
    from svs.engine import ENGINE

    info = {"platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
            "device": ENGINE.pick_device(), "threads": torch.get_num_threads(), "bf16_cpu_autocast": ENGINE._cpu_bf16()}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 1)
    voice = voices.default_voice_id()
    info["voice"] = voice
    out: dict = {"environment": info}
    print("البيئة:", json.dumps(info, ensure_ascii=False))

    # 1) single sentence, cold then warm
    ref, ref_text, _src = voices.reference_for(voices.get_voice(voice), "najdi", None)
    nfe = 32 if info["device"] == "cuda" else 16
    s = "ذهب القوم إلى السوق في الصباح الباكر ليشتروا بعض الهدايا."
    times = []
    for i in range(3):
        if info["device"] == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        w, sr = ENGINE.synth(s, str(ref), ref_text, engine="najdi", speed=1.0, nfe_step=nfe, seed=7 + i)
        if info["device"] == "cuda":
            torch.cuda.synchronize()
        times.append((time.time() - t, len(w) / sr))
    warm = times[1:]
    out["single_sentence"] = {"nfe": nfe, "cold_s": round(times[0][0], 2), "warm_s": round(float(np.mean([a for a, _ in warm])), 2),
                              "audio_s": round(float(np.mean([b for _, b in warm])), 2),
                              "rtf_warm": round(float(np.mean([a / b for a, b in warm])), 3)}
    print("جملة واحدة:", out["single_sentence"])

    # 2) full take (text -> final master) with all QC stages
    sam_mark = sam.mark()
    marks: list = []
    t0 = time.time()
    take = auto.run(TEXT, voice=voice, progress=lambda p, m: marks.append((round(time.time() - t0, 1), round(p, 2), m)))
    total = time.time() - t0
    qc = take.get("qc") or {}
    intel = list((take.get("intelligibility") or {}).values())
    out["full_take"] = {"sentences": len(take["segments"]), "audio_s": take["seconds"], "total_s": round(total, 1),
                        "rtf_total": round(total / max(take["seconds"], 1), 3),
                        "asr_match_mean": round(float(np.mean(intel)), 3) if intel else None,
                        "asr_match_min": round(float(np.min(intel)), 3) if intel else None,
                        "qc_status": qc.get("status"), "qc": {c["id"]: c["status"] for c in qc.get("checks", [])},
                        "fixed_sentences": qc.get("fixed_sentences"), "voice_match": take.get("voice_match")}
    # stage durations from the renderer's own progress marks (generation+QC / assembly / final processing)
    def at(frac):
        for t_, p_, _m in marks:
            if p_ >= frac:
                return t_
        return round(total, 1)
    t_syn, t_asm, t_end = at(0.80), at(0.90), round(total, 1)
    out["full_take"]["stages_s"] = {"synthesis_and_checks": t_syn, "assembly_and_cleanup": round(t_asm - t_syn, 1),
                                    "final_processing_and_export": round(t_end - t_asm, 1)}
    out["full_take"]["resources"] = sam.report(sam_mark)
    print("نص كامل:", json.dumps(out["full_take"], ensure_ascii=False))

    # 3) regenerate one sentence + merge (everything else comes from cache)
    seg_id = take["segments"][len(take["segments"]) // 2]["id"]
    sam_mark2 = sam.mark()
    t1 = time.time()
    take2 = render.regenerate(take["id"], seg_id, new_seed=True)
    out["regenerate_one_sentence_and_merge"] = {"segment": seg_id, "total_s": round(time.time() - t1, 1),
                                                "qc_status": (take2.get("qc") or {}).get("status")}
    out["regenerate_one_sentence_and_merge"]["resources"] = sam.report(sam_mark2)
    print("إعادة توليد جملة + دمج:", out["regenerate_one_sentence_and_merge"])

    # 4) final processing only (master chain + cleanup) on the finished mix
    from svs import audio
    d = config.DATA / "takes" / take2["id"]
    t2 = time.time()
    m = audio.metrics(d / "master.wav")
    out["final_processing_check_s"] = round(time.time() - t2, 2)
    out["master_metrics"] = {k: m.get(k) for k in ("duration_s", "lufs", "true_peak_db", "lra")}

    out["overall_resources"] = sam.report()
    p = config.DATA / "benchmark.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n=== النتيجة (أرسلها لي) ===\n" + json.dumps(out, ensure_ascii=False, indent=1))
    print("حُفظت في", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

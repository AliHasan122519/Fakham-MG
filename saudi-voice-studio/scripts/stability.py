#!/usr/bin/env python
"""Repeated-use stability test on the REAL pipeline: N consecutive generations (+ a sentence regeneration every other run).

    python scripts/stability.py [N]        # default 6 runs of 3 short sentences

Checks: every run succeeds, no QC 'fail', time does not drift upward, RAM/threads/open-files do not grow run after run
(leak detection), disk per take stays bounded. Writes stability.json into SVS_HOME.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MPLBACKEND", "Agg")

POOL = [
    "خرج الصياد في الفجر إلى البحر. كان الموج هادئًا والسماء صافية. وعاد عند الغروب محمّلًا بالخير.",
    "افتتحت المدرسة أبوابها لطلابها الجدد! ملأت الفرحة الوجوه. وبدأ العام الدراسي بروح جميلة.",
    "تقع المدينة على سفح الجبل الأخضر. تحيط بها المزارع من كل جانب. ويقصدها الزوار في كل موسم.",
    "سأل الطفل جدّه عن النجوم. ابتسم الجد وأشار إلى السماء. وقال بهدوء: لكل نجمة حكاية.",
    "أعلنت الجهة المختصة موعد التسجيل. يبدأ الاثنين القادم في جميع المناطق. وتنتهي المهلة بعد أسبوعين.",
    "هبّت الريح فجأة في الوادي. اهتزت الخيام وتطاير الغبار. ثم هدأ كل شيء كأن شيئًا لم يكن.",
    "قرأت كتابًا عن تاريخ القوافل. أدهشتني دقة الأوصاف. وقررت أن أزور تلك الطرق بنفسي.",
    "في الصباح الباكر تفتّحت الأزهار. وغنّت العصافير على الأغصان. فبدا العالم أجمل من أي وقت.",
]


def main() -> int:
    import numpy as np
    import psutil

    from svs import auto, config, render, studio, voices

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    proc = psutil.Process()
    voice = voices.default_voice_id()
    rows = []
    for i in range(n):
        txt = POOL[i % len(POOL)]
        t0 = time.time()
        ok, err, take = True, None, None
        try:
            take = studio.take_view(auto.run(txt, voice=voice))
            if i % 2 == 1:                                   # also exercise regeneration + merge
                take = studio.take_view(render.regenerate(take["id"], take["segments"][1]["id"], new_seed=True))
        except Exception as e:                               # a crash is a failed run, reported — not hidden
            ok, err = False, f"{type(e).__name__}: {e}"
        dt = time.time() - t0
        row = {"run": i + 1, "ok": ok, "error": err, "seconds": round(dt, 1), "ram_gb": round(proc.memory_info().rss / 2 ** 30, 2),
               "threads": proc.num_threads(), "open_files": len(proc.open_files()),
               "audio_s": None if take is None else take.get("seconds"),
               "qc": None if take is None else (take.get("qc") or {}).get("status"),
               "asr_min": None if take is None else (min((take.get("intelligibility") or {"x": 1}).values())),
               "regenerated": i % 2 == 1}
        if take is not None:
            row["take_mb"] = round(sum(f.stat().st_size for f in (config.DATA / "takes" / take["id"]).glob("*")) / 2 ** 20, 1)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    rss = [r["ram_gb"] for r in rows]
    ratio = [r["seconds"] / max(1.0, (r["audio_s"] or 1)) for r in rows if r["ok"] and r["audio_s"] and not r["regenerated"]]
    summary = {
        "runs": n, "failed": sum(1 for r in rows if not r["ok"]),
        "qc_fail": sum(1 for r in rows if r["qc"] == "fail"),
        "ram_growth_gb_after_warmup": round(rss[-1] - rss[min(1, len(rss) - 1)], 2),
        "threads_growth": rows[-1]["threads"] - rows[min(1, len(rows) - 1)]["threads"],
        "open_files_growth": rows[-1]["open_files"] - rows[min(1, len(rows) - 1)]["open_files"],
        "time_per_audio_second_first_vs_last": [round(ratio[0], 2), round(ratio[-1], 2)] if len(ratio) >= 2 else None,
    }
    summary["verdict"] = ("مستقر" if summary["failed"] == 0 and summary["qc_fail"] == 0 and summary["ram_growth_gb_after_warmup"] < 0.5
                          and summary["threads_growth"] <= 2 and summary["open_files_growth"] <= 3 else "يحتاج مراجعة")
    out = {"rows": rows, "summary": summary}
    (config.DATA / "stability.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n=== الملخص ===\n" + json.dumps(summary, ensure_ascii=False, indent=1))
    return 0 if summary["verdict"] == "مستقر" else 1


if __name__ == "__main__":
    raise SystemExit(main())

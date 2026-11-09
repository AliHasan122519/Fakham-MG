#!/usr/bin/env python
"""Long, realistic usage session against the REAL server process (HTTP API + real MCP JSON-RPC), not the engine alone.

    python scripts/session_test.py [ROUNDS]          # default 2 rounds of the full schedule (~40-60 min on CPU)

Starts `python -m svs.mcp_server --http` exactly like the Colab notebook does, then repeats a schedule of: short / long / varied-length
generations, several consecutive single-sentence regenerations (+merge), draft -> finalize, MCP generate/regenerate/get_job, and
WAV+MP3 downloads with integrity checks. After every step it records the SERVER process' RAM (before/after/idle-after-release),
threads, open files, child processes and bytes on disk, so a leak shows up as a trend across rounds. Writes session_test.json.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = int(os.environ.get("SVS_TEST_PORT", "8791"))
TOKEN = "session-test-token"
BASE = f"http://127.0.0.1:{PORT}"

SHORT = "عاد الصياد إلى بيته."
MEDIUM = "افتتحت المدرسة أبوابها لطلابها الجدد! ملأت الفرحة الوجوه. وبدأ العام الدراسي بروح جميلة وأمل كبير."
LONG = ("في قلب الصحراء الواسعة، تروى حكايات الأجداد جيلًا بعد جيل. كانت القوافل تعبر الرمال الذهبية تحت شمس لا ترحم. "
        "وكان الدليل العجوز يعرف كل نجمة وكل كثيب. لكن في تلك الليلة هبّت عاصفة لم يشهدوا مثلها من قبل! "
        "ارتفع الغبار حتى حجب السماء، وضاع الطريق. تساءل الجميع بخوف: هل سننجو؟ "
        "وفي الصباح الباكر انقشع الغبار، وظهرت الواحة أمامهم كحلم جميل. وجلس الشيوخ تحت النخيل يحمدون الله على السلامة.")


def call(path, body=None, raw=False, timeout=900):
    u = BASE + path + ("&" if "?" in path else "?") + "token=" + TOKEN
    req = urllib.request.Request(u, data=None if body is None else json.dumps(body).encode(),
                                 headers={"content-type": "application/json", "accept": "application/json, text/event-stream"} if body is not None else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = r.read()
    return d if raw else json.loads(d)


def mcp(name, args, timeout=900):
    r = call("/mcp", {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": args}}, timeout=timeout)
    res = r["result"]
    if res.get("isError"):
        raise RuntimeError(str(res)[:300])
    return res.get("structuredContent") or json.loads(res["content"][0]["text"])


def wait_job(jid):
    while True:
        j = call("/api/jobs/" + jid)
        if j["state"] == "error":
            raise RuntimeError(j.get("error"))
        if j["state"] == "done":
            return j["take"]
        time.sleep(3)


def mcp_wait(view):
    while view.get("state") not in ("done", "error"):
        time.sleep(4)
        view = mcp("get_job", {"job_id": view["id"]})
    if view["state"] == "error":
        raise RuntimeError(view.get("error"))
    return view


def check_files(take_id: str, out: Path) -> dict:
    import soundfile as sf
    wav, mp3 = out / f"{take_id}.wav", out / f"{take_id}.mp3"
    wav.write_bytes(call(f"/files/{take_id}/master.wav", raw=True))
    mp3.write_bytes(call(f"/files/{take_id}/master.mp3", raw=True))
    info = sf.info(str(wav))
    x, _ = sf.read(str(wav), dtype="float32")
    pr = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration,bit_rate", "-of", "json", str(mp3)],
                                   capture_output=True, text=True).stdout)["format"]
    ok = (info.subtype == "PCM_24" and info.samplerate == 24000 and float(abs(x).max()) < 0.999 and int(pr["bit_rate"]) >= 300000
          and abs(float(pr["duration"]) - info.duration) < 0.3)
    return {"wav": f"{info.subtype}/{info.samplerate}Hz/{info.duration:.1f}s", "peak": round(float(abs(x).max()), 3),
            "mp3_kbps": int(pr["bit_rate"]) // 1000, "mp3_s": round(float(pr["duration"]), 1), "ok": ok}


def main() -> int:
    import psutil
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    env = dict(os.environ, SVS_TOKEN=TOKEN, SVS_PORT=str(PORT), SVS_HOST="127.0.0.1", MPLBACKEND="Agg")
    env.setdefault("SVS_PUBLIC_URL", f"http://127.0.0.1:{PORT}")
    srv = subprocess.Popen([sys.executable, "-m", "svs.mcp_server", "--http"], cwd=ROOT, env=env,
                           stdout=open("/tmp/svs_session_server.log", "w"), stderr=subprocess.STDOUT)
    out = Path(os.environ.get("SVS_TEST_OUT", "/tmp/svs_session_out"))
    out.mkdir(parents=True, exist_ok=True)
    try:
        for _ in range(120):
            try:
                if json.load(urllib.request.urlopen(BASE + "/api/health", timeout=3)).get("ok"):
                    break
            except Exception:
                time.sleep(1)
        proc = psutil.Process(srv.pid)
        import gc
        rows = []

        def snap(step, **extra):
            kids = proc.children(recursive=True)
            ofs = proc.open_files()
            r = {"step": step, "ram_gb": round(proc.memory_info().rss / 2 ** 30, 2), "threads": proc.num_threads(),
                 "open_files": len(ofs), "open_paths": sorted({f.path for f in ofs})[-10:], "children": len(kids), **extra}
            rows.append(r)
            print(json.dumps(r, ensure_ascii=False), flush=True)
            return r

        snap("start")
        take = None
        failed = []
        for rd in range(1, rounds + 1):
            def step(name, fn):
                nonlocal take
                t0 = time.time()
                try:
                    res = fn()
                except Exception as e:
                    failed.append({"round": rd, "step": name, "error": f"{type(e).__name__}: {e}"})
                    snap(f"r{rd}:{name}:FAILED", error=str(e)[:160])
                    return None
                files = check_files(res["id"], out) if isinstance(res, dict) and res.get("id") else None
                if files and not files["ok"]:
                    failed.append({"round": rd, "step": name, "error": f"file integrity {files}"})
                snap(f"r{rd}:{name}", seconds=round(time.time() - t0, 1), take=(res or {}).get("id"), files=files,
                     qc=((res or {}).get("qc") or {}).get("status") if isinstance(res, dict) else None, draft=(res or {}).get("draft"))
                return res

            take_s = step("auto_short", lambda: wait_job(call("/api/auto", {"text": SHORT})["job"]))
            take_l = step("auto_long_http", lambda: wait_job(call("/api/auto", {"text": LONG})["job"]))
            # three consecutive single-sentence regenerations + merge on the same take chain
            cur = take_l
            for k, idx in enumerate((1, 3, 5)):
                if cur:
                    cur = step(f"regen{k + 1}", lambda c=cur, i=idx: wait_job(call(f"/api/takes/{c['id']}/regenerate",
                               {"segment_id": c["segments"][i]["id"], "new_seed": True})["job"])) or cur
            # draft -> finalize
            dr = step("draft_medium", lambda: wait_job(call("/api/auto", {"text": MEDIUM, "draft": True})["job"]))
            if dr:
                step("finalize", lambda: wait_job(call(f"/api/takes/{dr['id']}/finalize", {})["job"]))
            # real MCP protocol: generate, poll, regenerate
            def mcp_gen():
                v = mcp_wait(mcp("generate_voiceover", {"text": MEDIUM, "wait_seconds": 20}))
                return {"id": v["take_id"], "qc": {"status": v.get("qc")}, "draft": v.get("draft")}
            mt = step("mcp_generate", mcp_gen)
            if mt:
                def mcp_regen():
                    segs = call(f"/api/takes/{mt['id']}")["segments"]
                    v = mcp_wait(mcp("regenerate_sentence", {"take_id": mt["id"], "segment_id": segs[1]["id"], "wait_seconds": 20}))
                    return {"id": v["take_id"], "qc": {"status": v.get("qc")}}
                step("mcp_regenerate", mcp_regen)
            # controls (studio settings) path
            step("auto_controls", lambda: wait_job(call("/api/auto", {"text": MEDIUM, "controls": {"energy": 0.5, "warmth_db": 1.0}})["job"]))
            gc.collect()
            time.sleep(20)                                      # idle: the server must not keep growing while nothing runs
            snap(f"r{rd}:idle_after_round")
        # leak verdict: compare the idle RSS at the end of each round
        idle = [r["ram_gb"] for r in rows if r["step"].endswith("idle_after_round")]
        summary = {"rounds": rounds, "steps": len(rows) - 1, "failed": failed,
                   "idle_ram_gb_per_round": idle,
                   "ram_peak_gb": max(r["ram_gb"] for r in rows),
                   "ram_growth_between_last_two_rounds_gb": round(idle[-1] - idle[-2], 2) if len(idle) >= 2 else None,
                   "threads_start_end": [rows[0]["threads"], rows[-1]["threads"]],
                   "open_files_start_end": [rows[0]["open_files"], rows[-1]["open_files"]],
                   "children_max": max(r["children"] for r in rows)}
        summary["verdict"] = ("مستقر" if not failed and (summary["ram_growth_between_last_two_rounds_gb"] is None or
                              summary["ram_growth_between_last_two_rounds_gb"] < 0.3) and summary["open_files_start_end"][1] - summary["open_files_start_end"][0] <= 3
                              else "يحتاج مراجعة")
        res = {"rows": rows, "summary": summary}
        (Path(os.environ.get("SVS_HOME", "/tmp")) / "session_test.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print("\n=== الملخص ===\n" + json.dumps(summary, ensure_ascii=False, indent=1))
        return 0 if summary["verdict"] == "مستقر" else 1
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=15)
        except Exception:
            srv.kill()


if __name__ == "__main__":
    raise SystemExit(main())

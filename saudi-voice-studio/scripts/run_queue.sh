#!/bin/bash
# Resumable job queue for the long measurements. SAFE TO RE-RUN after the sandbox/container is re-created:
# finished jobs leave `docs/results/done/<job>.flag` + their logs in docs/results/ (committed to git, so they survive a new container);
# a re-run skips finished jobs and continues with the next one.
#   SVS_HOME=<data home> SVS_MODEL_DIR=<models> scripts/run_queue.sh
set -u
APP="$(cd "$(dirname "$0")/.." && pwd)"
R="$APP/docs/results"; mkdir -p "$R/done"
PY="${SVS_PY:-/tmp/svs-venv-test/bin/python}"
export SVS_HOME="${SVS_HOME:-/tmp/refdiag/homeUI}" SVS_MODEL_DIR="${SVS_MODEL_DIR:-/tmp/refdiag/home7/models}"
job() { name=$1; shift; [ -f "$R/done/$name.flag" ] && { echo "skip $name (done)" >> "$R/queue.log"; return 0; }
        echo "$(date +%T) START $name" >> "$R/queue.log"
        if "$@" > "$R/$name.log" 2>&1; then touch "$R/done/$name.flag"; echo "$(date +%T) END $name ok" >> "$R/queue.log"
        else echo "$(date +%T) END $name FAILED rc=$?" >> "$R/queue.log"; fi; }
cmp_mode() { m=$1; MODE=$m "$PY" "$APP/scripts/experiments/compare_modes.py" | tee "$R/cmp_$m.out" | grep -q "^SUMMARY"; }
job cmp_bf16_cap8 cmp_mode bf16_cap8
job cmp_fp32 cmp_mode fp32
job cmp_bf16_uncapped cmp_mode bf16_uncapped
job session "$PY" "$APP/scripts/session_test.py" 2
echo "QUEUE_DONE $(date +%T)" >> "$R/queue.log"

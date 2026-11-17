#!/usr/bin/env bash
# DFSI — build and run, in one script.
#
# From a bare checkout to a dashboard: venv + deps, UI deps, Memory (PySpark),
# Forecaster (MLflow), API, UI. Every stage is skipped when its output is
# already there, so a second run starts the API in seconds. Ctrl-C stops
# everything it started.
#
#   scripts/run.sh                 build what's missing, then serve
#   scripts/run.sh --rebuild       force Memory + training again
#   scripts/run.sh --test          run the test suite before serving
#   scripts/run.sh --no-ui         API only (foreground)
#   scripts/run.sh --build-only    build, don't serve
#
# Set ANTHROPIC_API_KEY first to run the explainers on Claude; without it they
# fall back to the offline rule engine, which is a supported mode.
set -eu
cd "$(dirname "$0")/.."

REBUILD=0
RUN_TESTS=0
START_UI=1
SERVE=1
MEMORY_ARGS=""
TRAIN_ARGS=""

while [ $# -gt 0 ]; do
  case "$1" in
    --rebuild)    REBUILD=1 ;;
    --test)       RUN_TESTS=1 ;;
    --no-ui)      START_UI=0 ;;
    --build-only) SERVE=0 ;;
    # Forwarded to build_memory.sh. Checked here so a missing value reports
    # itself rather than tripping `set -u` on $2.
    --days|--env)
      [ $# -ge 2 ] || { echo "run.sh: $1 needs a value" >&2; exit 2; }
      MEMORY_ARGS="$MEMORY_ARGS $1 $2"
      # The time split needs four weeks (warm-up, fit, calibration, test).
      if [ "$1" = --days ] && [ "$2" -lt 28 ]; then TRAIN_ARGS="--split scenario"; fi
      shift ;;
    -h|--help)    sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "run.sh: unknown option '$1' (try --help)" >&2; exit 2 ;;
  esac
  shift
done

# ---------------------------------------------------------------- 1. Python env
# env.sh wants .venv/bin/python to exist, so the venv is created before it is
# sourced. python3.12 is what the pins in requirements.txt were resolved against.
if [ ! -x .venv/bin/python ]; then
  PY312="$(command -v python3.12 || true)"
  if [ -z "$PY312" ]; then
    echo "DFSI: python3.12 not on PATH. Install it (e.g. 'brew install python@3.12')." >&2
    exit 1
  fi
  echo "==> Creating .venv (python3.12)"
  "$PY312" -m venv .venv
fi

# Reinstall only when requirements.txt has actually moved since the last install.
REQ_SHA="$(shasum requirements.txt | cut -d' ' -f1)"
REQ_STAMP=".venv/.dfsi-requirements-sha"
if [ "$REQ_SHA" != "$(cat "$REQ_STAMP" 2>/dev/null || true)" ]; then
  echo "==> Installing Python dependencies (a few minutes the first time)"
  .venv/bin/python -m pip install --quiet --upgrade pip
  .venv/bin/python -m pip install --quiet -r requirements.txt
  printf '%s' "$REQ_SHA" > "$REQ_STAMP"
else
  echo "==> Python dependencies up to date"
fi

# shellcheck source=scripts/env.sh
. scripts/env.sh
echo "==> JAVA_HOME=${JAVA_HOME:-none (only needed to build Memory)}"
echo "==> DFSI_HOME=$DFSI_HOME"

# -------------------------------------------------------------------- 2. UI deps
if [ "$START_UI" = 1 ] || [ "$SERVE" = 0 ]; then
  if ! command -v npm >/dev/null 2>&1; then
    if [ "$START_UI" = 1 ]; then
      echo "DFSI: npm not found; install Node 20+ or pass --no-ui." >&2
      exit 1
    fi
  elif [ ! -d ui/node_modules ] || [ ui/package-lock.json -nt ui/node_modules ]; then
    echo "==> Installing UI dependencies"
    (cd ui && npm install)
  else
    echo "==> UI dependencies up to date"
  fi
fi

# --------------------------------------------------------------- 3. Memory layer
# A complete build ends with the feature table's _SUCCESS marker and the replay
# catalogue. Checking only for data/features misses a half-written table, and
# the catalogue that older checkouts tracked in git (a pull deletes it).
if [ "$REBUILD" = 1 ] \
  || [ ! -f "$DFSI_HOME/data/features/_SUCCESS" ] \
  || [ ! -f "$DFSI_HOME/data/replay/catalogue.json" ]; then
  echo "==> Memory layer: PySpark over the logbook (~20 min for 28 days)"
  # shellcheck disable=SC2086  # MEMORY_ARGS is a deliberate word list
  scripts/build_memory.sh $MEMORY_ARGS
else
  echo "==> Memory layer present; skipping (--rebuild to force)"
fi

# ----------------------------------------------------------------- 4. Forecaster
if [ "$REBUILD" = 1 ] || [ ! -f "$DFSI_HOME/models/forecaster.pkl" ]; then
  echo "==> Forecaster: train, calibrate, register in MLflow"
  # shellcheck disable=SC2086  # TRAIN_ARGS is a deliberate word list
  scripts/train.sh $TRAIN_ARGS
else
  echo "==> Model present; skipping (--rebuild to force)"
fi

# ---------------------------------------------------------------------- 5. Tests
if [ "$RUN_TESTS" = 1 ]; then
  echo "==> Tests"
  .venv/bin/python -m pytest -q
fi

if [ "$SERVE" = 0 ]; then
  echo "==> Build complete (--build-only); not serving."
  exit 0
fi

# ------------------------------------------------------------------------ 6. API
PORT="${DFSI_PORT:-8000}"
UI_PORT="${DFSI_UI_PORT:-5173}"

for p in "$PORT" $([ "$START_UI" = 1 ] && echo "$UI_PORT"); do
  # -sTCP:LISTEN: only a listener blocks the bind. Without it, a client socket
  # left on the port (e.g. VS Code's port forwarder, often CLOSED) counts too.
  if lsof -ti "tcp:$p" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "DFSI: port $p is already in use. Stop that process, or set DFSI_PORT / DFSI_UI_PORT." >&2
    exit 1
  fi
done

if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
  echo "==> Agents: Claude (ANTHROPIC_API_KEY set)"
else
  echo "==> Agents: offline rule engine (set ANTHROPIC_API_KEY for Claude)"
fi

if [ "$START_UI" = 0 ]; then
  echo "==> API on http://127.0.0.1:$PORT  (docs at /docs)"
  exec .venv/bin/python -m uvicorn dfsi.api.app:app --host 127.0.0.1 --port "$PORT"
fi

# With the UI in the foreground the API goes to a log file, so Vite's output
# stays readable; the log is dumped if the API never comes up. No --reload here:
# the reloader forks a worker that would outlive a kill of this PID.
mkdir -p logs
API_LOG="logs/api.log"
echo "==> API on http://127.0.0.1:$PORT  (log: $API_LOG)"
.venv/bin/python -m uvicorn dfsi.api.app:app --host 127.0.0.1 --port "$PORT" >"$API_LOG" 2>&1 &
API_PID=$!

VITE_PID=""
stop_all() {
  trap - EXIT INT TERM
  [ -n "$VITE_PID" ] && kill "$VITE_PID" 2>/dev/null
  kill "$API_PID" 2>/dev/null
  wait "$API_PID" 2>/dev/null
  [ -n "$VITE_PID" ] && wait "$VITE_PID" 2>/dev/null
  return 0
}
trap stop_all EXIT INT TERM

# Vite proxies /api to the API, so it must be answering before the browser opens.
for _ in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then break; fi
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "DFSI: the API exited during startup. Last lines of $API_LOG:" >&2
    tail -n 30 "$API_LOG" >&2
    exit 1
  fi
  sleep 0.5
done

if ! curl -sf "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then
  echo "DFSI: the API did not answer /api/health within 30s. Last lines of $API_LOG:" >&2
  tail -n 30 "$API_LOG" >&2
  exit 1
fi

echo "==> Dashboard on http://localhost:$UI_PORT  (Ctrl-C stops both)"
# Vite is launched directly rather than through `npm run dev`: npm would be the
# child we hold a PID for, and killing it leaves vite orphaned holding the port.
# --host 127.0.0.1 pins it to IPv4 — left to itself it binds [::1] only, where
# the port check above and a 127.0.0.1 URL both miss it.
# DFSI_API_URL is what vite.config.ts points the /api proxy at.
DFSI_API_URL="http://127.0.0.1:$PORT" \
  ui/node_modules/.bin/vite ui --config ui/vite.config.ts \
  --host 127.0.0.1 --port "$UI_PORT" --strictPort &
VITE_PID=$!

# Not exec'd anywhere above: this shell stays alive to own the trap, so that one
# Ctrl-C takes down the API as well as the UI.
wait "$VITE_PID" 2>/dev/null || true
VITE_PID=""

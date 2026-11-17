#!/usr/bin/env bash
# Memory layer — PySpark over the synthetic logbook.
#
# Generates four weeks (by default) of ANL-shaped 10-second windows, computes
# past-only hour-of-week baselines, builds the feature table and the
# 120-second onset label, and writes the small per-scenario JSON replays the
# UI streams (from the held-out test week only).
#
# Needs a JDK 11/17/21; scripts/env.sh finds one and fails loudly if it cannot.
set -eu
cd "$(dirname "$0")/.."
export DFSI_REQUIRE_JDK=1
# shellcheck source=scripts/env.sh
. scripts/env.sh

echo "DFSI memory: JAVA_HOME=$JAVA_HOME"
echo "DFSI memory: writing under $DFSI_HOME/data"
exec "$PYTHON_BIN" -m dfsi.memory.pipeline "$@"

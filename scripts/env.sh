#!/usr/bin/env bash
# Shared environment for every DFSI script.
#
# PySpark 3.5 cannot run on JDK 23+ (javax.security.auth.Subject.getSubject was
# removed, which Hadoop's UserGroupInformation still calls). Pin JDK 17 — the
# same major version netsec-ddos-shared-state-2 itself builds against.
#
# Safe to `source` from bash or zsh, and safe to execute. zsh does not set
# BASH_SOURCE when sourcing, so the script's own path is resolved from
# whichever of the two the shell provides.
set -eu

if [ -n "${BASH_SOURCE:-}" ]; then
  _dfsi_self="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then
  # shellcheck disable=SC2296
  _dfsi_self="${(%):-%x}"
else
  _dfsi_self="$0"
fi

DFSI_ROOT="$(cd "$(dirname "$_dfsi_self")/.." && pwd)"
unset _dfsi_self
export DFSI_ROOT
export DFSI_HOME="${DFSI_HOME:-$DFSI_ROOT}"
export PYTHON_BIN="${PYTHON_BIN:-$DFSI_ROOT/.venv/bin/python}"

# True when $1 is a JDK home whose major version is 11, 17 or 21. JDK 8 is
# rejected too: it starts, then fails inside Spark with an opaque Py4J error.
_dfsi_jdk_ok() {
  [[ -n "$1" && -x "$1/bin/java" ]] &&
    "$1/bin/java" -version 2>&1 | grep -qE 'version "(11|17|21)[."]'
}

if ! _dfsi_jdk_ok "${JAVA_HOME:-}"; then
  [[ -n "${JAVA_HOME:-}" ]] &&
    echo "DFSI: ignoring JAVA_HOME=$JAVA_HOME (not JDK 11/17/21)" >&2
  unset JAVA_HOME
  for candidate in \
    "$(/usr/libexec/java_home -v 17 2>/dev/null || true)" \
    "$(/usr/libexec/java_home -v 21 2>/dev/null || true)" \
    "$(/usr/libexec/java_home -v 11 2>/dev/null || true)" \
    /opt/homebrew/opt/openjdk@17 /usr/local/opt/openjdk@17 \
    /usr/lib/jvm/java-17-openjdk-amd64 /usr/lib/jvm/java-17-openjdk-arm64; do
    if _dfsi_jdk_ok "$candidate"; then
      export JAVA_HOME="$candidate"
      break
    fi
  done
fi

unset -f _dfsi_jdk_ok

# Only the Memory stage (PySpark) needs a JDK; build_memory.sh sets
# DFSI_REQUIRE_JDK=1. Training, the API and the UI run without one.
if [[ -n "${JAVA_HOME:-}" ]]; then
  export PATH="$JAVA_HOME/bin:$PATH"
elif [[ "${DFSI_REQUIRE_JDK:-0}" == "1" ]]; then
  echo "DFSI: no JDK 11/17/21 found. PySpark 3.5 runs on neither JDK 8 nor 23+." >&2
  echo "      Install one, e.g. 'brew install openjdk@17', or set JAVA_HOME." >&2
  exit 1
fi
export PYSPARK_PYTHON="$PYTHON_BIN"
export PYSPARK_DRIVER_PYTHON="$PYTHON_BIN"

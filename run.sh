#!/usr/bin/env bash
# Bootstrap an isolated Python environment, then run the checker from any cwd.
set -Eeuo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
VENV_DIR="${KEYCHECKER_VENV:-"$SCRIPT_DIR/.venv"}"
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"

log() { printf '[keychecker] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

[[ -f "$SCRIPT_DIR/main.py" && -f "$REQUIREMENTS" ]] ||
    die 'Keep run.sh next to main.py and requirements.txt.'

supported_python() {
    "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' >/dev/null 2>&1
}

install_system_python() {
    local -a privilege=()
    if (( EUID != 0 )); then
        command -v sudo >/dev/null 2>&1 ||
            die 'Python 3.9+ with venv is required. Ask an administrator to install it, then rerun this file.'
        privilege=(sudo)
    fi

    log 'Installing missing Python/venv support using the system package manager.'
    if command -v apt-get >/dev/null 2>&1; then
        "${privilege[@]}" apt-get update >&2 || return 1
        "${privilege[@]}" apt-get install -y python3 python3-venv >&2
    elif command -v dnf >/dev/null 2>&1; then
        "${privilege[@]}" dnf install -y python3 python3-pip >&2
    elif command -v yum >/dev/null 2>&1; then
        "${privilege[@]}" yum install -y python3 python3-pip >&2
    elif command -v apk >/dev/null 2>&1; then
        "${privilege[@]}" apk add python3 py3-pip >&2
    else
        die 'No supported package manager found. Install Python 3.9+ with venv support, then rerun this file.'
    fi
}

select_python() {
    local candidate fallback=''
    # These versions have wheels for the project's existing dependency pins.
    for candidate in python3.12 python3.11 python3.10 python3.9 python3; do
        if command -v "$candidate" >/dev/null 2>&1 && supported_python "$candidate"; then
            if "$candidate" -c 'import venv, ensurepip' >/dev/null 2>&1; then
                PYTHON_BIN="$(command -v "$candidate")"
                return 0
            fi
            [[ -n "$fallback" ]] || fallback="$(command -v "$candidate")"
        fi
    done
    if [[ -n "$fallback" ]]; then
        PYTHON_BIN="$fallback"
        return 0
    fi
    return 1
}

if [[ -e "$VENV_DIR" || -L "$VENV_DIR" ]]; then
    [[ -f "$VENV_DIR/pyvenv.cfg" && -x "$VENV_DIR/bin/python" ]] ||
        die "Existing path is not a usable Linux virtual environment: $VENV_DIR. Choose another directory with KEYCHECKER_VENV."
else
    PYTHON_BIN="${KEYCHECKER_PYTHON:-}"
    if [[ -n "$PYTHON_BIN" ]]; then
        supported_python "$PYTHON_BIN" ||
            die 'KEYCHECKER_PYTHON must point to a working Python 3.9+ executable.'
    elif ! select_python; then
        install_system_python || die 'Could not install Python. Check the package manager output above.'
        select_python || die 'Python 3.9+ is still unavailable. Set KEYCHECKER_PYTHON to a compatible interpreter.'
    fi

    # Check ensurepip before creating the directory so a missing OS venv package
    # does not leave a half-created environment behind.
    if ! "$PYTHON_BIN" -c 'import venv, ensurepip' >/dev/null 2>&1; then
        if [[ -n "${KEYCHECKER_PYTHON:-}" ]]; then
            die 'The selected Python needs venv/ensurepip support (e.g. python3.12-venv on Debian/Ubuntu).'
        fi
        install_system_python || die 'Could not install venv support.'
        select_python || die 'Python is unavailable after installation.'
        "$PYTHON_BIN" -c 'import venv, ensurepip' >/dev/null 2>&1 ||
            die 'Install the venv package matching your Python version (e.g. python3.12-venv), then rerun.'
    fi
    log "Creating virtual environment: $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR" >&2 ||
        die 'Virtual environment creation failed. Check disk space, permissions and Python venv support.'
fi

VENV_PYTHON="$VENV_DIR/bin/python"
supported_python "$VENV_PYTHON" || die 'The virtual environment requires Python 3.9+.'
"$VENV_PYTHON" -c 'import sys; sys.exit(0 if sys.prefix != sys.base_prefix else 1)' ||
    die 'The selected interpreter is not isolated in a virtual environment.'

if ! "$VENV_PYTHON" -m pip --version >/dev/null 2>&1; then
    log 'Restoring pip inside the virtual environment.'
    "$VENV_PYTHON" -m ensurepip --upgrade >&2 || die 'Could not bootstrap pip inside the virtual environment.'
fi
# --dry-run first appeared in pip 22.2. Upgrade only when needed, in the venv.
if ! "$VENV_PYTHON" -c 'import pip, sys; sys.exit(0 if tuple(map(int, pip.__version__.split(".")[:2])) >= (22, 2) else 1)' >/dev/null 2>&1; then
    log 'Updating pip to support offline dependency checks.'
    "$VENV_PYTHON" -m pip install --disable-pip-version-check --no-input 'pip>=22.2' >&2 ||
        die 'Could not update pip. Check network access and package index settings.'
fi

log 'Checking Python dependencies.'
# A dry run checks exact requirement constraints, markers and transitive deps.
# --no-index avoids contacting PyPI when the environment is already ready.
dependencies_ready() {
    "$VENV_PYTHON" -m pip install --disable-pip-version-check --no-input --dry-run --no-index \
        --quiet --report - -r "$REQUIREMENTS" 2>/dev/null |
        "$VENV_PYTHON" -c 'import json, sys; sys.exit(0 if json.load(sys.stdin)["install"] == [] else 1)' 2>/dev/null
}
if ! dependencies_ready ||
   ! "$VENV_PYTHON" -m pip check >/dev/null 2>&1; then
    log 'Installing missing or incompatible Python dependencies.'
    "$VENV_PYTHON" -m pip install --disable-pip-version-check --no-input -r "$REQUIREMENTS" >&2 ||
        die 'Dependency installation failed. Check network access and the error above; use Python 3.12 if an older dependency cannot build on newer Python.'
fi

"$VENV_PYTHON" -m pip check >&2 || die 'The virtual environment has conflicting or missing dependencies.'
"$VENV_PYTHON" -c 'import aiohttp, boto3, botocore, requests, certifi; from google.oauth2 import service_account; from google.auth.transport.requests import Request' ||
    die 'A dependency cannot be imported. Choose a fresh directory with KEYCHECKER_VENV and rerun.'

log 'Starting keychecker.'
# Preserve the caller's cwd, stdin, arguments and the checker's exit status.
exec "$VENV_PYTHON" "$SCRIPT_DIR/main.py" "$@"

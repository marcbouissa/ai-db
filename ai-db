#!/bin/bash
# Wrapper script to run ai-db using the .venv in the installation directory

set -euo pipefail

# Get the installation directory (where this script is located)
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Path to the virtual environment's python
VENV_PYTHON="${INSTALL_DIR}/.venv/bin/python"

# Check if the venv python exists
if [[ ! -x "${VENV_PYTHON}" ]]; then
    echo "Error: Virtual environment not found at ${INSTALL_DIR}/.venv" >&2
    echo "Please run 'python -m venv .venv && .venv/bin/pip install -e .' in ${INSTALL_DIR}" >&2
    exit 1
fi

# Execute ai-db using the venv's python
exec "${VENV_PYTHON}" -m ai_db.cli "$@"
#!/bin/bash
# Set up node + Claude Code on a SNAP compute node.
#
# Everything lands on that node's local disk (/lfs/<host>/0/$USER), never AFS:
# the AFS home is a 5 GB quota, and Claude Code ships as a Bun single-file
# executable that mmaps itself at startup, which AFS does not serve reliably.
# A truncated or badly mapped binary shows up as "Bus error" on launch.
#
# Idempotent — safe to re-run. Run once per node (skampere1/2/3, hyperturing1/2);
# each gets its own local install, since /lfs is per-machine.
#
#   bash scripts/setup_claude_node.sh
#
# To update Claude Code later, on that node:
#   npm install -g @anthropic-ai/claude-code@latest
set -euo pipefail

NODE_VERSION=24
LFS_ROOT="/lfs/$(hostname -s)/0/$USER"

if [ ! -d "$LFS_ROOT" ]; then
    echo "ERROR: $LFS_ROOT does not exist. Is this a SNAP node with local scratch?" >&2
    exit 1
fi

export NVM_DIR="$LFS_ROOT/nvm"
export TMPDIR="$LFS_ROOT/tmp"
mkdir -p "$NVM_DIR" "$TMPDIR"

if [ ! -s "$NVM_DIR/nvm.sh" ]; then
    echo "== installing nvm into $NVM_DIR =="
    # PROFILE=/dev/null so the installer does not edit shell rc files; ~/.bashrc.lfs
    # already exports NVM_DIR as "$LFS_HOME/nvm" and sources nvm.sh.
    PROFILE=/dev/null bash -c \
        "curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash" \
        >/dev/null 2>&1
fi

# shellcheck disable=SC1091
. "$NVM_DIR/nvm.sh"

if ! nvm ls "$NODE_VERSION" >/dev/null 2>&1; then
    echo "== installing node $NODE_VERSION =="
    nvm install "$NODE_VERSION" >/dev/null 2>&1
fi
nvm alias default "$NODE_VERSION" >/dev/null 2>&1
nvm use default >/dev/null 2>&1

# nvm refuses to operate when npm's global prefix is overridden.
npm config delete prefix >/dev/null 2>&1 || true

echo "node $(node --version), npm $(npm --version)"

echo "== installing Claude Code =="
npm install -g @anthropic-ai/claude-code 2>&1 | tail -2

BIN="$(readlink -f "$(which claude)")"
echo "== verifying =="
echo "path : $(which claude)"
echo "size : $(stat -c %s "$BIN") bytes"
if file "$BIN" | grep -q "missing section headers"; then
    echo "FAIL : binary is truncated (out of disk/quota?)" >&2
    exit 1
fi
echo "check: binary intact"
claude --version

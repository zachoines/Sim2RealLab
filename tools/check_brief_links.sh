#!/usr/bin/env bash
# Thin wrapper around check_brief_links.py — the documented entry point.
# Exits with the underlying script's status (0 on clean, 1 on any breakage).

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${DIR}/check_brief_links.py" "$@"

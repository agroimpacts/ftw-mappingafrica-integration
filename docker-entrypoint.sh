#!/usr/bin/env bash
set -euo pipefail

# If arguments were provided, run them. Otherwise open a shell.
if [ "$#" -gt 0 ]; then
  exec "$@"
else
  exec bash
fi
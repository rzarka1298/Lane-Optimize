#!/usr/bin/env bash
# Regenerate highway.net.xml from the plain-XML sources (highway.nod.xml,
# highway.edg.xml). Idempotent — safe to re-run any time the network
# topology changes.
#
# Run from anywhere: `bash sumo_scenarios/highway_3lane/build.sh`
# Or via the Makefile: `make scenario`
set -euo pipefail

cd "$(dirname "$0")"

uv run netconvert \
    --node-files highway.nod.xml \
    --edge-files highway.edg.xml \
    --output-file highway.net.xml \
    --no-turnarounds true

echo "✓ Wrote $(pwd)/highway.net.xml"

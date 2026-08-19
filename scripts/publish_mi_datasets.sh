#!/bin/bash
# Zip and publish collected MI datasets to Manifold.
#
#   scripts/publish_mi_datasets.sh data/mi_data/manifeel_pih_*_2000.zarr
#
# MI datasets live at manifold://agios-dexman-interns/tree/users/meih/mi_data
# (see fbsource manipulation_mochi/.llms/rules/dataset_storage.md). They go up as
# a single `<name>.zarr.zip`, not as a directory: a 2000-episode store is ~190k
# chunk files and `manifold putr` is per-file bound -- a 3.6 GiB / 9.5k-file
# pilot took 14 min that way, which extrapolates to ~4.6 h per real dataset.
#
# `zip -0` (store) because every zarr chunk is already Blosc-compressed, so
# deflate would burn CPU for nothing. Same recipe as the shapefilter collector's
# merge step.
#
# ⚠️ To read one back, extract with python `zipfile` -- Info-ZIP 6.00 rejects
# these archives as a zip-bomb false positive and `bsdtar` silently extracts a
# subset and exits 0. See .llms/rules/manifeel_devserver_setup.md.
#
# Env knobs:
#   DEST_ROOT   manifold destination (default the mi_data path above)
#   KEEP_ZIP    1 -> keep the local .zip after a successful upload
#   THREADS     manifold put threads (default 8)

set -euo pipefail

DEST_ROOT="${DEST_ROOT:-agios-dexman-interns/tree/users/meih/mi_data}"
KEEP_ZIP="${KEEP_ZIP:-0}"
THREADS="${THREADS:-8}"

[ $# -gt 0 ] || { echo "usage: $0 <dataset.zarr> [...]"; exit 1; }

manifold mkdirs "$DEST_ROOT" >/dev/null 2>&1 || true

status=0
for store in "$@"; do
  store="${store%/}"
  [ -d "$store" ] || { echo "no such store: $store"; status=1; continue; }
  [ -f "$store/mi_collect_info.json" ] || {
    echo "SKIP $store: no mi_collect_info.json -- not a finished dataset"; status=1; continue; }

  dir="$(dirname "$store")"
  name="$(basename "$store" .zarr)"
  zip_path="$dir/$name.zarr.zip"

  echo "[$name] zipping -> $zip_path"
  rm -f "$zip_path"
  ( cd "$dir" && zip -r -0 -q "$name.zarr.zip" "$name.zarr" )
  echo "[$name] $(du -h "$zip_path" | cut -f1) -> manifold://$DEST_ROOT/$name.zarr.zip"

  if manifold put "$zip_path" "$DEST_ROOT/$name.zarr.zip" --threads "$THREADS"; then
    remote="$(manifold ls "$DEST_ROOT" 2>/dev/null | awk -v n="$name.zarr.zip" '$2==n {print $1}')"
    local_bytes="$(stat -c %s "$zip_path")"
    if [ "$remote" = "$local_bytes" ]; then
      echo "[$name] uploaded, size verified ($local_bytes bytes)"
      [ "$KEEP_ZIP" = "1" ] || rm -f "$zip_path"
    else
      echo "[$name] ERROR size mismatch: local $local_bytes, remote ${remote:-missing}" >&2
      status=1
    fi
  else
    echo "[$name] ERROR upload failed -- local zip kept at $zip_path" >&2
    status=1
  fi
done
exit "$status"

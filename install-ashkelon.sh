#!/bin/sh
set -eu
REV=f00c455f3dd7f042242501cb42a543fb9abeb094
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DEST=${1:-"$ROOT/.build/ashkelon"}
if [ ! -d "$DEST/.git" ]; then
  git clone https://github.com/project-kikkuli/ashkelon.git "$DEST"
fi
cd "$DEST"
git fetch origin "$REV"
git checkout --detach "$REV"
if git apply --check "$ROOT/patches/ashkelon-transient-signal.patch"; then
  git apply "$ROOT/patches/ashkelon-transient-signal.patch"
else
  git apply --reverse --check "$ROOT/patches/ashkelon-transient-signal.patch"
fi
cargo build --locked --release
printf '%s\n' "$DEST/target/release/ashkelon"

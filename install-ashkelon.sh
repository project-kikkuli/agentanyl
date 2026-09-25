#!/bin/sh
set -eu
REV=08e123938137b59f3e4e7631f5d24cda0199866a
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DEST=${1:-"$ROOT/.build/ashkelon"}
if [ ! -d "$DEST/.git" ]; then
  git clone https://github.com/project-kikkuli/ashkelon.git "$DEST"
fi
cd "$DEST"
git fetch origin "$REV"
git checkout --detach "$REV"
cargo build --locked --release
printf '%s\n' "$DEST/target/release/ashkelon"

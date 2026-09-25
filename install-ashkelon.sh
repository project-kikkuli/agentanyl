#!/bin/sh
set -eu
REV=5612b96e31d8f85963c6e3d646ef85d64a29201d
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

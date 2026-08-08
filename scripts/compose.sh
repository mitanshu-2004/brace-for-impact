#!/usr/bin/env bash
# Edits built from clips render_all.sh already produced. Nothing here re-simulates.
#
#   bash scripts/compose.sh compare 250         all three robots side by side at 250 N
#   bash scripts/compose.sh allforces warned    one robot at all seven shove strengths
#   bash scripts/compose.sh all                 every combination of the two above
#
# Captions are drawn with PIL and composited with ffmpeg's overlay filter. The ffmpeg
# that ships inside imageio-ffmpeg is built without libfreetype, so drawtext is not
# available in it. Drawing the text here also keeps captions in the same font as the
# in-clip HUD.
set -eu

OUT=${CLIPS_DIR:-clips}
FF=$(python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
HERE="$(dirname "$0")"
FORCES="35 60 90 120 160 200 250"

need() { for f in "$@"; do [ -s "$f" ] || { echo "missing $f"; return 1; }; done; }

# caption <text> <width> <height> <fontsize> <outfile>
caption() {
  TEXT="$1" W="$2" H="$3" SZ="$4" DEST="$5" python - <<'EOF'
import os
from PIL import Image, ImageDraw, ImageFont

w, h, sz = int(os.environ["W"]), int(os.environ["H"]), int(os.environ["SZ"])
img = Image.new("RGB", (w, h), (0, 0, 0))
d = ImageDraw.Draw(img)
for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
             "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf"):
    if os.path.exists(path):
        f = ImageFont.truetype(path, sz)
        break
else:
    f = ImageFont.load_default()
text = os.environ["TEXT"]
lo, _, hi, bottom = d.textbbox((0, 0), text, font=f)
d.text(((w - (hi - lo)) / 2, (h - bottom) / 2), text, fill=(255, 255, 255), font=f)
img.save(os.environ["DEST"])
EOF
}

# The caption says "separate runs" because that is what these are. Episodes end
# at different moments, which shifts every later resample, so the three panels are not
# the same impulse sequence replayed. The clip illustrates the measured difference; the
# measurement is the fixed-force sweep in eval.py.
compare() {
  local F=$1
  local a="$OUT/warned_${F}N.mp4"
  local r="$OUT/unwarned_${F}N.mp4"
  local b="$OUT/warning_removed_${F}N.mp4"
  need "$a" "$r" "$b" || return 0
  echo "compare ${F}N"
  caption "${F} N shove    warned  /  not warned  /  warning switched off    (separate runs)" \
          2880 60 24 /tmp/cap_tw.png
  "$FF" -loglevel error -y -i "$a" -i "$r" -i "$b" -i /tmp/cap_tw.png -filter_complex "\
[0:v]scale=960:540[a];[1:v]scale=960:540[r];[2:v]scale=960:540[b];\
[a][r][b]hstack=inputs=3,pad=2880:600:0:0:black[grid];\
[grid][3:v]overlay=0:540[out]" \
    -map "[out]" -c:v libx264 -crf 16 -preset slow -pix_fmt yuv420p \
    "$OUT/compare_${F}N.mp4"
}

# One robot across all seven shove strengths, 4x2. Seven clips into eight cells, so the last cell
# gets a black source; xstack requires every cell to be fed.
allforces() {
  local robot=$1 inputs="" maps="" i=0
  for F in $FORCES; do
    local f="$OUT/${robot}_${F}N.mp4"
    need "$f" || return 0
    inputs="$inputs -i $f"
    maps="$maps[$i:v]scale=640:360[v$i];"
    i=$((i + 1))
  done
  echo "all-forces montage $robot"
  caption "${robot} robot at every shove strength - 35, 60, 90, 120 / 160, 200, 250 N" \
          2560 70 28 /tmp/cap_lad.png
  # shellcheck disable=SC2086
  "$FF" -loglevel error -y $inputs -f lavfi -i color=black:s=640x360:d=18 \
    -i /tmp/cap_lad.png -filter_complex "\
${maps}[7:v]scale=640:360[v7];\
[v0][v1][v2][v3][v4][v5][v6][v7]xstack=inputs=8:\
layout=0_0|640_0|1280_0|1920_0|0_360|640_360|1280_360|1920_360,\
pad=2560:790:0:0:black[grid];\
[grid][8:v]overlay=0:720[out]" \
    -map "[out]" -c:v libx264 -crf 16 -preset slow -pix_fmt yuv420p \
    "$OUT/${robot}_all_forces.mp4"
}

case "${1:-all}" in
  compare)  compare "${2:-250}" ;;
  allforces) allforces "${2:-warned}" ;;
  all)
    for F in $FORCES; do compare "$F"; done
    for robot in warned unwarned warning_removed; do allforces "$robot"; done
    ;;
  *) echo "usage: compose.sh [compare FORCE | allforces ROBOT | all]"; exit 1 ;;
esac

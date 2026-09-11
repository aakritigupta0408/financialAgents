#!/bin/zsh
C="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
for p in home ab_dashboard live_online experiment_review live_training index; do
  "$C" --headless --disable-gpu --hide-scrollbars --window-size=1280,9000 \
    --virtual-time-budget=15000 --screenshot=/tmp/audit_$p.png \
    "http://localhost:8901/site/$p.html" 2>/dev/null
done
ls -la /tmp/audit_*.png | awk '{print $9, $5}'

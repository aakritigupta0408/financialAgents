#!/bin/sh
# One-shot deadline timer for tonight's post-review deploy; delete after use.
while true; do
  hm=$(date +%H%M)
  [ "$hm" -ge 2030 ] && break
  sleep 60
done
echo DEPLOY_WINDOW_OPEN

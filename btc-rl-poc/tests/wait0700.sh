#!/bin/sh
# One-shot timer for the 7:00 AM final submission sweep; delete after use.
while true; do
  hm=$(date +%H%M)
  [ "$hm" -ge 0700 ] && break
  sleep 120
done
echo SUBMISSION_SWEEP_TIME

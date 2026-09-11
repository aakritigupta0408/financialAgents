"""Minimal Chrome DevTools controller for the dedicated Kalshi-demo
Chrome instance (launched with --remote-debugging-port=9222).

Usage:
  python3 scripts/chrome_ctl.py shot [outfile.png]   screenshot the page
  python3 scripts/chrome_ctl.py goto <url>           navigate
  python3 scripts/chrome_ctl.py js "<expression>"    eval JS, print result
  python3 scripts/chrome_ctl.py click <x> <y>        click at coordinates

Boundary: this tool is never used on payment or credential forms.
"""
import base64
import json
import sys
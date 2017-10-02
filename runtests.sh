#!/bin/sh
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib"

set -e

# E999 -- syntax error
# F821 -- undefined local variable
flake8 ./traildb/ | grep '[ ]E999[ ]\|[ ]F821[ ]' | awk '{print} END {exit(NR > 0)}'

env PYTHONPATH='.' python test/test.py

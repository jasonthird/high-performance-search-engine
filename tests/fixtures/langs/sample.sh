#!/bin/sh
compute_total() {
  local t=0
  for v in "$@"; do t=$((t + v)); done
  echo "$t"
}

render() { printf '%*s' "$1" ''; }

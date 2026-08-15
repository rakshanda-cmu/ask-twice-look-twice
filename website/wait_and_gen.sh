#!/bin/bash
# Poll until any GPU has >=20 GB free (without disturbing other jobs), then generate
# the gt=Yes examples. Self-triggers; writes gen_yes.log.
MR=/home/grg/Research/middle_layers_indicating_hallucinations
PY=/home/grg/anaconda3/envs/logitlens/bin/python
GEN=/home/grg/Research/ask-twice-look-twice-supp/website/gen_new_examples.py
LOG=/home/grg/Research/ask-twice-look-twice-supp/website/gen_yes.log
while true; do
  gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | awk -F',' '{f=$2+0; if(f>=20000){print $1; exit}}')
  if [ -n "$gpu" ]; then
    echo "[watch] GPU $gpu free enough; starting generation $(date)" > "$LOG"
    cd "$MR" && env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$MR" \
      TRANSFORMERS_VERBOSITY=error CHOSEN_FILE=chosen_yes.json \
      "$PY" "$GEN" >> "$LOG" 2>&1
    echo "[watch] generation exited $? $(date)" >> "$LOG"
    break
  fi
  sleep 90
done

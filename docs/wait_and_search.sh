#!/bin/bash
# Poll until a GPU has >=17 GB free, then run the "more" evidence search.
MR=/home/grg/Research/middle_layers_indicating_hallucinations
PY=/home/grg/anaconda3/envs/logitlens/bin/python
S=/home/grg/Research/ask-twice-look-twice-supp/website/search_evidence.py
LOG=/home/grg/Research/ask-twice-look-twice-supp/website/search_more.log
while true; do
  gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | awk -F',' '{f=$2+0; if(f>=17000){print $1; exit}}')
  if [ -n "$gpu" ]; then
    echo "[watch] GPU $gpu free; searching $(date)" > "$LOG"
    cd "$MR" && env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$MR" \
      TRANSFORMERS_VERBOSITY=error PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      EVID_FILE=evidence_sets_more.json EVID_OUT=evidence_more.json \
      "$PY" "$S" >> "$LOG" 2>&1
    echo "[watch] search exited $? $(date)" >> "$LOG"
    break
  fi
  sleep 90
done

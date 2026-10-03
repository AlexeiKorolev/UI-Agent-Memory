#!/bin/bash
# Run on the LOGIN node (compute nodes have no internet). ~35 GB models + ~6.7 GB screenshots.
set -e
source "$(cd "$(dirname "$0")/.." && pwd)/env.sh"
cd $PROJ
# 1. models (pinned revisions)
hf download Qwen/Qwen3-VL-8B-Instruct --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
hf download inclusionAI/UI-Venus-1.5-8B --revision a06ff6c6f15a9eca210769dacc1603f73b4a500c
# 2. GUIOdyssey v2 annotations + splits (random_split test only)
D=data/guiodyssey_v2; R=https://huggingface.co/datasets/hflqf88888/GUIOdyssey/resolve/61632d0f3f4d51d7e9561ce4f84347dd06b2019d
mkdir -p $D/splits $D/annotations
for f in splits/random_split.json splits/task_split.json splits/app_split.json splits/device_split.json all_annot.json README.md; do curl -sfL $R/$f -o $D/$f; done
python -c "import json;[print(x) for x in json.load(open('$D/splits/random_split.json'))['test']]" > $D/test_files.txt
cat $D/test_files.txt | xargs -P 8 -I{} sh -c "[ -s $D/annotations/{} ] || curl -sfL $R/annotations/{} -o $D/annotations/{}"
# 2b. reuse precomputed artifacts (saves ~25 CPU-min of OCR and the zip central-directory fetch)
[ -d data/ocr ] || tar -xzf artifacts/ocr_cache_600.tar.gz -C data
[ -f $D/zip_index.json ] || gunzip -c artifacts/zip_index.json.gz > $D/zip_index.json
# 3. screenshots for the first 600 stratified episodes, extracted from the 92 GB split zip by HTTP range reads
python -c "
from src.data import *
o=sample(600); open('data/sample_order_600.txt','w').write('\n'.join(o)+'\n')
open('data/shots_600.txt','w').write('\n'.join(s['screenshot'] for e in o for s in load_episode(e)['steps'])+'\n')"
python -m src.remote_zip --files data/shots_600.txt --out $D/screenshots --index $D/zip_index.json --workers 8
# 4. official reference code
mkdir -p third_party
[ -d third_party/GUI-Odyssey ] || git clone https://github.com/OpenGVLab/GUI-Odyssey.git third_party/GUI-Odyssey
git -C third_party/GUI-Odyssey checkout 5cdf76d9bd72034f3cb36bcf2cad77b8dcc81251
[ -d third_party/UI-Venus ] || git clone https://github.com/inclusionAI/UI-Venus.git third_party/UI-Venus
git -C third_party/UI-Venus checkout 1ffbaf653fb3d5d08e9ea14b013865496d68408e

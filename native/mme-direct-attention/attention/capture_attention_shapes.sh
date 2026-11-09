#!/usr/bin/env bash
# Capture compiler MME descriptors for the attention shapes and summarize the splits.
cd /home/a/gsv-data/media/mme-direct-20261108/repro/attention
bash capture_shapes_m.sh g "128 37760 128" "128 128 37760" "128 18944 128" "128 128 18944" 2>&1 | tail -5
cd fixtures
for d in g-m128n37760k128 g-m128n128k37760 g-m128n18944k128 g-m128n128k18944; do
  echo "== $d: $(ls $d/mme_debug 2>/dev/null | tr '\n' ' ')"
  grep -h 'Activation\|hx2\|accumEn\|aguOut\[0\]\[SLAVE\]\|aguIn\[0\]\[SLAVE\]\|spatialSize\|dataTypeIn' $d/mme_debug/*.txt 2>/dev/null | sed 's/^ *//' | sort | uniq -c | head -16
done

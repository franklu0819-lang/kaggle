#!/bin/bash
# Quick eval: v10 vs 3 opponents (20 games each)
cd /Users/leo/Projects/kaggle/maze-crawler
for opp in random v1 v49 v50; do
    result=$(python eval.py v10 $opp 2>&1 | grep "v10 vs $opp" | tail -1)
    echo "vs $opp: $result"
done

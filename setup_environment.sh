#!/usr/bin/env bash
conda create -n IACVimeq python=3.6 --yes
source activate IACVimeq
cat resources/requirements.txt | while read requirement; do
    conda install --yes ${requirement};
done
#!/usr/bin/env bash
echo "creating environment IACVimeq..."
conda create -n IACVimeq python=3.6 --yes
source activate IACVimeq
cat resources/conda_requirements.txt | while read requirement; do
    conda install --yes ${requirement};
done
pip install -r resources/pip_requirements.txt
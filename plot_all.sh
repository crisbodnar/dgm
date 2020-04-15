#!/usr/bin/env bash

# Plot the SDGM graphs with supervised GCN network
#python -u run_dgm.py --dataset=spam --reduce_method=binary_prob --intervals=20 --overlap=0 --eps=0.01 --min_component_size=4 --sdgm=True
python -u run_dgm.py --dataset=cora --reduce_method=tsne --intervals=20 --overlap=0 --eps=0.01 --min_component_size=20 --sdgm=True

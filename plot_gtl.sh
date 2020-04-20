#!/usr/bin/env bash

# ===================================== Graph Theoretic Lens SDGM ======================================================

# Plot the SDGM graphs with embeddings computed by a supervised GCN network
#python -u run_gtl.py --dataset=spam --reduce_method=none --intervals=20 --overlap=0 --eps=0.01 \
#                     --min_component_size=4 --sdgm --dir=sdgm_gtl/pr/ --lens=PR
#
#python -u run_gtl.py --dataset=cora --reduce_method=none --intervals=26 --overlap=0 --eps=0.04 \
#                     --min_component_size=5 --sdgm --dir=sdgm_gtl/pr --lens=PR
#
#python -u run_gtl.py --dataset=citeseer --reduce_method=none --intervals=20 --overlap=0 --eps=0.02 \
#                     --min_component_size=5 --sdgm --dir=sdgm_gtl/pr --lens=PR
#
#python -u run_gtl.py --dataset=pubmed --reduce_method=none --intervals=20 --overlap=0 --eps=0.02 \
#                     --min_component_size=15 --sdgm --dir=sdgm_gtl/pr --lens=PR


#python -u run_gtl.py --dataset=spam --reduce_method=none --intervals=20 --overlap=0 --eps=0.02 \
##                     --min_component_size=4 --sdgm --dir=sdgm_gtl/density/ --lens=density --scale=1.0
#
##python -u run_gtl.py --dataset=cora --reduce_method=none --intervals=20 --overlap=0 --eps=0.04 \
##                     --min_component_size=5 --sdgm --dir=sdgm_gtl/density --lens=density --scale=3.0
#
##python -u run_gtl.py --dataset=citeseer --reduce_method=none --intervals=20 --overlap=0 --eps=0.04 \
##                     --min_component_size=5 --sdgm --dir=sdgm_gtl/density --lens=density --scale=4.0
#
##python -u run_gtl.py --dataset=pubmed --reduce_method=none --intervals=20 --overlap=0 --eps=0.01 \
##                     --min_component_size=15 --sdgm --dir=sdgm_gtl/density --lens=density --scale=1 --cutoff=4



# Plot the DGM graphs with embeddings computed by a supervised GCN network
python -u run_gtl.py --dataset=spam --reduce_method=none --intervals=20 --overlap=0 --eps=0.01 \
                     --min_component_size=4 --dir=dgm_gtl/pr/ --lens=PR

python -u run_gtl.py --dataset=cora --reduce_method=none --intervals=26 --overlap=0 --eps=0.04 \
                     --min_component_size=5 --dir=dgm_gtl/pr --lens=PR

python -u run_gtl.py --dataset=citeseer --reduce_method=none --intervals=20 --overlap=0 --eps=0.02 \
                     --min_component_size=5 --dir=dgm_gtl/pr --lens=PR

python -u run_gtl.py --dataset=pubmed --reduce_method=none --intervals=20 --overlap=0 --eps=0.02 \
                     --min_component_size=15 --dir=dgm_gtl/pr --lens=PR


python -u run_gtl.py --dataset=spam --reduce_method=none --intervals=20 --overlap=0 --eps=0.02 \
                     --min_component_size=4 --dir=dgm_gtl/density/ --lens=density --scale=1.0

python -u run_gtl.py --dataset=cora --reduce_method=none --intervals=20 --overlap=0 --eps=0.04 \
                     --min_component_size=5 --dir=dgm_gtl/density --lens=density --scale=3.0

python -u run_gtl.py --dataset=citeseer --reduce_method=none --intervals=20 --overlap=0 --eps=0.04 \
                     --min_component_size=5 --dir=dgm_gtl/density --lens=density --scale=4.0

python -u run_gtl.py --dataset=pubmed --reduce_method=none --intervals=20 --overlap=0 --eps=0.01 \
                     --min_component_size=15 --dir=dgm_gtl/density --lens=density --scale=1 --cutoff=4
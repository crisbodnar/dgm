#!/usr/bin/env bash

# ===================================== Graph Theoretic Lens SDGM ======================================================

# Plot the SDGM graphs using PageRank
python -u run_gtl.py --dataset=spam --reduce_method=none --intervals=20 --overlap=0 --eps=0.01 \
                     --min_component_size=4 --sdgm --dir=sdgm_gtl/pr/ --lens=PR

python -u run_gtl.py --dataset=cora --reduce_method=none --intervals=26 --overlap=0 --eps=0.04 \
                     --min_component_size=5 --sdgm --dir=sdgm_gtl/pr --lens=PR

python -u run_gtl.py --dataset=citeseer --reduce_method=none --intervals=20 --overlap=0 --eps=0.02 \
                     --min_component_size=5 --sdgm --dir=sdgm_gtl/pr --lens=PR

python -u run_gtl.py --dataset=pubmed --reduce_method=none --intervals=20 --overlap=0 --eps=0.02 \
                     --min_component_size=15 --sdgm --dir=sdgm_gtl/pr --lens=PR

# Plot the SDGM graphs using a graph density function with RBF Kernel
python -u run_gtl.py --dataset=spam --reduce_method=none --intervals=20 --overlap=0 --eps=0.02 \
                     --min_component_size=4 --sdgm --dir=sdgm_gtl/density/ --lens=density --scale=1.0

python -u run_gtl.py --dataset=cora --reduce_method=none --intervals=20 --overlap=0 --eps=0.04 \
                     --min_component_size=5 --sdgm --dir=sdgm_gtl/density --lens=density --scale=3.0

python -u run_gtl.py --dataset=citeseer --reduce_method=none --intervals=20 --overlap=0 --eps=0.04 \
                     --min_component_size=5 --sdgm --dir=sdgm_gtl/density --lens=density --scale=4.0

python -u run_gtl.py --dataset=pubmed --reduce_method=none --intervals=20 --overlap=0 --eps=0.01 \
                     --min_component_size=15 --sdgm --dir=sdgm_gtl/density --lens=density --scale=1 --cutoff=4


# =====================================+ Graph Theoretic Lens DGM ======================================================

# Plot the DGM graphs using PageRank
python -u run_gtl.py --dataset=spam --reduce_method=none --intervals=20 --overlap=0.1  \
                     --min_component_size=4 --dir=dgm_gtl/pr/ --lens=PR

python -u run_gtl.py --dataset=cora --reduce_method=none --intervals=26 --overlap=0.15  \
                     --min_component_size=5 --dir=dgm_gtl/pr --lens=PR

python -u run_gtl.py --dataset=citeseer --reduce_method=none --intervals=30 --overlap=0.15 \
                     --min_component_size=5 --dir=dgm_gtl/pr --lens=PR

python -u run_gtl.py --dataset=pubmed --reduce_method=none --intervals=20 --overlap=0.15 \
                     --min_component_size=15 --dir=dgm_gtl/pr --lens=PR


# Plot the DGM graphs using a graph density function with RBF Kernel
python -u run_gtl.py --dataset=spam --reduce_method=none --intervals=20 --overlap=0.15 \
                     --min_component_size=4 --dir=dgm_gtl/density/ --lens=density --scale=1.0

python -u run_gtl.py --dataset=cora --reduce_method=none --intervals=20 --overlap=0.2 \
                     --min_component_size=10 --dir=dgm_gtl/density --lens=density --scale=3.0

python -u run_gtl.py --dataset=citeseer --reduce_method=none --intervals=20 --overlap=0.2 \
                     --min_component_size=10 --dir=dgm_gtl/density --lens=density --scale=4.0

python -u run_gtl.py --dataset=pubmed --reduce_method=none --intervals=20 --overlap=0.2 \
                     --min_component_size=15 --dir=dgm_gtl/density --lens=density --scale=1 --cutoff=4
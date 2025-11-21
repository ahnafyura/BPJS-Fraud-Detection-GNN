#!/bin/bash

python -m etl.load
python -m louvain.louvain
python -m etl.export
python -m gnn.hybrid_gnn
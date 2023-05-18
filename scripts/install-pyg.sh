#!/bin/bash
set -e

source .env || echo "No .env found."

PLATFORM=$(python -c 'import sys; print(sys.platform)')
TORCH_VERSION=$(python -c 'import sys, torch; pyg = torch.__version__ + "+cpu" if sys.platform in ["darwin"] else torch.__version__; print(pyg)')
URL=https://data.pyg.org/whl/torch-$TORCH_VERSION.html
PYG_LIBS="torch_scatter torch_sparse torch_cluster torch_spline_conv"

if [ "$PLATFORM" == "darwin" ]; then
    echo "You are on Mac ($PLATFORM)."
    pip install ninja
    pip install git+https://github.com/pyg-team/pyg-lib.git
    pip install --upgrade $PYG_LIBS -f $URL || echo "\n+++ Could not install $PYG_LIBS +++\n"
    pip install --upgrade torch_geometric -f $URL
else
    echo "You are on $PLATFORM"
    pip install --upgrade pyg_lib -f $URL || echo "\n+++ Could not install pyg_lib +++\n"
    pip install --upgrade $PYG_LIBS -f $URL || echo "\n+++ Could not install $PYG_LIBS +++\n"
    pip install --upgrade torch_geometric -f $URL
fi
#!/bin/bash
set -e
COMPUTE_CAPABILITY="CC70"
echo "Setup Speck with Compute Capability " $COMPUTE_CAPABILITY
mkdir build
cd build/
cmake -DCUDA_BUILD_$COMPUTE_CAPABILITY=TRUE ..
make -j
echo "Done with setup, ready to run, call:"
echo "build/runspECK.exe <path-to-matrix> config.ini"
echo "For more information please read readme.txt"
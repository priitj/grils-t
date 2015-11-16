#!/bin/sh
# files part of the distribution:
#
# fileformat.py monitor.py relink.py wireformat.py opdata.py profile.py trajectory.py
# grasp.pyx op.pyx gop.pyx toptw.pyx mctoptw.pyx tdop.pyx
# loader.py pyxify.sh setup.py
# op.pxd toptw.pxd
#
# compile: python setup.py build_ext --inplace
#

PYFILES="fileformat.py monitor.py relink.py wireformat.py opdata.py profile.py trajectory.py"

PROF_FILES="grasp.pyx op.pyx gop.pyx toptw.pyx mctoptw.pyx tdop.pyx"

PROF=$1

if [ "X${PROF}" = "X-p" ]; then
  echo "Enabling profiling for Cython."
fi

for fn in ${PYFILES}; do
  if [ "X${PROF}" = "X-p" ]; then
    sed "s/# cython: profile=False/# cython: profile=True/" < ${fn} > ${fn}x
  else
    cp ${fn} ${fn}x
  fi
done

for fn in ${PROF_FILES}; do
  if [ "X${PROF}" = "X-p" ]; then
    sed -ibak "s/# cython: profile=False/# cython: profile=True/" ${fn}
  else
    sed -ibak "s/# cython: profile=True/# cython: profile=False/" ${fn}
  fi
done

echo "Done, run 'python setup.py build_ext --inplace' to compile"


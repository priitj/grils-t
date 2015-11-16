#!/bin/bash

LOGPREFIX=$1
PROGRAM=$2
FILELIST=$3
MAXCORES=$4
REPEATS=$5
ITERS=$6

[ "X" = "X${LOGPREFIX}" ] && exit 1
[ "X" = "X${PROGRAM}" ] && exit 1
[ "X" = "X${FILELIST}" ] && exit 1
[ "X" = "X${MAXCORES}" ] && MAXCORES=8
[ "X" = "X${REPEATS}" ] && REPEATS=10
[ "X" = "X${ITERS}" ] && ITERS=10000

for nproc in 1 2 4 8 16 32 64 128 256 512; do
  [ ${nproc} -gt ${MAXCORES} ] && break
  NP=$((${nproc} + 1))
  ITERPART=$((${ITERS} / ${nproc} + 1))
  # don't use stdin to read, mpiexec does something to it
  while read -u 3 F; do
    FBASE=`basename ${F}`
    LOG="${LOGPREFIX}_n${nproc}_${FBASE}.log"
    echo "mpiexec -n ${NP} python ./loader.py ${PROGRAM} ${F} ${REPEATS} ${ITERPART} ${LOG}"
    mpiexec -n ${NP} python ./loader.py ${PROGRAM} ${F} ${REPEATS} ${ITERPART} ${LOG}
  done 3< ${FILELIST}
done


# grils-t
Parallelized cooperative GRASP-ILS hybrid OP solver

Some initial pointers for reproducing the experiments with parallelized GRASP-ILS hybrid on various OP variants:

Dependencies
------------

* Cython
* mpi4py
* MPI

Setting up
----------

Extract, then run `pyxify.sh` and follow the instructions printed.

Acquiring test data
-------------------

OP and GOP test data:

http://josilber.scripts.mit.edu/gop.zip

TOPTW, MCTOPTW and TDOP benchmarks are available here:

http://www.mech.kuleuven.be/en/cib/op/

Running tests
-------------

Run 2000 iterations, 10 repeated tests on OP instance in file `1rat99.gop` with GRILS-T
```
  mpirun python ./loader.py op 1rat99.gop 10 2000 logfile
```
Use `op_i` as the first parameter to `loader.py` to run GRILS-I and `op_dpr` to run distributed strategy GRASP-PR.

Run 2000 iterations, 10 repeated tests on GOP instance in file `4china27.gop` (k=4) with GRILS-T
```
  mpirun python ./loader.py gop 4china27.gop 10 2000 4 logfile
```
Use `gop_i` to run GRILS-I and `gop_dpr` to run distributed strategy GRASP-PR.

Run 2000 iterations, 10 repeated tests on TOPTW instance in file `r201.txt` (m=2) with GRILS-T
```
  mpirun python ./loader.py toptw r201.txt 10 2000 2 logfile
```
Use `toptw_i` to run GRILS-I and `toptw_dpr` to run distributed strategy GRASP-PR.

Run 2000 iterations, 10 repeated tests on MCTOPTW instance in file `MCTOPMTW-2-rc106.txt` with GRILS-T
```
  mpirun python ./loader.py mctoptw MCTOPMTW-2-rc106.txt 10 2000 logfile
```
Use `mctoptw_i` to run GRILS-I and `mctoptw_dpr` to run distributed strategy GRASP-PR.

Run 10000 iterations, 10 repeated tests on TDOP instance in file `p1.1.a.txt` with GRILS-T. `arc_cat_1.txt` contains the arc cateories
```
  mpirun python ./loader.py tdop p1.1.a.txt arc_cat_1.txt  10 10000 logfile
```
Use `tdop_i` to run GRILS-I and `tdop_dpr` to run distributed strategy GRASP-PR.

In these examples, the number of workers is pre-configured, hence no parameters to `mpirun`.

GRILS-I runs faster and GRASP-PR runs significantly slower that GRILS-T (number of iterations < 1000 is usually appropriate, except with very small instances).

Helper scripts
--------------

Outdated `runtests.sh` and `test.bash` are included as they can be helpful in providing a starting point for automated tests. Both will require fixing and tweaking to match the current version of the Python scripts and the local implementation of MPI.

Licence
-------

This work is copyright protected. All rights reserved, except:

you may download and use this code for non-commercial purposes, including reproducing test results as part of academic peer review process. The right to use does not extend to redistributing the code in original or modified form.

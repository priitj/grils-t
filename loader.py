#!/usr/bin/env python
#
# stub loader for cythonized modules

import op
import gop
import toptw
import mctoptw
import tdop
import profile

if __name__ == "__main__":
    import sys
    test = sys.argv[1]
    argv = sys.argv[:1] + sys.argv[2:]
    if test == "op":
        op.traj_test(argv)
    elif test == "op_i":
        op.ind_test(argv)
    elif test == "op_pr":
        op.pr_test(argv)
    elif test == "op_dpr":
        op.dpr_test(argv)
    elif test == "op_prof":
        profile.traj_test(argv)
    elif test == "op_pr_prof":
        profile.pr_test(argv)
    elif test == "gop":
        gop.traj_test(argv)
    elif test == "gop_i":
        gop.ind_test(argv)
    elif test == "gop_pr":
        gop.pr_test(argv)
    elif test == "gop_dpr":
        gop.dpr_test(argv)
    elif test == "gop_prof":
        profile.gop_traj_test(argv)
    elif test == "gop_pr_prof":
        profile.gop_pr_test(argv)
    elif test == "toptw":
        toptw.traj_test(argv)
    elif test == "toptw_i":
        toptw.ind_test(argv)
    elif test == "toptw_dpr":
        toptw.dpr_test(argv)
    elif test == "toptw_pr":
        toptw.pr_test(argv)
    elif test == "toptw_prof":
        profile.toptw_traj_test(argv)
    elif test == "toptw_pr_prof":
        profile.toptw_pr_test(argv)
    elif test == "mctoptw":
        mctoptw.traj_test(argv)
    elif test == "mctoptw_i":
        mctoptw.ind_test(argv)
    elif test == "mctoptw_pr":
        mctoptw.pr_test(argv)
    elif test == "mctoptw_dpr":
        mctoptw.dpr_test(argv)
    elif test == "tdop":
        tdop.traj_test(argv)
    elif test == "tdop_i":
        tdop.ind_test(argv)
    elif test == "tdop_pr":
        tdop.pr_test(argv)
    elif test == "tdop_dpr":
        tdop.dpr_test(argv)
    elif test == "tdop_prof":
        profile.tdop_traj_test(argv)
    elif test == "tdop_pr_prof":
        profile.tdop_pr_test(argv)


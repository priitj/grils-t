#!/usr/bin/env python
# cython: profile=False

import op
import gop
import toptw
import tdop
import trajectory
import relink

# single process versions for profiling

class OP_GRASP_T_S(op.OP_GRASP_T_Common, trajectory.TrajectoryGRASP):
    pass

class OP_GRASP_PR_S(op.OP_GRASP_PR_Common, relink.PathRelinkGRASP):
    pass

def traj_test(argv):
    do_profile(argv, op.TEST_GRASP_T)

def pr_test(argv):
    do_profile(argv, op.TEST_GRASP_PR)

def gop_traj_test(argv):
    do_gopprofile(argv, op.TEST_GRASP_T)

def gop_pr_test(argv):
    do_gopprofile(argv, op.TEST_GRASP_PR)

def toptw_traj_test(argv):
    do_toptwprofile(argv, op.TEST_GRASP_T)

def toptw_pr_test(argv):
    do_toptwprofile(argv, op.TEST_GRASP_PR)

def tdop_traj_test(argv):
    do_tdopprofile(argv, op.TEST_GRASP_T)

def tdop_pr_test(argv):
    do_tdopprofile(argv, op.TEST_GRASP_PR)

def do_search(problem, dlim, iters, searchclass):
    for d in dlim:
        problem.set_capacity(d)
        g = searchclass()
        solution = g.search(problem, iters)
        # comment this out to not to profile IO
        #print d
        #solution.pretty_print()

def do_gopsearch(problem, Wvect, iters, searchclass):
    for W in Wvect:
        problem.set_W(W)
        g = searchclass()
        solution = g.search(problem, iters)

def do_profile(argv, testid):
    import fileformat
    import opdata
    import os.path
    import pstats
    import cProfile

    if len(argv) < 3:
        return
    fr = fileformat.GOPReader(argv[1])
    dlim = opdata.get_dlim(os.path.basename(argv[1]))
    iters = int(argv[2])
    if len(argv) > 3:
        pfile = argv[3]
    else:
        pfile = "Profile.prof"

    matrix = fr.get_distmatrix()
    problem = op.OPProblem(
        [ op.OPItem(i, x[0], 0.0, matrix[i])
            for i, x in enumerate(fr.get_scores()) ],
        fr.get_start(),
        fr.get_end(),
        0.0
    )

    if testid == op.TEST_GRASP_T:
        searchclass = OP_GRASP_T_S
    elif testid == op.TEST_GRASP_PR:
        searchclass = OP_GRASP_PR_S

    cProfile.runctx("do_search(problem, dlim, iters, searchclass)",
        globals(), locals(), pfile)

    s = pstats.Stats(pfile)
    s.strip_dirs().sort_stats("time").print_stats()

def do_gopprofile(argv, testid):
    import fileformat
    import os.path
    import pstats
    import cProfile

    if len(argv) < 4:
        return
    fr = fileformat.GOPReader(argv[1])
    iters = int(argv[2])
    k = int(argv[3])
    if len(argv) > 4:
        pfile = argv[4]
    else:
        pfile = "Profile.prof"

    matrix = fr.get_distmatrix()
    problem = gop.GOPProblem(
        [ gop.GOPItem(i, x, 0.0, matrix[i])
            for i, x in enumerate(fr.get_scores()) ],
        fr.get_start(),
        fr.get_end(),
        5000.0,  # "27 Chinese Cities"
        k
    )

    if testid == op.TEST_GRASP_T:
        searchclass = gop.GOP_GRASP_T_S
    elif testid == op.TEST_GRASP_PR:
        searchclass = gop.GOP_GRASP_PR_S

    Wvect = [
        [0.25,0.25,0.25,0.25],
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1]
    ]

    cProfile.runctx("do_gopsearch(problem, Wvect, iters, searchclass)",
        globals(), locals(), pfile)

    s = pstats.Stats(pfile)
    s.strip_dirs().sort_stats("time").print_stats()

def do_toptwprofile(argv, testid):
    import fileformat
    import os.path
    import pstats
    import cProfile

    if len(argv) < 4:
        return
    fr = fileformat.TOPTWReader(argv[1])
    iters = int(argv[2])
    tours = int(argv[3])
    if len(argv) > 4:
        pfile = argv[4]
    else:
        pfile = "Profile.prof"

    matrix = fr.get_distmatrix()
    dlim = fr.get_dlim()
    problem = toptw.TOPTWProblem(
        [ toptw.TOPTWItem(i, x[0][0], x[1], matrix[i], x[2][0][0], x[2][0][1])
            for i, x in enumerate(fr.get_tuples()) ],
        fr.get_start(),
        fr.get_end(),
        dlim,
        tours,
    )

    if testid == op.TEST_GRASP_T:
        searchclass = toptw.TOPTW_GRASP_T_S
    elif testid == op.TEST_GRASP_PR:
        searchclass = toptw.TOPTW_GRASP_PR_S

    cProfile.runctx("do_search(problem, [dlim], iters, searchclass)",
        globals(), locals(), pfile)

    s = pstats.Stats(pfile)
    s.strip_dirs().sort_stats("time").print_stats()

def do_tdopprofile(argv, testid):
    import fileformat
    import os.path
    import pstats
    import cProfile

    if len(argv) < 4:
        return
    fr = fileformat.TDOPReader(argv[1], argv[2])
    iters = int(argv[3])
    if len(argv) > 4:
        pfile = argv[4]
    else:
        pfile = "Profile.prof"

    matrix = fr.get_distmatrix()
    arctype = fr.get_arctype()
    dlim = fr.get_dlim()
    problem = tdop.TDOPProblem(
        [ tdop.TDOPItem(i, x[0], 0.0, matrix[i], arctype[i])
            for i, x in enumerate(fr.get_scores()) ],
        fr.get_start(),
        fr.get_end(),
        dlim,
    )

    if testid == op.TEST_GRASP_T:
        searchclass = tdop.TDOP_GRASP_T_S
    elif testid == op.TEST_GRASP_PR:
        searchclass = tdop.TDOP_GRASP_PR_S

    cProfile.runctx("do_search(problem, [dlim], iters, searchclass)",
        globals(), locals(), pfile)

    s = pstats.Stats(pfile)
    s.strip_dirs().sort_stats("time").print_stats()


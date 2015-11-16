#!/usr/bin/env python
# cython: profile=False

import grasp
import op
import trajectory
import relink

cimport op

from cpython.mem cimport PyMem_Malloc, PyMem_Free

class GOPItem(op.OPItem):
    """Orienteering problem vertice"""
    def __init__(self, idx, scores = [], cost = 0.0, distvector = []):
        super(GOPItem, self).__init__(idx, 0.0, cost)
        self.distvector = distvector
        self.attrib = scores
        del self.reward     # for detecting invalid use

class GOPProblem(op.OPProblem):
    """OP instance"""
    def __init__(self, items=[], startidx=0, endidx=0, capacity = 0.0,
        k=1, W=[1.0]):
        super(GOPProblem, self).__init__(items, startidx, endidx, capacity)
        self.k = k
        self.W = W

    def get_k(self):
        return self.k

    def get_W(self):
        return self.W

    def set_k(self, k):
        self.k = k

    def set_W(self, W):
        self.W = W


class GOPSolution(op.OPSolution):
    """Solution for the GOP."""

    def __init__(self, k=1, W=[1.0]):
        self.k = k
        self.W = W
        self.attrcnt = len(self.W)
        super(GOPSolution, self).__init__()

    def reset(self):
        super(GOPSolution, self).reset()
        self.cat_sum = [0.0] * self.attrcnt

    def add_score(self, cand):
        """Compute the score after inserting the item"""
        mod_score = 0.0
        for attridx in xrange(self.attrcnt):
            cs = self.cat_sum[attridx] + cand.cat_sum[attridx]
            mod_score += self.W[attridx] * (cs ** (1.0/self.k))
        return mod_score

    def del_score(self, cand):
        """Compute the score after removing the item"""
        mod_score = 0.0
        for attridx in xrange(self.attrcnt):
            cs = self.cat_sum[attridx] - cand.cat_sum[attridx]
            mod_score += self.W[attridx] * (cs ** (1.0/self.k))
        return mod_score

    def mk_cand(self, item, position):
        """Make a candidate item with precomputed score"""
        cdef int attrcnt, attridx
        cdef double attr, k
        attrcnt = self.attrcnt
        k = self.k
        cand = super(GOPSolution, self).mk_cand(item, position)
        cand.cat_sum = []
        #cand.score = 0.0
        attridx = 0
        while attridx < attrcnt:
            attr = item.attrib[attridx]
            cand.cat_sum.append(attr ** k)
            attridx = attridx + 1
            #cand.score += self.W[attridx] * (cs ** (1.0/self.k))
        return cand

    def insert(self, cand, update=True):
        super(GOPSolution, self).insert(cand, update)
        if update:
            for attridx in xrange(self.attrcnt):
                self.cat_sum[attridx] += cand.cat_sum[attridx]

    def remove(self, idx):
        cand = super(GOPSolution, self).remove(idx)
        for attridx in xrange(self.attrcnt):
            self.cat_sum[attridx] -= cand.cat_sum[attridx]
        return cand

    def receive_solution(self, comm, source, problem):
        super(GOPSolution, self).receive_solution(comm, source, problem)
        # repair the category sum cache
        self.cat_sum = [0.0] * self.attrcnt
        for i, cand in self.get_items():
            if i == 0:
                start_idx = cand.item.idx
            if i == 0 or cand.item.idx != start_idx:
                for attridx in xrange(self.attrcnt):
                    self.cat_sum[attridx] += cand.item.attrib[attridx] ** self.k

    def copy(self):
        clone = super(GOPSolution, self).copy()
        clone.W = self.W  # OK to copy reference, W never changes
        clone.k = self.k
        clone.attrcnt = self.attrcnt
        clone.cat_sum = []
        for attridx in xrange(self.attrcnt):
            clone.cat_sum.append(self.cat_sum[attridx])
        return clone

    def verify(self):
        """Check the integrity of the solution"""
        prov_cost = 0.0
        prov_score = 0.0
        cs = [0.0] * self.attrcnt
        last_cand = None
        for i, cand in self.get_items():
            if i > 0:
                prov_cost += last_cand.item.travel_cost(cand.item)
            else:
                start_idx = cand.item.idx
            last_cand = cand
            if i == 0 or cand.item.idx != start_idx:
                for attridx in xrange(self.attrcnt):
                    cs[attridx] += cand.item.attrib[attridx] ** self.k
        for attridx in xrange(self.attrcnt):
            prov_score += self.W[attridx] * (cs[attridx] ** (1.0/self.k))

        if (abs(prov_cost - self.get_cost()) > 0.0001 or
            abs(prov_score - self.get_score()) > 0.0001):
            return False, "Total score or cost mismatch"
        else:
            return True, "OK"


class GOP_GRASP(op.OP_GRASP):
    """GOP solver"""
    def mk_solution(self):
        solution = GOPSolution(self.problem.get_k(), self.problem.get_W())
        start = self.problem.get_start()
        end = self.problem.get_end()
        solution.insert(solution.mk_cand(start, 0), start != end)
        solution.insert(solution.mk_cand(end, 1))
        return solution

    # Copy-paste of the OP RCL builder.
    # It is a bit tricky to optimize and modularize at the same time, hence
    # this ugliness.
    def make_rcl(self, solution, alpha):
        cdef int aidx, idx, pos, maxidx, maxpos, cli, rcli, maxcli, nitems
        cdef int maxattr, attridx
        cdef double cost, score, score_delta, maxcost, h, kinv
        cdef double minh, maxh, threshold
        cdef int *sol_idx
        cdef int *cl_idx
        cdef double *cl_hval
        cdef int *cl_pos
        cdef double *cl_cost
        cdef int *rcl_idx
        cdef double *tc
        cdef double *sol_attr
        cdef double *W
        cdef op.FastDistMatrix dm
        cdef op.FastRCL rcl
        cdef double best_h
        cdef int best_cli, best_pos

        # Preparation. Get available items
        avail_items = list(self.available_items(solution))
        maxidx = len(avail_items)
        maxpos = solution.get_size()

        # Allocate memory
        rcl = op.FastRCL(maxpos, maxidx, solution.mk_cand, avail_items)
        sol_idx = rcl.sol_idx_m
        cl_idx = rcl.cl_idx_m
        cl_hval = rcl.cl_hval_m
        cl_pos = rcl.cl_pos_m
        cl_cost = rcl.cl_cost_m
        rcl_idx = rcl.rcl_idx_m

        # Solution items table and distance lookup table
        for idx, cand in solution.get_items():
            sol_idx[idx] = cand.item.idx
        dm = self.problem.distmatrix
        tc = dm.table
        nitems = dm.nitems

        # Prepare the cat_sum table and the weights table
        maxattr = solution.attrcnt
        sol_attr = <double *> PyMem_Malloc(maxattr * sizeof(double))
        W = <double *> PyMem_Malloc(maxattr * sizeof(double))
        attridx = 0
        while attridx < maxattr:
            sol_attr[attridx] = solution.cat_sum[attridx]
            W[attridx] = solution.W[attridx]
            attridx = attridx + 1
        kinv = 1.0 / solution.k
        score = solution.get_score()

        # First pass. Compute the cost of all insertions and hval if
        # the move is feasible. Determine bounds.
        maxcost = self.problem.capacity - solution.cost
        minh = 1
        maxh = 0
        aidx = 0  # index in the available items list
        while aidx < maxidx:
            item = avail_items[aidx]
            idx = item.idx  # the global id of the item

            score_delta = -score
            attridx = 0
            while attridx < maxattr:
                score_delta = score_delta + (W[attridx] * (
                    sol_attr[attridx] + cand.cat_sum[attridx] ** kinv))
                attridx = attridx + 1

            pos = 1
            cli = aidx * (maxpos - 1)   # pos 0 not used
            # if optimum insertion
            best_h = 1.0
            best_cli = -1
            best_pos = -1
            while pos < maxpos:
                cost = op.cost_delta_noshift(tc,
                    sol_idx, idx, pos, maxpos, nitems)

#                if cost > maxcost:
#                    cl_pos[cli] = -1 # not feasible, not included etc
#                else:
#                    h = score_delta / (cost + 0.0001)
#                    cl_idx[cli] = aidx  # not the same as problem index
#                    cl_pos[cli] = pos
#                    cl_hval[cli] = h
#                    cl_cost[cli] = cost
#                    if minh > maxh:
#                        minh = h
#                        maxh = h
#                    else:
#                        if h < minh:
#                            minh = h
#                        if h > maxh:
#                            maxh = h

                cl_pos[cli] = -1 # not feasible, not included etc
                if cost <= maxcost:
                    h = score_delta / (cost + 0.0001)
                    if best_cli == -1 or h > best_h:
                        best_h = h
                        best_cli = cli
                        best_pos = pos

                    # these can be pre-filled
                    cl_idx[cli] = aidx  # not the same as problem index
                    cl_hval[cli] = h
                    cl_cost[cli] = cost

                pos = pos + 1
                cli = cli + 1

            # enable the best position
            if best_cli > -1:
                cl_pos[best_cli] = best_pos
                if minh > maxh:
                    minh = best_h
                    maxh = best_h
                else:
                    if best_h < minh:
                        minh = best_h
                    if best_h > maxh:
                        maxh = best_h

            aidx = aidx + 1

        # Second pass. Build the RCL lookup index
        threshold = minh + (1 - alpha) * (maxh - minh)
        if threshold > maxh:
            threshold = maxh - 0.000000000001
        maxcli = (maxpos - 1) * maxidx
        cli = 0
        rcli = 0
        while cli < maxcli:
            if cl_pos[cli] > -1 and cl_hval[cli] >= threshold:
                rcl_idx[rcli] = cli
                rcli = rcli + 1
            cli = cli + 1
        rcl.ncand = rcli

        PyMem_Free(sol_attr)
        PyMem_Free(W)
        return rcl

    def hval0(self, cand, solution):
        """Naive heuristic"""
        return cand.score / (cand.item.cost + cand.travel + 0.0001)

    def hval(self, cand, solution):
        """Full heuristic"""
        score_delta = solution.add_score(cand) - solution.get_score()
        return score_delta / (cand.item.cost + cand.travel + 0.0001)


class GOP_GRASP_T_Common(GOP_GRASP, op.OP_GRASP_T_Common):
    """GOP solver using trajectory rejoining"""
    pass

class GOP_GRASP_PR_Common(GOP_GRASP, op.OP_GRASP_PR_Common):
    """GOP solver using path relinking"""
    def removal_hval(self, solution, idx):
        cand = solution.get_item(idx)
        score_delta = solution.get_score() - solution.del_score(cand)
        if score_delta == 0.0:
            score_delta = 0.0001
        return (solution.cost_delta(cand.item, idx, 1) / score_delta)


class GOP_GRASP_T(GOP_GRASP_T_Common, trajectory.CoopGRASPT):
    pass

class GOP_GRASP_I(GOP_GRASP_T_Common, trajectory.IndGRASPT):
    pass

class GOP_GRASP_PR(GOP_GRASP_PR_Common, relink.CoopGRASPPR):
    pass

class GOP_GRASP_DPR(GOP_GRASP_PR_Common, relink.DistribGRASPPR):
    pass

def traj_test(argv):
    do_test(argv, op.TEST_GRASP_T)

def ind_test(argv):
    do_test(argv, op.TEST_GRASP_I)

def pr_test(argv):
    do_test(argv, op.TEST_GRASP_PR)

def dpr_test(argv):
    do_test(argv, op.TEST_GRASP_DPR)

def do_test(argv, testid):
    from mpi4py import MPI
    import fileformat
    import sys
    import monitor
    import os.path

    if len(argv) < 5:
        sys.exit(1)
    fr = fileformat.GOPReader(argv[1])
    repeats = int(argv[2])
    iters = int(argv[3])
    k = int(argv[4])

    matrix = fr.get_distmatrix()
    problem = GOPProblem(
        [ GOPItem(i, x, 0.0, matrix[i])
            for i, x in enumerate(fr.get_scores()) ],
        fr.get_start(),
        fr.get_end(),
        5000.0,  # "27 Chinese Cities"
        k
    )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        if len(argv) > 5:
            logfile = open(argv[5], "w")
        else:
            logfile = sys.stdout
    else:
        logfile = None

    try:
        elim_time = int(argv[6])
        reactive_delta = int(argv[7])
    except:
        elim_time = None
        reactive_delta = None

    if testid == op.TEST_GRASP_T:
        monitorf = monitor.monitor_best
        searchclass = GOP_GRASP_T
    elif testid == op.TEST_GRASP_I:
        monitorf = monitor.monitor_best
        searchclass = GOP_GRASP_I
    elif testid == op.TEST_GRASP_PR:
        monitorf = monitor.monitor_pool
        searchclass = GOP_GRASP_PR
    elif testid == op.TEST_GRASP_DPR:
        monitorf = monitor.monitor_distpool
        searchclass = GOP_GRASP_DPR

    Wvect = [
        [0.25,0.25,0.25,0.25],
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1]
    ]
    for W in xrange(5):
        problem.set_W(Wvect[W])
        for i in xrange(repeats):
            comm.Barrier()
            if rank == 0:
                # control process
                best = monitorf(comm, logfile, {"aux1":W, "aux2": i})
                #best.pretty_print()
            else:
                # search process
                g = searchclass(comm)
                if testid == op.TEST_GRASP_T:
                    if elim_time is not None:
                        g.ELIMINATION_TIME = elim_time
                        g.ra.REACTIVE_DELTA = reactive_delta
                g.search(problem, iters)

class GOP_GRASP_T_S(GOP_GRASP_T_Common, trajectory.TrajectoryGRASP):
    pass

class GOP_GRASP_PR_S(GOP_GRASP_PR_Common, relink.PathRelinkGRASP):
    pass

if __name__ == "__main__":
    import fileformat
    import sys
    import os.path

    if len(sys.argv) < 5:
        sys.exit(1)
    fr = fileformat.GOPReader(sys.argv[1])
    iters = int(sys.argv[2])
    k = int(sys.argv[3])
    W = int(sys.argv[4])

    matrix = fr.get_distmatrix()
    problem = GOPProblem(
        [ GOPItem(i, x, 0.0, matrix[i])
            for i, x in enumerate(fr.get_scores()) ],
        fr.get_start(),
        fr.get_end(),
        5000.0,  # "27 Chinese Cities"
        k
    )

    Wvect = [
        [0.25,0.25,0.25,0.25],
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1]
    ]

    problem.set_W(Wvect[W])
    #g = GOP_GRASP_T_S()
    g = GOP_GRASP_PR_S()
    solution = g.search(problem, iters)
    solution.pretty_print()


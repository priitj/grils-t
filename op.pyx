#!/usr/bin/env python
# cython: profile=False

import grasp
import trajectory
import relink
import random

from cpython.mem cimport PyMem_Malloc, PyMem_Free
import cython

class OPItem(grasp.Item):
    """Orienteering problem vertice"""
    def __init__(self, idx, reward = 0.0, cost = 0.0, distvector = []):
        super(OPItem, self).__init__(idx, reward, cost)
        self.distvector = distvector

    def travel_cost(self, to_item):
        return self.distvector[to_item.idx]

cdef class FastDistMatrix:
    """Container class for the distance table. Just a chunk of memory."""
    def __cinit__(self, items):
        cdef int i, j, tidx
        nitems = len(items)
        self.table = <double *> PyMem_Malloc(nitems * nitems * sizeof(double))
        i = 0
        while i < nitems:
            fromitem = items[i]
            j = 0
            tidx = i * nitems
            while j < nitems:
                self.table[tidx] = fromitem.travel_cost(items[j])
                j = j + 1
                tidx = tidx + 1
            i = i + 1
        self.nitems = nitems

    def __dealloc__(self):
        PyMem_Free(self.table)

class OPProblem(grasp.Problem):
    """OP instance"""
    def __init__(self, items=[], startidx=0, endidx=0, capacity = 0.0):
        super(OPProblem, self).__init__(items, capacity)
        self.startidx = startidx
        self.endidx = endidx
        self.distmatrix = FastDistMatrix(self.items)

    def get_start(self):
        return self.items[self.startidx]

    def get_end(self):
        return self.items[self.endidx]


class OPCandidate(grasp.Candidate):
    """Insertion candidate with an index and travel cost"""
    def __init__(self, item, position):
        self.item = item
        self.hval = None
        self.travel = 0
        self.index = position


class OPSolution(grasp.Solution):
    """Solution for the orienteering problem. Items are ordered"""

    def add_cost(self, cand):
        """Compute the total cost after inserting the item"""
        return self.cost + cand.item.cost + cand.travel

    def cost_delta(self, item, position, shift=0):
        """Helper function for determining the change in cost"""
        delta = 0.0
        if position > 0:
            delta += self.items[position-1].item.travel_cost(item)
        if len(self.items) > position + shift:
            if position > 0:
                delta -= self.items[position-1].item.travel_cost(
                    self.items[position + shift].item)
            delta += item.travel_cost(self.items[position + shift].item)
        return delta

    def del_cost(self, cand):
        """Compute the total cost after removing the item"""
        position = cand.index
        return self.cost - self.cost_delta(cand.item, position)

    def mk_cand(self, item, position):
        """Build an appropriate solution candidate"""
        cand = OPCandidate(item, position)
        cand.travel += self.cost_delta(item, position)
        return cand

    def removables(self):
        return list(self.get_items())[1:-1]

    def insert_cand(self, cand):
        self.items.insert(cand.index, cand)
        for c in self.items[cand.index + 1:]:
            c.index += 1

    def remove_pos(self, idx):
        for c in self.items[idx + 1:]:
            c.index -= 1
        return self.items.pop(idx)

# XXX: redundant?
#    def insert(self, cand, update=True):
#        super(OPSolution, self).insert(cand, update)

    def swap_cost(self, s1, e1, s2, e2):
        """Computes the time delta resulting from reconnecting edges
        (s1, e1), (s2, e2) as (s1, s2) and (e1, e2)"""
        is1 = self.get_item(s1)
        ie1 = self.get_item(e1)
        is2 = self.get_item(s2)
        ie2 = self.get_item(e2)
        delta = -is1.item.travel_cost(ie1.item)
        delta += is1.item.travel_cost(is2.item)
        delta -= is2.item.travel_cost(ie2.item)
        delta += ie1.item.travel_cost(ie2.item)
        return delta

    def subpath_reversal(self, pos_from, pos_to):
        """A 2-opt move always reverses a subpath. This does the
        necessary bookkeeping in the solution"""
        self.cost += self.swap_cost(pos_from-1, pos_from, pos_to, pos_to+1)
        self.items = (self.items[:pos_from] +
                        list(reversed(self.items[pos_from:pos_to+1])) +
                        self.items[pos_to+1:])
        # we only repair the index, travel time is recalculated on demand
        for i in xrange(pos_from, pos_to+1):
            self.items[i].index = i

    def verify(self):
        """Check the integrity of the solution"""
        prov_cost = 0.0
        prov_score = 0.0
        last_cand = None
        for i, cand in self.get_items():
            if i > 0:
                prov_cost += last_cand.item.travel_cost(cand.item)
            else:
                start_idx = cand.item.idx
            last_cand = cand
            if i == 0 or cand.item.idx != start_idx:
                prov_score += cand.item.reward
        if (abs(prov_cost - self.get_cost()) > 0.0001 or
            abs(prov_score - self.get_score()) > 0.0001):
            return False, "Total score or cost mismatch"
        else:
            return True, "OK"

cdef class FastRCL(object):
    """Restricted candidate list. Items restricted by threshold"""
    def __cinit__(self, sol_size, nitems, mkcand, items):
        """Prepare the memory for the list data"""
        cdef int cl_size = (sol_size - 1) * nitems

        # memory for the solution lookup index
        self.sol_idx_m = <int *> PyMem_Malloc(sol_size * sizeof(int))

        # memory for the full candidate list
        self.cl_idx_m = <int *> PyMem_Malloc(cl_size * sizeof(int))
        self.cl_hval_m = <double *> PyMem_Malloc(cl_size * sizeof(double))
        self.cl_pos_m = <int *> PyMem_Malloc(cl_size * sizeof(int))
        self.cl_cost_m = <double *> PyMem_Malloc(cl_size * sizeof(double))

        # memory for the candidate index
        self.ncand = 0
        self.rcl_idx_m = <int *> PyMem_Malloc(cl_size * sizeof(int))

        # candidate constructor
        self.mkcand = mkcand
        # items can be taken from here
        self.items = items

    def __dealloc__(self):
        """Free memory buffers"""
        PyMem_Free(self.sol_idx_m)
        PyMem_Free(self.cl_idx_m)
        PyMem_Free(self.cl_hval_m)
        PyMem_Free(self.cl_pos_m)
        PyMem_Free(self.cl_cost_m)
        PyMem_Free(self.rcl_idx_m)

    def pick_random(self):
        """Compatible interface with the RCL class"""
        cdef int i, cli
        if self.ncand < 1:
            return None
        else:
            # build the Python object
            i = random.randint(0, self.ncand-1)
            cli = self.rcl_idx_m[i]
            item = self.items[self.cl_idx_m[cli]]
            cand = self.mkcand(item, self.cl_pos_m[cli])
            cand.hval = self.cl_hval_m[cli]
            cand.travel = self.cl_cost_m[cli]
            return cand

@cython.profile(False)
cdef inline double travel_cost(double *table, int nitems, int i, int j):
    return table[i * nitems + j]

# Cost $\Delta c_{ikj}$ of inserting item $k$ between $i$ and $j$
#
cdef double cost_delta_noshift(double *tc, int *sol_idx, int ik,
  int position, int maxpos, int nitems):
    cdef double delta = 0.0
    cdef int ii, ij # solution item indexes

    if position > 0:
        ii = sol_idx[position - 1]
        delta = travel_cost(tc, nitems, ii, ik)
    if maxpos > position:
        ij = sol_idx[position]
        if position > 0:
            delta = delta - travel_cost(tc, nitems, ii, ij)
        delta = delta + travel_cost(tc, nitems, ik, ij)
    return delta

cdef double swap_cost(double *tc, int s1, int e1, int s2, int e2, int nitems):
    """Computes the time delta resulting from reconnecting edges
    (s1, e1), (s2, e2) as (s1, s2) and (e1, e2)"""
    return (-travel_cost(tc, nitems, s1, e1)
        + travel_cost(tc, nitems, s1, s2)
        - travel_cost(tc, nitems, s2, e2)
        + travel_cost(tc, nitems, e1, e2))


class OP_GRASP(grasp.GRASP):
    """OP solver"""

    def mk_solution(self):
        solution = OPSolution()
        start = self.problem.get_start()
        end = self.problem.get_end()
        solution.insert(solution.mk_cand(start, 0), start != end)
        solution.insert(solution.mk_cand(end, 1))
        return solution

    # Optimized RCL builder.
    # Note: this will crash and burn if the solution does not
    # contain the start and end vertex.
    def make_rcl(self, solution, alpha):
        cdef int aidx, idx, pos, maxidx, maxpos, cli, rcli, maxcli, nitems
        cdef double cost, reward, maxcost, h, minh, maxh, threshold
        cdef int *sol_idx
        cdef int *cl_idx
        cdef double *cl_hval
        cdef int *cl_pos
        cdef double *cl_cost
        cdef int *rcl_idx
        cdef double *tc
        cdef FastDistMatrix dm
        cdef double best_h
        cdef int best_cli, best_pos

        # Preparation. Get available items
        avail_items = list(self.available_items(solution))
        maxidx = len(avail_items)
        maxpos = solution.get_size()

        # Allocate memory
        rcl = FastRCL(maxpos, maxidx, solution.mk_cand, avail_items)
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

        # First pass. Compute the cost of all insertions and hval if
        # the move is feasible. Determine bounds.
        maxcost = self.problem.capacity - solution.cost
        minh = 1
        maxh = 0
        aidx = 0  # index in the available items list
        while aidx < maxidx:
            item = avail_items[aidx]
            idx = item.idx  # the global id of the item
            reward = item.reward
            pos = 1
            cli = aidx * (maxpos - 1)   # pos 0 not used
            # if optimum insertion
            best_h = 1.0
            best_cli = -1
            best_pos = -1
            while pos < maxpos:
                cost = cost_delta_noshift(tc, sol_idx, idx, pos, maxpos, nitems)

#                if cost > maxcost:
#                    cl_pos[cli] = -1 # not feasible, not included etc
#                else:
#                    h = reward / (cost + 0.0001)
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
                    h = reward / (cost + 0.0001)
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

        return rcl

    def available_items(self, solution):
        for item in super(OP_GRASP, self).available_items(solution):
            if not solution.contains(item):
                yield item

# these are needed by make_rcl() which is rewritten in the cythonized
# version. Thus deprecated.
#
#    def all_moves(self, solution):
#        l = solution.get_size()
#        for item in self.available_items(solution):
#            for position in xrange(1, l):
#                yield solution.mk_cand(item, position)

#    def feasible(self, cand, solution):
#        if (cand.item.cost + cand.travel > self.problem.capacity -
#                                                            solution.cost):
#            return False
#        else:
#            return True

    def hval(self, cand, solution):
        return cand.item.reward / (cand.item.cost + cand.travel + 0.0001)


class OP_GRASP_T_Common(OP_GRASP):
    """OP solver using trajectory rejoining"""
    def available_items(self, solution):
        for item in self.problem.items:
            if not (solution.contains(item) or self.to.is_kicked(item)):
                yield item


class OP_GRASP_PR_Common(OP_GRASP):
    """OP solver using path relinking"""
    def relink_cl(self, solution, difference):
        l = solution.get_size()
        for idx in difference:
            item = self.problem.items[idx]
            for position in xrange(1, l):
                yield solution.mk_cand(item, position)

    def removal_hval(self, solution, idx):
        cand = solution.get_item(idx)
        return (solution.cost_delta(cand.item, idx, 1) /
            cand.item.reward + 0.0001)

    def local_search(self, solution):
        """Minimize the cost of the solution with 2-opt"""
        cdef int nitems, s1, e1, s2, e2, f, t, best_f, best_t, idx, sol_size
        cdef int tmp
        cdef double delta, improvement
        cdef double *tc
        cdef FastDistMatrix dm
        cdef int *sol_idx

        sol_size = solution.get_size()
        sol_idx = <int *> PyMem_Malloc(sol_size * sizeof(int))
        for idx, cand in solution.get_items():
            sol_idx[idx] = cand.item.idx
        dm = self.problem.distmatrix
        tc = dm.table
        nitems = dm.nitems

        # The OP is not constrained (other than the capacity)
        # which makes search moves like $\lambda$-opt more effective. We
        # only minimize the tour length and rely on the path relinking
        # steps for insertions and removals.
        improvement = -1.0
        while improvement < -0.00001:
            improvement = 0.0
            best_f = -1
            best_t = -1
            f = 1
            while f < sol_size - 2:
                s1 = sol_idx[f - 1]
                e1 = sol_idx[f]
                t = f + 1
                while t < sol_size - 1:
                    s2 = sol_idx[t]
                    e2 = sol_idx[t + 1]
                    delta = swap_cost(tc, s1, e1, s2, e2, nitems)
                    if delta < improvement:
                        improvement = delta
                        best_f = f
                        best_t = t
                    t = t + 1
                f = f + 1
            if best_f > -1:
                solution.subpath_reversal(best_f, best_t)
                while best_t > best_f: # index needs to be updated too
                    tmp = sol_idx[best_f]
                    sol_idx[best_f] = sol_idx[best_t]
                    sol_idx[best_t] = tmp
                    best_f = best_f + 1
                    best_t = best_t - 1


        PyMem_Free(sol_idx)
        return solution

class OP_GRASP_T(OP_GRASP_T_Common, trajectory.CoopGRASPT):
    """Cooperative OP solver using trajectory rejoining"""
    pass

class OP_GRASP_I(OP_GRASP_T_Common, trajectory.IndGRASPT):
    """Independent OP solver (GRILS)"""
    pass

class OP_GRASP_PR(OP_GRASP_PR_Common, relink.CoopGRASPPR):
    """Cooperative OP solver using path relinking"""
    pass

class OP_GRASP_DPR(OP_GRASP_PR_Common, relink.DistribGRASPPR):
    """Cooperative OP solver using path relinking"""
    pass

TEST_GRASP_T=1
TEST_GRASP_I=3
TEST_GRASP_PR=2
TEST_GRASP_DPR=5

def traj_test(argv):
    do_test(argv, TEST_GRASP_T)

def ind_test(argv):
    do_test(argv, TEST_GRASP_I)

def pr_test(argv):
    do_test(argv, TEST_GRASP_PR)

def dpr_test(argv):
    do_test(argv, TEST_GRASP_DPR)

def do_test(argv, testid):
    from mpi4py import MPI
    import fileformat
    import sys
    import monitor
    import opdata
    import os.path

    if len(argv) < 4:
        sys.exit(1)
    fr = fileformat.GOPReader(argv[1])
    dlim = opdata.get_dlim(os.path.basename(argv[1]))
    repeats = int(argv[2])
    iters = int(argv[3])

    matrix = fr.get_distmatrix()
    problem = OPProblem(
        [ OPItem(i, x[0], 0.0, matrix[i])
            for i, x in enumerate(fr.get_scores()) ],
        fr.get_start(),
        fr.get_end(),
        0.0
    )

    try:
        elim_time = int(argv[5])
        reactive_delta = int(argv[6])
    except:
        elim_time = None
        reactive_delta = None

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        if len(argv) > 4:
            logfile = open(argv[4], "w")
        else:
            logfile = sys.stdout
    else:
        logfile = None

    if testid == TEST_GRASP_T:
        monitorf = monitor.monitor_best
        searchclass = OP_GRASP_T
    elif testid == TEST_GRASP_I:
        monitorf = monitor.monitor_best
        searchclass = OP_GRASP_I
    elif testid == TEST_GRASP_PR:
        monitorf = monitor.monitor_pool
        searchclass = OP_GRASP_PR
    elif testid == TEST_GRASP_DPR:
        monitorf = monitor.monitor_distpool
        searchclass = OP_GRASP_DPR

    for d in dlim:
        problem.set_capacity(d)
        for i in xrange(repeats):
            comm.Barrier()
            if rank == 0:
                # control process
                best = monitorf(comm, logfile, {"aux1":d, "aux2": i})
                #best.pretty_print()
            else:
                # search process
                g = searchclass(comm)
                if testid == TEST_GRASP_T:
                    if elim_time is not None:
                        g.ELIMINATION_TIME = elim_time
                        g.ra.REACTIVE_DELTA = reactive_delta
                g.search(problem, iters)

if __name__ == "__main__":
    pass
#    g = OP_GRASP_PR()
#    random.seed(3767070252)
#    solution = g.search(problem, 20)
#    print (solution.encode())

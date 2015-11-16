#!/usr/bin/env python
# cython: profile=False

import grasp
import toptw
import op
import trajectory
import relink
import wireformat

cimport op
cimport toptw

from cpython.mem cimport PyMem_Malloc, PyMem_Free
import cython

class MCTOPTWItem(op.OPItem):
    """MCTOPTW vertex"""
    def __init__(self, idx, reward=0.0, cost=0.0, distvector=[],
                                        ot=[], fee=0.0, mode=1, types=()):
        super(MCTOPTWItem, self).__init__(idx, reward, cost, distvector)
        self.ot = ot
        self.fee = fee
        self.mode = mode
        self.types = types
        self.blacklist = False

    # assumes there are 4 opening times
    def get_ot(self, tour=None):
        if tour is None:
            return self.ot
        if (tour + self.mode) % 2 == 0:
            return [self.ot[1], self.ot[3]]
        else:
            return [self.ot[0], self.ot[2]]

    def get_tw(self, arrival, tour):
        tw = self.get_ot(tour)
        ot, ct = tw[0]
        for tour in tw[1:]:
            if arrival > ct:
                ot, ct = tour
            if arrival < ct:
                break
        return ot, ct

# opening time pairs
#
cdef class FastOpenTimes:
    """Container class for the opening times, flattened for quick lookup"""
    cdef double *table
    cdef int nitems

    def __cinit__(self, items):
        cdef int i, j, k, tidx
        nitems = len(items)
        self.table = <double *> PyMem_Malloc(nitems * 8 * sizeof(double))
        i = 0
        while i < nitems:
            item = items[i]
            tidx = i * 8
            j = 0
            if item.mode == 1:
                for ot, ct in item.ot[:4]:
                    self.table[tidx + j] = ot
                    self.table[tidx + j + 1] = ct
                    j = j + 2
            else:
                k = 2
                for ot, ct in item.ot[:4]:
                    self.table[tidx + j + k] = ot
                    self.table[tidx + j + k + 1] = ct
                    j = j + 2
                    k = -k  # flip the pairs
            i = i + 1
        self.nitems = nitems

    def __dealloc__(self):
        PyMem_Free(self.table)


class MCTOPTWProblem(toptw.TOPTWProblem):
    """MCTOPTW instance"""
    def __init__(self, items=[], startidx=0, endidx=0, capacity = 0.0,
        m=1, budget=0.0, types=()):
        super(MCTOPTWProblem, self).__init__(
            items, startidx, endidx, capacity, m)
        self.budget = budget
        self.types = types
        self.fot = FastOpenTimes(self.items)

    def get_budget(self):
        return self.budget

    def set_budget(self, budget):
        self.budget = budget

    def get_types(self):
        return self.types

    def set_types(self, types):
        self.types = types

    def clear_blacklist(self):
        for item in self.items:
            item.blacklist = False


class MCTOPTWSolution(toptw.TOPTWSolution):
    """Solution for the MCTOPTW."""

    def __init__(self, problem, m=1):
        self.problem = problem
        self.spent = 0.0
        self.types = [ 0 for t in self.problem.get_types() ]
        self.ntypes = len(self.types)
        super(MCTOPTWSolution, self).__init__(m)

    def reset(self):
        super(MCTOPTWSolution, self).reset()
        self.spent = 0.0
        for i in xrange(self.ntypes):
            self.types[i] = 0

    def get_spent(self):
        return self.spent

    def get_typecount(self, t):
        return self.types[t]

    def insert(self, cand, update=True):
        super(MCTOPTWSolution, self).insert(cand, update)
        if update:
            self.spent += cand.item.fee
            for i in xrange(self.ntypes):
                self.types[i] += cand.item.types[i]

    def remove(self, idx, update=True):
        cand = super(MCTOPTWSolution, self).remove(idx, update)
        if update:
            self.spent -= cand.item.fee
            for i in xrange(self.ntypes):
                self.types[i] -= cand.item.types[i]

        return cand

    def copy(self):
        clone = type(self)(self.problem, self.m)
        clone.items = [self.copy_cand(c)
            for i, c in self.get_items()]
        clone.score = self.get_score()
        clone.cost = [ c for c in self.cost ]
        clone.iters = self.get_iters()
        clone.idx = set(self.get_idx())
        clone.m = self.m
        clone.spent = self.get_spent()
        clone.types = [ t for t in self.types ]
        return clone

    def verify(self):
        prov_cost = [0.0] * self.m
        prov_score = 0.0
        prov_types = [0.0] * self.ntypes
        prov_spent = 0.0
        tour = 0
        skip = True
        for i, cand in self.get_items():
            if i > 0:
                prov_cost[tour] += (last_cand.item.travel_cost(cand.item)
                    + cand.item.cost
                    + cand.wait)
                if cand.boundary:
                    if tour == 0 and cand.item.idx != start_idx:
                        prov_score += cand.item.reward
                else:
                    prov_score += cand.item.reward
                    prov_spent += cand.item.fee
                    for i in xrange(self.ntypes):
                        prov_types[i] += cand.item.types[i]
            else:
                start_idx = cand.item.idx
                if tour == 0:
                    prov_score += cand.item.reward
                    prov_spent += cand.item.fee
                    for i in xrange(self.ntypes):
                        prov_types[i] += cand.item.types[i]

            ot, ct = cand.item.get_tw(cand.start, tour)
            if cand.start - ot < -0.00000001  or cand.start - ct > 0.00000001:
                return False, "candidate start is not in nearest TW (%d, %d)"%(
                    cand.index, cand.item.idx)
            if cand.maxshift < -0.00000001:
                return False, "negative maxshift (%d, %d)"%(cand.index,
                    cand.item.idx)
            if cand.boundary:
                tour += 1
            last_cand = cand

        if abs(prov_score - self.get_score()) > 0.0001:
            return False, "Total score mismatch"
        elif abs(prov_spent - self.get_spent()) > 0.0001:
            return False, "Monetary budget mismatch"
        else:
            for tour in xrange(self.m):
                if abs(prov_cost[tour] - self.get_cost(tour)) > 0.0001:
                    return False, "Tour %d cost mismatch"%(tour)
            for t in xrange(self.ntypes):
                if abs(prov_types[t] - self.get_typecount(t)) > 0.0001:
                    return False, "Type %d count mismatch"%(t)
        return True, "OK"


@cython.profile(False)
cdef inline void get_otct(double *otct, int tour, double arrival,
    double *ot, double *ct):
    cdef int step
    step = (tour % 2) * 2
    if arrival > otct[step + 1]:
        ot[0] = otct[step + 4]
        ct[0] = otct[step + 5]
    else:
        ot[0] = otct[step]
        ct[0] = otct[step + 1]


class MCTOPTW_GRASP(toptw.TOPTW_GRASP):
    """MCTOPTW solver"""
    def mk_solution(self):
        m = self.problem.get_m()
        solution = MCTOPTWSolution(self.problem, m)
        start = self.problem.get_start()
        end = self.problem.get_end()
        for tour in xrange(m):
            solution.insert(solution.mk_cand(start, 2*tour, False, tour,
                0), start != end)
            solution.insert(solution.mk_cand(end, 2*tour+1, True, tour))
        solution.score = start.reward
        if start != end:
            solution.score += end.reward
        return solution

    def get_itemconstraints(self, solution):
        """Extract data for filtering items"""
        fullcat = []
        maxcat = self.problem.get_types()
        for t in xrange(solution.ntypes):
            if solution.get_typecount(t) >= maxcat[t]:
                fullcat.append(t)
        maxfee = self.problem.get_budget() - solution.get_spent()
        return fullcat, maxfee

    def available_items(self, solution):
        fullcat, maxfee = self.get_itemconstraints(solution)
        for item in super(MCTOPTW_GRASP, self).available_items(solution):
            if not solution.contains(item):
                if item.fee <= maxfee:
                    banned = False
                    for t in fullcat:
                        if item.types[t] > 0:
                            banned = True
                            break
                    if not banned:
                        yield item

    def make_rcl(self, solution, alpha):
        cdef int aidx, idx, pos, maxidx, maxpos, cli, rcli, maxcli, nitems
        cdef int tour
        cdef double cost, reward, h, minh, maxh, threshold
        cdef double t_ij, t_ik, t_kj, ot, ct, shift, arrival
        cdef int *sol_idx
        cdef int *cl_idx
        cdef double *cl_hval
        cdef int *cl_pos
        cdef double *cl_cost
        cdef int *rcl_idx
        cdef double *tc
        cdef int *sol_tour
        cdef double *sol_maxshift
        cdef double *sol_end
        cdef op.FastDistMatrix dm
        cdef op.FastRCL rcl
        cdef double best_h
        cdef int best_cli, best_pos
        cdef FastOpenTimes fot
        cdef double *otcttable
        cdef double *otct

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

        # Various quick lookup tables
        sol_tour = <int *> PyMem_Malloc(maxpos * sizeof(int))
        sol_maxshift = <double *> PyMem_Malloc(maxpos * sizeof(double))
        sol_end = <double *> PyMem_Malloc(maxpos * sizeof(double))
        tour = 0
        skip = 1
        for idx, cand in solution.get_items():
            sol_idx[idx] = cand.item.idx
            if skip:
                sol_tour[idx] = -1
                skip = 0
            else:
                sol_tour[idx] = tour
                if cand.boundary:
                    tour = tour + 1
                    skip = 1
            sol_maxshift[idx] = cand.maxshift + cand.wait
            sol_end[idx] = cand.end

        # distance lookup table
        dm = self.problem.distmatrix
        tc = dm.table
        nitems = dm.nitems

        # opening times table
        fot = self.problem.fot
        otcttable = fot.table

        # First pass. Compute the cost of all insertions and hval if
        # the move is feasible. Determine bounds.
        minh = 1
        maxh = 0
        aidx = 0  # index in the available items list
        while aidx < maxidx:
            item = avail_items[aidx]
            idx = item.idx  # the global id of the item
            reward = item.reward
            cost = item.cost
            otct = otcttable + (idx * 8)

            pos = 1
            cli = aidx * (maxpos - 1)   # pos 0 not used
            # if optimum insertion
            best_h = 1.0
            best_cli = -1
            best_pos = -1
            while pos < maxpos:
                # no "all positions" version here

                cl_pos[cli] = -1 # not feasible, not included etc
                tour = sol_tour[pos]
                if tour > -1:
                    toptw.cost_delta_parts(tc,
                        sol_idx, idx, pos, maxpos, nitems,
                        &t_ij, &t_ik, &t_kj)
                    arrival = sol_end[pos - 1] + t_ik
                    get_otct(otct, tour, arrival, &ot, &ct)
                    if arrival < ct:
                        wait = ot - arrival
                        if wait < 0:
                            wait = 0
                        shift = t_ik + t_kj - t_ij + cost + wait

                        if shift <= sol_maxshift[pos]:
                            h = reward / (shift + 0.0001)
                            if best_cli == -1 or h > best_h:
                                best_h = h
                                best_cli = cli
                                best_pos = pos

                            # these can be pre-filled
                            cl_idx[cli] = aidx  # not the same as problem index
                            cl_hval[cli] = h
                            cl_cost[cli] = shift

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

        PyMem_Free(sol_tour)
        PyMem_Free(sol_maxshift)
        PyMem_Free(sol_end)
        return rcl

    def hval(self, cand, solution):
        return cand.item.reward / (cand.wait + cand.item.cost + cand.travel + 0.0001)


class MCTOPTW_GRASP_T_Common(MCTOPTW_GRASP, toptw.TOPTW_GRASP_T_Common):
    """MCTOPTW solver using trajectory rejoining"""
    def do_perturb(self, i, solution, beta):
        """Kick one or more items from each tour"""
        # separate tours
        tours = {}
        tour = -1
        for idx, cand in solution.removables():
            if cand.tour != tour:
                tour = cand.tour
                tours[tour] = []
            tours[tour].append((idx, cand))

        available = (self.problem.nitems - solution.get_size() -
            len(self.to.kicked))
        removals = []
        for removables in tours.values():
            kl = trajectory.KickList(solution, removables)
            for j in xrange(kl.kick_count(beta)):
                # shorten elimination time, if the candidates are running out
                elim_time = min(available, trajectory.ELIMINATION_TIME)
                # always remove at least one item
                idx, item = kl.pick_item()
                removals.append(idx)
                if elim_time > 0:
                    self.to.add_item(i, item, elim_time)
                    available -= 1
        # start from the end so the indexes remain valid
        for idx in reversed(sorted(removals)):
            solution.remove(idx)
        self.to.release_items(i)
        return solution


class MCTOPTW_GRASP_PR_Common(MCTOPTW_GRASP, toptw.TOPTW_GRASP_PR_Common):
    """MCTOPTW solver using path relinking"""
    def relink_cl(self, solution, difference):
        """Only return feasible candidates"""
        l = solution.get_size()
        fullcat, maxfee = self.get_itemconstraints(solution)
        for idx in difference:
            item = self.problem.items[idx]
            if item.fee > maxfee:
                continue
            banned = False
            for t in fullcat:
                if item.types[t] > 0:
                    banned = True
                    break
            if banned:
                continue
            skip = False
            tour = 0
            for position in xrange(1, l):
                if not skip:
                    cand = solution.mk_cand(item, position, False, tour)
                    if self.feasible(cand, solution):
                        yield cand
                    if solution.get_item(position).boundary:
                        skip = True
                        tour += 1
                else:
                    skip = False

    def removal_hval(self, solution, idx):
        cand = solution.get_item(idx)
        return (solution.cost_delta(cand.item, idx, 1) + cand.wait /
            cand.item.reward + 0.0001)


class MCTOPTW_GRASP_T(MCTOPTW_GRASP_T_Common, trajectory.CoopGRASPT):
    pass

class MCTOPTW_GRASP_I(MCTOPTW_GRASP_T_Common, trajectory.IndGRASPT):
    pass

class MCTOPTW_GRASP_PR(MCTOPTW_GRASP_PR_Common, relink.CoopGRASPPR):
    pass

class MCTOPTW_GRASP_DPR(MCTOPTW_GRASP_PR_Common, relink.DistribGRASPPR):
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

    if len(argv) < 4:
        sys.exit(1)
    fr = fileformat.MCTOPTWReader(argv[1])
    repeats = int(argv[2])
    iters = int(argv[3])

    matrix = fr.get_distmatrix()
    dlim = fr.get_dlim()
    tours = fr.get_tours()
    problem = MCTOPTWProblem(
        [ MCTOPTWItem(i, x[0][0], x[1], matrix[i], x[2], x[3], x[4], x[5])
            for i, x in enumerate(fr.get_tuples()) ],
        fr.get_start(),
        fr.get_end(),
        dlim,
        tours,
        fr.get_budget(),
        fr.get_types()
    )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        if len(argv) > 4:
            logfile = open(argv[4], "w")
        else:
            logfile = sys.stdout
    else:
        logfile = None

    if testid == op.TEST_GRASP_T:
        monitorf = monitor.monitor_best
        searchclass = MCTOPTW_GRASP_T
    elif testid == op.TEST_GRASP_I:
        monitorf = monitor.monitor_best
        searchclass = MCTOPTW_GRASP_I
    elif testid == op.TEST_GRASP_PR:
        monitorf = monitor.monitor_pool
        searchclass = MCTOPTW_GRASP_PR
    elif testid == op.TEST_GRASP_DPR:
        monitorf = monitor.monitor_distpool
        searchclass = MCTOPTW_GRASP_DPR

    for i in xrange(repeats):
        comm.Barrier()
        if rank == 0:
            # control process
            best = monitorf(comm, logfile, {"aux1":dlim, "aux2": i})
            #best.pretty_print()
        else:
            # search process
            g = searchclass(comm)
            g.search(problem, iters)

class MCTOPTW_GRASP_T_S(MCTOPTW_GRASP_T_Common, trajectory.TrajectoryGRASP):
    pass

class MCTOPTW_GRASP_PR_S(MCTOPTW_GRASP_PR_Common, relink.PathRelinkGRASP):
    pass

if __name__ == "__main__":
    import fileformat
    import sys
    import os.path

    if len(sys.argv) < 3:
        sys.exit(1)
    fr = fileformat.MCTOPTWReader(sys.argv[1])
    iters = int(sys.argv[2])

    matrix = fr.get_distmatrix()
    dlim = fr.get_dlim()
    tours = fr.get_tours()
    problem = MCTOPTWProblem(
        [ MCTOPTWItem(i, x[0][0], x[1], matrix[i], x[2], x[3], x[4], x[5])
            for i, x in enumerate(fr.get_tuples()) ],
        fr.get_start(),
        fr.get_end(),
        dlim,
        tours,
        fr.get_budget(),
        fr.get_types()
    )

    g = MCTOPTW_GRASP_T_S()
    #g = MCTOPTW_GRASP_PR_S()
    import random
    random.seed(3767070252)
    solution = g.search(problem, iters)
    solution.pretty_print()


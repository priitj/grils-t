#!/usr/bin/env python
# cython: profile=False

import grasp
import op
import trajectory
import relink
import wireformat

cimport op

from cpython.mem cimport PyMem_Malloc, PyMem_Free
import cython

TIMETABLE=[-1000, 2, 10, 12, 1000] # hours, starting time given
TT_SIZE=4

# 5 categories, 4 speeds
SPEEDMATRIX = [
  [ 0.5, 0.81, 0.5, 0.81 ], # always busy
  [ 0.5, 0.7, 1, 1.5 ], # morning peak
  [ 0.5, 1.5, 0.5, 1.5 ], # two peaks
  [ 1, 1.5, 0.5, 0.7 ], # evening peak
  [ 1.5, 1.5, 1.5, 1.5 ], # seldom traveled
]

cdef double c_timetable[5]
c_timetable[:] = [-1000, 2, 10, 12, 1000]

cdef double c_speed[5][4]
c_speed[0][:] = [ 0.5, 0.81, 0.5, 0.81 ] # always busy
c_speed[1][:] = [ 0.5, 0.7, 1, 1.5 ] # morning peak
c_speed[2][:] = [ 0.5, 1.5, 0.5, 1.5 ] # two peaks
c_speed[3][:] = [ 1, 1.5, 0.5, 0.7 ] # evening peak
c_speed[4][:] = [ 1.5, 1.5, 1.5, 1.5 ] # seldom traveled

class TDOPItem(op.OPItem):
    """TDOP vertex"""
    def __init__(self, idx, reward=0.0, cost=0.0, distvector=[], arctype=[]):
        super(TDOPItem, self).__init__(idx, reward, cost, distvector)
        self.arctype = arctype

    def travel_cost(self, to_item, departure):
        """Compute travel time piecewise"""
        dist = self.distvector[to_item.idx]
        speeds = SPEEDMATRIX[self.arctype[to_item.idx]]
        now = departure
        for k in xrange(TT_SIZE):
            if departure < TIMETABLE[k + 1]:
                arrival = now + (dist/speeds[k])
                break
        while arrival > TIMETABLE[k + 1]:
            dist = dist - speeds[k] * (TIMETABLE[k + 1] - now)
            k = k + 1
            now = TIMETABLE[k]
            arrival = now + (dist/speeds[k])
        return arrival - departure

    # instead of Verbeeck, et al. algorithm we use the travel time
    # algorithm in reverse.
    def departure_time(self, from_item, arrival):
        """Reverse-engineer departure time from known arrival time"""
        dist = self.distvector[from_item.idx]
        speeds = SPEEDMATRIX[from_item.arctype[self.idx]]
        now = arrival
        for k in xrange(TT_SIZE):
            if arrival > TIMETABLE[3 - k]:
                break
        k = 3 - k
        departure = now - (dist/speeds[k])
        while departure <= TIMETABLE[k]:
            dist = dist - speeds[k] * (now - TIMETABLE[k])
            now = TIMETABLE[k]
            k = k - 1
            departure = now - (dist/speeds[k])
        return departure

cdef class FastArcMatrix:
    cdef:
        double *tctable
        int *attable
        int nitems

    """Container class for the arctypes table"""
    def __cinit__(self, items):
        cdef int i, j, tidx
        nitems = len(items)
        self.tctable = <double *> PyMem_Malloc(nitems * nitems * sizeof(double))
        self.attable = <int *> PyMem_Malloc(nitems * nitems * sizeof(int))
        i = 0
        while i < nitems:
            fromitem = items[i]
            j = 0
            tidx = i * nitems
            while j < nitems:
                self.tctable[tidx] = fromitem.distvector[items[j].idx]
                self.attable[tidx] = fromitem.arctype[items[j].idx]
                j = j + 1
                tidx = tidx + 1
            i = i + 1
        self.nitems = nitems

    def __dealloc__(self):
        PyMem_Free(self.tctable)
        PyMem_Free(self.attable)


class TDOPProblem(grasp.Problem):
    """TDOP instance"""
    def __init__(self, items=[], startidx=0, endidx=0, capacity = 0.0):
        super(TDOPProblem, self).__init__(items, capacity)
        self.startidx = startidx
        self.endidx = endidx
        self.arcmatrix = FastArcMatrix(self.items)

    def get_start(self):
        return self.items[self.startidx]

    def get_end(self):
        return self.items[self.endidx]


class TDOPCandidate(op.OPCandidate):
    """Insertion candidate with an index and travel cost"""
    def __init__(self, item, position):
        super(TDOPCandidate, self).__init__(item, position)
        self.start = None
        self.end = None
        self.shift = 0
        self.maxshift = 0
        self.travelto = 0

    def __repr__(self):
        return "<C: %d %.2f %.2f %.2f %.2f>"%(self.item.idx,
            self.start, self.shift, self.travel, self.maxshift)


class TDOPSolution(op.OPSolution):
    """Solution for the TDOP."""

    def __init__(self, problem):
        self.problem = problem
        super(TDOPSolution, self).__init__()

    def add_cost(self, cand):
        return 0.0

    def del_cost(self, cand):
        return 0.0

    def cost_delta_parts(self, item, position, shift=0):
        """Helper function for determining the change in cost"""
        if position > 0:
            prev = self.items[position-1]
            now = prev.end
            t_ik = prev.item.travel_cost(item, now)
        else:
            t_ik = 0.0
            now = 0.0
        if len(self.items) > position + shift:
            nextc = self.items[position + shift]
            if position > 0:
                if shift == 0:
                    t_ij = prev.travel # cached
                else:
                    t_ij = prev.item.travel_cost(nextc.item, now)
            else:
                t_ij = 0.0
            t_kj = item.travel_cost(nextc.item, now + t_ik + item.cost)
        else:
            t_ij = 0.0
            t_kj = 0.0
        return t_ij, t_ik, t_kj

    def mk_cand(self, item, position, fixstart=None):
        cand = TDOPCandidate(item, position)
        t_ij, t_ik, t_kj = self.cost_delta_parts(item, position)
        cand.travel = t_kj # cache to next item
        cand.travelto = t_ik # copy on insertion, ignored later
        #cand.maxshift = 0
        if fixstart is None:
            now = self.get_item(position - 1).end
            cand.start = now + t_ik
            cand.end = cand.start + item.cost # the Verbeeck benchmark is
                                              # costless though
        else:
            cand.start = fixstart
            cand.end = cand.start + item.cost
        cand.shift = t_ik + t_kj - t_ij + item.cost
        return cand

    def receive_solution(self, comm, source, problem):
        wireformat.Solution.receive_solution(self, comm, source)
        score, cost, iters, items = self.decode()
        self.reset()
        #self.score = score
        self.iters = iters

        # set the tour endpoints
        start = problem.get_start()
        end = problem.get_end()
        endidx = end.idx
        self.insert(self.mk_cand(start, 0, 0), start.idx != end.idx)
        self.insert(self.mk_cand(end, 1))

        # insert the rest
        tour = 0
        is_start = True
        for position, idx in items:
            if is_start:
                is_start = False
                continue
            if problem.items[idx].idx == endidx:
                continue
            cand = self.mk_cand(problem.items[idx], position)
            self.insert(cand)
        self.repair_forward()

    def repair_forward(self):
        """Adjust maxshift for the entire solution"""
        arrival = self.problem.capacity
        last_idx = self.get_size() - 1
        last = self.get_item(last_idx)
        last.maxshift = arrival - last.end
        for i in xrange(last_idx - 1, 0, -1):
            nextc = self.get_item(i)
            departure = last.item.departure_time(nextc.item, arrival)
            nextc.maxshift = departure - nextc.end
            arrival = nextc.start
            last = nextc

    def insert(self, cand, update=True):
        super(TDOPSolution, self).insert(cand, False)
        if update:
            self.score = self.add_score(cand)
            last_idx = self.get_size() - 1
            if cand.index > 0:
                last = self.get_item(cand.index - 1)
                last.travel = cand.travelto
            last = cand
            for i in xrange(cand.index + 1, self.get_size()):
                nextc = self.get_item(i)
                if last.index > cand.index:
                    # costs have changed, need to recalculate
                    last.shift = -(last.end + last.travel)
                    last.start += shift
                    last.end = last.start + last.item.cost
                    last.travel = last.item.travel_cost(nextc.item,
                        last.end)
                    last.shift += last.end + last.travel

                if nextc.index == last_idx:
                    nextc.start += last.shift
                    nextc.end = nextc.start + nextc.item.cost
                    self.cost = nextc.end
                    self.repair_forward()
                    break # redundant
                else:
                    shift = last.shift
                    last = nextc

    def repair_maxshift(self):
        """Remove items until there are no maxshift violations"""
        last_idx = self.get_size() - 1
        bad = True
        #count = 0
        #origsize = last_idx + 1
        while bad and last_idx > 1:
            bad = False
            for i in xrange(last_idx, 0, -1):
                nextc = self.get_item(i)
                if nextc.maxshift < -0.00000001:
                    bad = True
                    #count += 1
                    if i == last_idx:
                        self.remove(i - 1)
                    else:
                        self.remove(i)
                    last_idx = self.get_size() - 1
                    break
        #print "repaired %d/%d"%(count, origsize)

    def check_maxshift(self):
        """Check maxshift without repairing"""
        last_idx = self.get_size() - 1
        for i in xrange(last_idx, 0, -1):
            nextc = self.get_item(i)
            if nextc.maxshift < -0.00000001:
                return False
        return True

    def remove(self, idx, update=True):
        cand = super(TDOPSolution, self).remove(idx, False)
        if update:
            self.score = self.del_score(cand)
            last_idx = self.get_size() - 1
            last = self.get_item(idx - 1)
            for i in xrange(idx, self.get_size()):
                nextc = self.get_item(i)
                if last.index >= idx:
                    # costs have changed, need to recalculate
                    last.shift = -(last.end + last.travel)
                    last.start += shift
                    last.end = last.start + last.item.cost
                    last.travel = last.item.travel_cost(nextc.item,
                        last.end)
                    last.shift += last.end + last.travel
                else:
                    last.travel = last.item.travel_cost(nextc.item,
                        last.end)
                    last.shift = last.end + last.travel - nextc.end

                if nextc.index == last_idx:
                    nextc.start += last.shift
                    nextc.end = nextc.start + nextc.item.cost
                    self.cost = nextc.end
                    self.repair_forward()
                    break # redundant
                else:
                    shift = last.shift
                    last = nextc

        return cand

    def copy_cand(self, cand):
        clone = TDOPCandidate(cand.item, cand.index)
        clone.hval = cand.hval
        clone.travel = cand.travel
        clone.start = cand.start
        clone.end = cand.end
        clone.shift = cand.shift
        clone.maxshift = cand.maxshift
        return clone

    def copy(self):
        clone = type(self)(self.problem)
        clone.items = [self.copy_cand(c)
            for i, c in self.get_items()]
        clone.score = self.get_score()
        clone.cost = self.get_cost()
        clone.iters = self.get_iters()
        clone.idx = set(self.get_idx())
        return clone

    def verify(self):
        prov_cost = 0.0
        prov_score = 0.0
        for i, cand in self.get_items():
            if i > 0:
                prov_cost += (last_cand.item.travel_cost(cand.item,
                    last_cand.end)
                    + cand.item.cost)
                if cand.item.idx != start_idx:
                    prov_score += cand.item.reward
                if (abs(last_cand.end + last_cand.travel - cand.start)
                                                                > 0.00000001):
                    return (False,
                        "last_cand.travel time and arrival mismatch (%d, %d)"%(
                        cand.index, cand.item.idx))
            else:
                start_idx = cand.item.idx
                prov_score += cand.item.reward
            if cand.maxshift < -0.00000001:
                return False, "negative maxshift (%d, %d)"%(cand.index,
                    cand.item.idx)
            last_cand = cand

        if abs(prov_score - self.get_score()) > 0.0001:
            return False, "total score mismatch"
        elif abs(prov_cost - self.get_cost()) > 0.0001:
            return False, "total cost mismatch"
        return True, "OK"

@cython.profile(False)
cdef inline double travel_cost(double *tctable, int nitems, int i, int j,
        int *attable, double departure):
    cdef:
        int k
        double arrival = c_timetable[4]
        double dist = tctable[i * nitems + j]
        double *speeds = c_speed[attable[i * nitems + j]]
        double now = departure

    k = 0
    while k < 4:
        if departure < c_timetable[k + 1]:
            arrival = now + (dist/speeds[k])
            break
        k = k + 1

    while arrival > c_timetable[k + 1]:
        dist = dist - speeds[k] * (c_timetable[k + 1] - now)
        k = k + 1
        now = c_timetable[k]
        arrival = now + (dist/speeds[k])
    return arrival - departure

cdef void cost_delta_parts(double *tc, int *at, double *sol_trav,
  double *sol_end, int *sol_idx, int ik, int position, int maxpos, int nitems,
  double cost, double *t_ij, double *t_ik, double *t_kj):
    cdef:
        int ii, ij # solution item indexes
        double now
        int prev

    if position > 0:
        prev = position - 1
        ii = sol_idx[prev]
        now = sol_end[prev]
        t_ik[0] = travel_cost(tc, nitems, ii, ik, at, now)
    else:
        t_ik[0] = 0.0
        now = 0.0
    if maxpos > position:
        ij = sol_idx[position]
        if position > 0:
            t_ij[0] = sol_trav[prev]
        else:
            t_ij[0] = 0.0
        t_kj[0] = travel_cost(tc, nitems, ik, ij, at, now + t_ik[0] + cost)
    else:
        t_ij[0] = 0.0
        t_kj[0] = 0.0


class TDOP_GRASP(op.OP_GRASP):
    """TDOP solver"""
    def mk_solution(self):
        solution = TDOPSolution(self.problem)
        start = self.problem.get_start()
        end = self.problem.get_end()
        solution.insert(solution.mk_cand(start, 0, 0), start.idx != end.idx)
        solution.insert(solution.mk_cand(end, 1))
        solution.score = start.reward
        if start != end:
            solution.score += end.reward
        solution.repair_forward()
        return solution

    def make_rcl(self, solution, alpha):
        cdef int aidx, idx, pos, maxidx, maxpos, cli, rcli, maxcli, nitems
        cdef double cost, reward, h, minh, maxh, threshold
        cdef double t_ij, t_ik, t_kj, shift
        cdef int *sol_idx
        cdef int *cl_idx
        cdef double *cl_hval
        cdef int *cl_pos
        cdef double *cl_cost
        cdef int *rcl_idx
        cdef double *tc
        cdef int *ac
        cdef double *sol_maxshift
        cdef double *sol_trav
        cdef double *sol_end
        cdef op.FastRCL rcl
        cdef FastArcMatrix am
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

        # Various quick lookup tables
        sol_maxshift = <double *> PyMem_Malloc(maxpos * sizeof(double))
        sol_end = <double *> PyMem_Malloc(maxpos * sizeof(double))
        sol_trav = <double *> PyMem_Malloc(maxpos * sizeof(double))
        for idx, cand in solution.get_items():
            sol_idx[idx] = cand.item.idx
            sol_maxshift[idx] = cand.maxshift
            sol_end[idx] = cand.end
            sol_trav[idx] = cand.travel

        # distance and arc type lookup tables
        am = self.problem.arcmatrix
        tc = am.tctable
        ac = am.attable
        nitems = am.nitems

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
            pos = 1
            cli = aidx * (maxpos - 1)   # pos 0 not used
            # if optimum insertion
            best_h = 1.0
            best_cli = -1
            best_pos = -1
            while pos < maxpos:
                # no "all positions" version here

                cl_pos[cli] = -1 # not feasible, not included etc
                cost_delta_parts(tc, ac, sol_trav,
                    sol_end, sol_idx, idx, pos, maxpos, nitems,
                    cost, &t_ij, &t_ik, &t_kj)
                shift = t_ik + t_kj - t_ij + cost

                if shift <= sol_maxshift[pos]:
                    if shift < 0.0001:
                        h = reward / 0.0001
                    else:
                        h = reward / shift
                    if best_cli == -1 or h > best_h:
                        best_h = h
                        best_cli = cli
                        best_pos = pos

                    # these can be pre-filled
                    cl_idx[cli] = aidx  # not the same as problem index
                    cl_hval[cli] = h
                    cl_cost[cli] = t_kj

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

        PyMem_Free(sol_maxshift)
        PyMem_Free(sol_trav)
        PyMem_Free(sol_end)
        return rcl

    def feasible(self, cand, solution):
        #if cand.maxshift < 0:
        #    return False
        nextc = solution.get_item(cand.index)
        if (cand.shift > nextc.maxshift):
            return False
        else:
            return True

    def hval(self, cand, solution):
        return cand.item.reward / max(cand.shift, 0.0001)


class TDOP_GRASP_T_Common(TDOP_GRASP, op.OP_GRASP_T_Common):
    """TDOP solver using trajectory rejoining"""
    def do_perturb(self, i, solution, beta):
        """Kick one or more items from the solution"""
        available = (self.problem.nitems - solution.get_size() -
            len(self.to.kicked))
        kl = trajectory.KickList(solution, list(solution.removables()))
        removals = []
        for j in xrange(kl.kick_count(beta)):
            # shorten elimination time, if the candidates are running out
            elim_time = min(available, self.ELIMINATION_TIME)
            # always remove at least one item
            idx, item = kl.pick_item()
            removals.append(idx)
            if elim_time > 0:
                self.to.add_item(i, item, elim_time)
                available -= 1
        # start from the end so the indexes remain valid
        for idx in reversed(sorted(removals)):
            solution.remove(idx)
        solution.repair_maxshift()
        self.to.release_items(i)
        return solution


class TDOP_GRASP_PR_Common(TDOP_GRASP, op.OP_GRASP_PR_Common):
    """TDOP solver using path relinking"""
    def relink_cl(self, solution, difference):
        """Only return feasible candidates"""
        l = solution.get_size()
        for idx in difference:
            item = self.problem.items[idx]
            for position in xrange(1, l):
                cand = solution.mk_cand(item, position)
                if self.feasible(cand, solution):
                    yield cand

    # XXX: clone of TOPTW method
    def relink_step(self, solution, difference, intersection):
        """Relinking using only legal moves"""
        cand = self.best_relink_insert(solution, difference)
        skip_idx = set(intersection)
        while cand is None:  # no feasible insertions
            # pick item to remove
            idx = self.best_relink_remove(solution, skip_idx)
            if idx is None:
                return None # no more steps possible
            solution.remove(idx)
            solution = self.local_search(solution) # trim any slack created
            if not solution.check_maxshift(): # keep removing?
                continue
            cand = self.best_relink_insert(solution, difference)
        solution.insert(cand)
        return solution

    def removal_hval(self, solution, idx):
        cand = solution.get_item(idx)
        t_ij, t_ik, t_kj = solution.cost_delta_parts(cand.item, idx, 1)
        return (max(t_ik + t_kj - t_ij + cand.item.cost, 0.0001) /
            cand.item.reward + 0.0001)

    def local_search(self, solution):
        return solution

class TDOP_GRASP_T(TDOP_GRASP_T_Common, trajectory.CoopGRASPT):
    pass

class TDOP_GRASP_I(TDOP_GRASP_T_Common, trajectory.IndGRASPT):
    pass

class TDOP_GRASP_PR(TDOP_GRASP_PR_Common, relink.CoopGRASPPR):
    pass

class TDOP_GRASP_DPR(TDOP_GRASP_PR_Common, relink.DistribGRASPPR):
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
    fr = fileformat.TDOPReader(argv[1], argv[2])
    repeats = int(argv[3])
    iters = int(argv[4])

    matrix = fr.get_distmatrix()
    arctype = fr.get_arctype()
    dlim = fr.get_dlim()
    problem = TDOPProblem(
        [ TDOPItem(i, x[0], 0.0, matrix[i], arctype[i])
            for i, x in enumerate(fr.get_scores()) ],
        fr.get_start(),
        fr.get_end(),
        dlim,
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

    if testid == op.TEST_GRASP_T:
        monitorf = monitor.monitor_best
        searchclass = TDOP_GRASP_T
    elif testid == op.TEST_GRASP_I:
        monitorf = monitor.monitor_best
        searchclass = TDOP_GRASP_I
    elif testid == op.TEST_GRASP_PR:
        monitorf = monitor.monitor_pool
        searchclass = TDOP_GRASP_PR
    elif testid == op.TEST_GRASP_DPR:
        monitorf = monitor.monitor_distpool
        searchclass = TDOP_GRASP_DPR

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

class TDOP_GRASP_T_S(TDOP_GRASP_T_Common, trajectory.TrajectoryGRASP):
    pass

class TDOP_GRASP_PR_S(TDOP_GRASP_PR_Common, relink.PathRelinkGRASP):
    pass

if __name__ == "__main__":
    import fileformat
    import sys
    import os.path

    if len(sys.argv) < 4:
        sys.exit(1)
    fr = fileformat.TDOPReader(sys.argv[1], sys.argv[2])
    iters = int(sys.argv[3])

    matrix = fr.get_distmatrix()
    arctype = fr.get_arctype()
    dlim = fr.get_dlim()
    problem = op.OPProblem(
        [ TDOPItem(i, x[0], 0.0, matrix[i], arctype[i])
            for i, x in enumerate(fr.get_scores()) ],
        fr.get_start(),
        fr.get_end(),
        dlim,
    )

    g = TDOP_GRASP_T_S()
    #g = TDOP_GRASP_PR_S()
    import random
    random.seed(3767070252)
    solution = g.search(problem, iters)
    solution.pretty_print()


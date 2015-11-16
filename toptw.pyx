#!/usr/bin/env python
# cython: profile=False

import grasp
import op
import trajectory
import relink
import wireformat

cimport op

from cpython.mem cimport PyMem_Malloc, PyMem_Free

class TOPTWItem(op.OPItem):
    """TOPTW vertex"""
    def __init__(self, idx, reward=0.0, cost=0.0, distvector=[], ot=0, ct=0):
        super(TOPTWItem, self).__init__(idx, reward, cost, distvector)
        self.ot = ot
        self.ct = ct

    def get_tw(self, arrival, tour):
        return (self.ot, self.ct)

class TOPTWProblem(op.OPProblem):
    """TOPTW instance"""
    def __init__(self, items=[], startidx=0, endidx=0, capacity = 0.0,
        m=1):
        super(TOPTWProblem, self).__init__(items, startidx, endidx, capacity)
        self.m = m

    def get_m(self):
        return self.m

    def set_m(self, m):
        self.m = m

class TOPTWCandidate(op.OPCandidate):
    """Insertion candidate with an index and travel cost"""
    def __init__(self, item, position):
        super(TOPTWCandidate, self).__init__(item, position)
        self.start = None
        self.end = None
        self.wait = 0
        self.shift = 0
        self.maxshift = 0
        self.boundary = False
        self.tour = None

    def __repr__(self):
        ot, ct = self.item.get_tw(self.start, self.tour)
        return "<C: %d %d %.2f %.2f %.2f %.2f %.2f>"%(self.item.idx,
            self.tour, self.wait, self.start, ot, ct, self.maxshift)


class TOPTWSolution(op.OPSolution):
    """Solution for the TOPTW."""

    def __init__(self, m=1):
        self.m = m
        super(TOPTWSolution, self).__init__()

    def reset(self):
        super(TOPTWSolution, self).reset()
        self.cost = [0.0] * self.m

    def get_cost(self, tour=None):
        if tour is None:
            return 0.0 # dummy
        else:
            return self.cost[tour]

    def add_cost(self, cand):
#        return self.cost[cand.tour] + cand.shift
        return 0.0

    def del_cost(self, cand):
#        position = cand.index
#        return self.cost[cand.tour] - self.cost_delta(cand.item, position)
        return 0.0

    def cost_delta_parts(self, item, position, shift=0):
        """Helper function for determining the change in cost"""
        if position > 0:
            t_ik = self.items[position-1].item.travel_cost(item)
        else:
            t_ik = 0.0
        if len(self.items) > position + shift:
            if position > 0:
                t_ij = self.items[position-1].item.travel_cost(
                    self.items[position + shift].item)
            else:
                t_ij = 0.0
            t_kj = item.travel_cost(self.items[position + shift].item)
        else:
            t_ij = 0.0
            t_kj = 0.0
        return t_ij, t_ik, t_kj

    def mk_cand(self, item, position, boundary=False, tour=None, fixstart=None):
        cand = TOPTWCandidate(item, position)
        t_ij, t_ik, t_kj = self.cost_delta_parts(item, position)
        cand.travel = t_ik + t_kj - t_ij  # not really travel time
        cand.boundary = boundary
        if tour is None:
            cand.tour = self.get_item(position).tour
        else:
            cand.tour = tour
        is_last = (position == self.get_size())
        if fixstart is None:
            now = self.get_item(position - 1).end
            arrival = now + t_ik
            ot, ct = item.get_tw(arrival, cand.tour)
            cand.wait = max(0, ot - arrival)
            cand.start = arrival + cand.wait
            cand.end = cand.start + item.cost
            if is_last:
                cand.maxshift = ct - cand.start  # as per the formulation
                                                      # in literature, closing
                                                      # time is actually the
                                                      # last allowed start time
            else:
                nextc = self.get_item(position)
                cand.maxshift = min(ct - cand.start,
                    nextc.wait + nextc.maxshift)
            cand.shift = cand.travel + item.cost + cand.wait
        else:
            cand.start = fixstart
            cand.end = cand.start + item.cost
            #cand.wait = 0
            #cand.shift = 0
            #cand.maxshift = 0
        return cand

    def _removables(self):
        remove = False
        for idx, cand in self.get_items():
            if not remove:
                remove = True
                continue
            elif cand.boundary:
                remove = False
                continue
            else:
                yield idx, cand

    def removables(self):
        return list(self._removables())

    def full(self, capacity):
        for tour in xrange(self.m):
            if self.get_cost(tour) > capacity:
                return True
        return False

    def receive_solution(self, comm, source, problem):
        wireformat.Solution.receive_solution(self, comm, source)
        score, cost, iters, items = self.decode()
        self.reset()
        #self.score = score
        self.cost = [0.0] * self.m
        self.iters = iters

        # set the tour endpoints
        start = problem.get_start()
        end = problem.get_end()
        endidx = end.idx
        endpoints = []
        for tour in xrange(self.m):
            self.insert(self.mk_cand(start, 2*tour, False, tour,
                0), start != end)
            cand = self.mk_cand(end, 2*tour+1, True, tour)
            self.insert(cand)
            endpoints.append(cand)

        # insert the rest
        tour = 0
        is_start = True
        for position, idx in items:
            if is_start:
                is_start = False
                continue
            if problem.items[idx].idx == endidx:
                tour += 1
                is_start = True
                continue
            cand = self.mk_cand(problem.items[idx], position, False,
                tour)
            self.insert(cand)
        # fix maxshifts
        for cand in endpoints:
            self.repair_forward(cand)

    def repair_forward(self, cand):
        """Adjust maxshift for items before the last change"""
        last = cand
        for i in xrange(cand.index - 1, 0, -1):
            nextc = self.get_item(i)
            if nextc.boundary:
                break
            ot, ct = nextc.item.get_tw(nextc.start, nextc.tour)
            nextc.maxshift = min(ct - nextc.start,
                last.wait + last.maxshift)
            last = nextc

    def insert(self, cand, update=True):
        super(TOPTWSolution, self).insert(cand, False)
        if update:
            self.score = self.add_score(cand)
            last = cand
            for i in xrange(cand.index + 1, self.get_size()):
                if last.shift <= 0:
                    self.repair_forward(last)
                    break
                nextc = self.get_item(i)
                nextc.shift = max(0, last.shift - nextc.wait)
                nextc.wait = max(0, nextc.wait - last.shift)
                nextc.start = nextc.start + nextc.shift
                nextc.end = nextc.start + nextc.item.cost
                nextc.maxshift = nextc.maxshift - nextc.shift

                if nextc.boundary:
                    self.cost[cand.tour] = nextc.end
                    self.repair_forward(nextc)
                    break
                else:
                    last = nextc


    def remove(self, idx, update=True):
        cand = super(TOPTWSolution, self).remove(idx, False)
        if update:
            self.score = self.del_score(cand)
            last = self.get_item(idx - 1)
            for i in xrange(idx, self.get_size()):
                nextc = self.get_item(i)
                arrival = last.end + last.item.travel_cost(nextc.item)
                ot, ct = nextc.item.get_tw(arrival, nextc.tour)
                nextc.wait = max(0, ot - arrival)
                nextc.start = arrival + nextc.wait
                nextc.shift = nextc.end
                nextc.end = nextc.start + nextc.item.cost
                nextc.shift -= nextc.end
                nextc.maxshift = nextc.maxshift + nextc.shift

                if nextc.shift <= 0:
                    self.repair_forward(nextc)
                    break
                if nextc.boundary:
                    self.cost[cand.tour] = nextc.end
                    self.repair_forward(nextc)
                    break
                else:
                    last = nextc

        return cand

    def copy_cand(self, cand):
        clone = TOPTWCandidate(cand.item, cand.index)
        clone.hval = cand.hval
        clone.travel = cand.travel
        clone.start = cand.start
        clone.end = cand.end
        clone.wait = cand.wait
        clone.shift = cand.shift
        clone.maxshift = cand.maxshift
        clone.boundary = cand.boundary
        clone.tour = cand.tour
        return clone

    def copy(self):
        clone = type(self)()
        clone.items = [self.copy_cand(c)
            for i, c in self.get_items()]
        clone.score = self.get_score()
        clone.cost = [ c for c in self.cost ]
        clone.iters = self.get_iters()
        clone.idx = set(self.get_idx())
        clone.m = self.m
        return clone

#    def subpath_reversal(self, pos_from, pos_to):
#        tour = self.get_item(pos_from).tour
#        self.cost[tour] += self.swap_cost(pos_from-1,
#            pos_from, pos_to, pos_to+1)
#        self.items = (self.items[:pos_from] +
#                        list(reversed(self.items[pos_from:pos_to+1])) +
#                        self.items[pos_to+1:])
#        # we only repair the index, travel time is recalculated on demand
#        for i in xrange(pos_from, pos_to+1):
#            self.items[i].index = i

    def verify(self):
        prov_cost = [0.0] * self.m
        prov_score = 0.0
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
            else:
                start_idx = cand.item.idx
                if tour == 0:
                    prov_score += cand.item.reward
            ot, ct = cand.item.get_tw(cand.start, cand.tour)
            if cand.start - ot < -0.00000001  or cand.start - ct > 0.00000001:
                return False, "cand.start is not in nearest TW (%d, %d)"%(
                    cand.index, cand.item.idx)
            if cand.maxshift < -0.00000001:
                return False, "negative maxshift (%d, %d)"%(cand.index,
                    cand.item.idx)
            if cand.boundary:
                tour += 1
            last_cand = cand

        if abs(prov_score - self.get_score()) > 0.0001:
            return False, "Total score mismatch"
        else:
            for tour in xrange(self.m):
                if abs(prov_cost[tour] - self.get_cost(tour)) > 0.0001:
                    return False, "tour %d cost mismatch"%(tour)
        return True, "OK"


cdef void cost_delta_parts(double *tc, int *sol_idx, int ik,
  int position, int maxpos, int nitems,
  double *t_ij, double *t_ik, double *t_kj):
    cdef int ii, ij # solution item indexes

    if position > 0:
        ii = sol_idx[position - 1]
        t_ik[0] = op.travel_cost(tc, nitems, ii, ik)
    else:
        t_ik[0] = 0.0
    if maxpos > position:
        ij = sol_idx[position]
        if position > 0:
            t_ij[0] = op.travel_cost(tc, nitems, ii, ij)
        else:
            t_ij[0] = 0.0
        t_kj[0] = op.travel_cost(tc, nitems, ik, ij)
    else:
        t_ij[0] = 0.0
        t_kj[0] = 0.0


class TOPTW_GRASP(op.OP_GRASP):
    """TOPTW solver"""
    def mk_solution(self):
        m = self.problem.get_m()
        solution = TOPTWSolution(m)
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

    def all_moves(self, solution):
        l = solution.get_size()
        for item in self.available_items(solution):
            skip = False
            tour = 0
            for position in xrange(1, l):
                if not skip:
                    yield solution.mk_cand(item, position, False, tour)
                    if solution.get_item(position).boundary:
                        skip = True
                        tour += 1
                else:
                    skip = False

    def make_rcl(self, solution, alpha):
        cdef int aidx, idx, pos, maxidx, maxpos, cli, rcli, maxcli, nitems
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
            ot = item.ot
            ct = item.ct
            pos = 1
            cli = aidx * (maxpos - 1)   # pos 0 not used
            # if optimum insertion
            best_h = 1.0
            best_cli = -1
            best_pos = -1
            while pos < maxpos:
                # no "all positions" version here

                cl_pos[cli] = -1 # not feasible, not included etc
                if sol_tour[pos] > -1:
                    cost_delta_parts(tc,
                        sol_idx, idx, pos, maxpos, nitems,
                        &t_ij, &t_ik, &t_kj)
                    arrival = sol_end[pos - 1] + t_ik
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

    def feasible(self, cand, solution):
        if cand.maxshift < 0:
            return False
        nextc = solution.get_item(cand.index)
        if (cand.shift > nextc.wait + nextc.maxshift):
            return False
        else:
            return True

    def hval(self, cand, solution):
        return cand.item.reward / (cand.wait + cand.item.cost + cand.travel + 0.0001)


class TOPTW_GRASP_T_Common(TOPTW_GRASP, op.OP_GRASP_T_Common):
    """TOPTW solver using trajectory rejoining"""
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


class TOPTW_GRASP_PR_Common(TOPTW_GRASP, op.OP_GRASP_PR_Common):
    """TOPTW solver using path relinking"""
    def relink_cl(self, solution, difference):
        """Only return feasible candidates"""
        l = solution.get_size()
        for idx in difference:
            item = self.problem.items[idx]
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
            cand = self.best_relink_insert(solution, difference)
        solution.insert(cand)
        return solution

    def removal_hval(self, solution, idx):
        cand = solution.get_item(idx)
        return (solution.cost_delta(cand.item, idx, 1) + cand.wait /
            cand.item.reward + 0.0001)

    def local_search(self, solution):
#        improvement = -1.0
#        tours = []
#        for idx, cand in solution.get_items():
#            if cand.boundary:
#                tours.append(idx + 1)
#        while improvement < -0.00001:
#            improvement = 0.0
#            best_f = None
#            best_t = None
#            p_l = 0
#            for l in tours:
#                for f in xrange(p_l+1, l-2):
#                    for t in xrange(f+1, l-1):
#                        delta = solution.swap_cost(f-1, f, t, t+1)
#                        if delta < improvement:
#                            improvement = delta
#                            best_f = f
#                            best_t = t
#                p_l = l
#            if best_f is not None:
#                solution.subpath_reversal(best_f, best_t)
#        return solution
        return solution

class TOPTW_GRASP_T(TOPTW_GRASP_T_Common, trajectory.CoopGRASPT):
    pass

class TOPTW_GRASP_I(TOPTW_GRASP_T_Common, trajectory.IndGRASPT):
    pass

class TOPTW_GRASP_PR(TOPTW_GRASP_PR_Common, relink.CoopGRASPPR):
    pass

class TOPTW_GRASP_DPR(TOPTW_GRASP_PR_Common, relink.DistribGRASPPR):
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
    fr = fileformat.TOPTWReader(argv[1])
    repeats = int(argv[2])
    iters = int(argv[3])
    tours = int(argv[4])

    matrix = fr.get_distmatrix()
    dlim = fr.get_dlim()
    problem = TOPTWProblem(
        [ TOPTWItem(i, x[0][0], x[1], matrix[i], x[2][0][0], x[2][0][1])
            for i, x in enumerate(fr.get_tuples()) ],
        fr.get_start(),
        fr.get_end(),
        dlim,
        tours,
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
        searchclass = TOPTW_GRASP_T
    elif testid == op.TEST_GRASP_I:
        monitorf = monitor.monitor_best
        searchclass = TOPTW_GRASP_I
    elif testid == op.TEST_GRASP_PR:
        monitorf = monitor.monitor_pool
        searchclass = TOPTW_GRASP_PR
    elif testid == op.TEST_GRASP_DPR:
        monitorf = monitor.monitor_distpool
        searchclass = TOPTW_GRASP_DPR

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

class TOPTW_GRASP_T_S(TOPTW_GRASP_T_Common, trajectory.TrajectoryGRASP):
    pass

class TOPTW_GRASP_PR_S(TOPTW_GRASP_PR_Common, relink.PathRelinkGRASP):
    pass

if __name__ == "__main__":
    import fileformat
    import sys
    import os.path

    if len(sys.argv) < 4:
        sys.exit(1)
    fr = fileformat.TOPTWReader(sys.argv[1])
    iters = int(sys.argv[2])
    tours = int(sys.argv[3])

    matrix = fr.get_distmatrix()
    dlim = fr.get_dlim()
    problem = TOPTWProblem(
        [ TOPTWItem(i, x[0][0], x[1], matrix[i], x[2][0][0], x[2][0][1])
            for i, x in enumerate(fr.get_tuples()) ],
        fr.get_start(),
        fr.get_end(),
        dlim,
        tours,
    )

#    if len(sys.argv) < 3:
#        sys.exit(1)
#    fr = fileformat.TOPReader(sys.argv[1])
#    iters = int(sys.argv[2])
#
#    matrix = fr.get_distmatrix()
#    problem = TOPTWProblem(
#        [ op.OPItem(i, x[0], 0.0, matrix[i])
#            for i, x in enumerate(fr.get_scores()) ],
#        fr.get_start(),
#        fr.get_end(),
#        fr.get_dlim(),
#        fr.get_tours()
#    )

    g = TOPTW_GRASP_T_S()
    #g = TOPTW_GRASP_PR_S()
    import random
    random.seed(3767070252)
    solution = g.search(problem, iters)
    solution.pretty_print()


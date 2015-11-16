#!/usr/bin/env python
# cython: profile=False

import random
import wireformat
import grasp

POOL_SIZE=20
QUALIFY_THRESHOLD=0.9   # similarity below this qualifies the solution for
                        # the pool
GUIDING_THRESHOLD=0.9   # similarity below this makes the elite solution
                        # a suitable guiding solution

class ElitePool(object):
    """Pool of elite solutions for the path relinking technique"""
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.pool = []

    def add_solution(self, solution):
        """Add the solution, if it qualifies (or the pool is not full)"""
        l = len(self.pool)
        qualified = False
        if l < self.maxsize:
            qualified = True
        else:
            if solution > self.pool[0]:
                qualified = True
            elif solution > self.pool[l-1]:
                for i in xrange(l):
                    if (self.similarity(solution,
                                        self.pool[i]) < QUALIFY_THRESHOLD):
                        qualified = True
                        break
        if qualified:
            self.update_pool(solution)
            return True
        else:
            return False

    def update_pool(self, solution):
        """Insert a qualified solution; sorted by score"""
        l = len(self.pool)
        for i in xrange(l):
            if self.pool[i] < solution:
                self.pool.insert(i, solution)
                # now trim the pool; most limilar lower scoring solution
                if l + 1 > self.maxsize:
                    remove_idx = l
                    remove_sim = self.similarity(solution, self.pool[l])
                    for j in xrange(i+1, l):
                        sim = self.similarity(solution, self.pool[j])
                        if sim < remove_sim:
                            remove_idx = j
                            remove_sim = sim
                    self.pool.pop(remove_idx)
                return
        self.pool.append(solution)

    def pick_guiding(self, solution):
        """Select the most suitable guiding solution"""
        guiding = [ s for s in self.pool
            if self.similarity(solution, s) < GUIDING_THRESHOLD ]
        if not guiding:
            return solution # no difference, so no relinking will result
        else:
            return random.choice(guiding)

    def get_best(self):
        """Returns the current best solution"""
        if len(self.pool) > 0:
            return self.pool[0]
        else:
            return None

    def similarity(self, solA, solB):
        """Compute the similarity of the solutions"""
        # XXX: fix the dependence on solution object internal structure
        is_size = len(solA.intersect(solB))
        return (2.0 * is_size / (solA.get_size() + solB.get_size()))

class PathRelinkGRASP(grasp.GRASP):
    """Version of GRASP with path relinking"""
    def __init__(self):
        super(PathRelinkGRASP, self).__init__()
        self.init_elites()

    # to be overridden
    def init_elites(self):
        """Create the local elites pool"""
        self.ep = ElitePool(POOL_SIZE)

    def find_alpha(self, i):
        """Random alpha with uniform distribution"""
        # with path relinking, alpha does not contribute as directly
        # to the solution quality (alpha used to produce the guiding
        # solution also contributes), so use random here
        if i == 0:
            return 0.0
        else:
            return random.random()

    # to be overridden
    def relink_cl(self, solution, difference):
        """Generate all candidate moves from difference objects"""
        for idx in difference:
            item = self.problem.items[idx]
            yield solution.mk_cand(item)

    def best_relink_insert(self, solution, difference):
        """Pick the best move to make"""
        best_cand = None
        for cand in self.relink_cl(solution, difference):
            cand.hval = self.hval(cand, solution)
            if best_cand is None or cand.hval > best_cand.hval:
                best_cand = cand
        return best_cand

    # to be overridden
    def removal_hval(self, solution, idx):
        """Compute the heuristic value of removing the item"""
        cand = solution.get_item(idx)
        return (cand.item.cost / cand.item.reward)

    def best_relink_remove(self, solution, skipidx):
        """Pick the best object to remove to restore solution feasibility"""
        best_idx = None
        best_hval = 0
        for idx, cand in solution.removables():
            if cand.item.idx in skipidx:
                continue
            hval = self.removal_hval(solution, idx)
            if hval > best_hval:
                best_idx = idx
                best_hval = hval
        return best_idx

    def relink_step(self, solution, difference, intersection):
        """Do one relinking move"""
        cand = self.best_relink_insert(solution, difference)
        solution.insert(cand)
        skip_idx = set(intersection)
        skip_idx.add(cand.item.idx)
        # insertion may have been illegal
        while solution.full(self.problem.capacity):
            # pick item to remove
            idx = self.best_relink_remove(solution, skip_idx)
            if idx is None:
                return None # no more steps possible
            solution.remove(idx)
            solution = self.local_search(solution) # trim any slack created
        return solution

    def get_guiding(self, solution):
        """Obtain a guiding solution for relinking"""
        return self.ep.pick_guiding(solution).copy()

    def submit_pool(self, solution):
        """Submit a solution to the elites pool"""
        return self.ep.add_solution(solution)

    def update_best(self, i, solution, best):
        """Performs mixed path relinking"""
        guiding = self.get_guiding(solution)
        candidate = solution.copy() # elite pool candidate
        difference = guiding.minus(solution)
        intersection = guiding.intersect(solution)
        while difference:
            solution = self.relink_step(solution, difference, intersection)
            if solution is None:
                break
            if solution > candidate:
                candidate = solution.copy()
            # mixed relinking: jump to the other half-path
            tmp = guiding
            guiding = solution
            solution = tmp
            difference = guiding.minus(solution)
            intersection = guiding.intersect(solution)
        candidate.set_iters(i)
        self.submit_pool(candidate)

        # keeping the method signature compatible
        if candidate > best:
            best = candidate
        return best

class CoopGRASPPR(PathRelinkGRASP):
    """Cooperative thread parallel strategy (centralized path relinking)"""
    def __init__(self, comm):
        super(CoopGRASPPR, self).__init__()
        self.comm = comm

    # the pool is centralized, don't need anything here
    def init_elites(self):
        pass

    def get_guiding(self, solution):
        solution.send_guidingreq(self.comm, 0)
        guiding = self.mk_solution()
        guiding.receive_solution(self.comm, 0, self.problem)
        return guiding

    def submit_pool(self, solution):
        ok, msg = solution.verify()
        if not ok:
            solution.pretty_print()
            raise RuntimeError("Broken solution detected: "+msg)
        solution.send_solution(self.comm, 0)
        resp = wireformat.MessagePoolSuccess()
        return resp.receive(self.comm, 0)

    def search(self, problem, maxiters=1000):
        best = super(CoopGRASPPR, self).search(problem, maxiters)
        msg = wireformat.MessageDone()
        msg.send(self.comm, 0)
        return best

class DistribGRASPPR(PathRelinkGRASP):
    """Cooperative thread parallel strategy (distributed path relinking)"""
    def __init__(self, comm):
        super(DistribGRASPPR, self).__init__()
        self.comm = comm

    def submit_pool(self, solution):
        ok, msg = solution.verify()
        if not ok:
            solution.pretty_print()
            raise RuntimeError("Broken solution detected: "+msg)

        # if we have a good candidate, share it with others. If not,
        # just ask for the next remote solution.
        if self.ep.add_solution(solution):
            solution.send_solution(self.comm, 0)
        else:
            msg = wireformat.MessagePoll()
            msg.send(self.comm, 0)

        remote_cand = self.mk_solution()
        remote_cand.receive_solution(self.comm, 0, self.problem)
        if remote_cand.get_size() > 0:
            return self.ep.add_solution(remote_cand)
        else:
            return False

    def search(self, problem, maxiters=1000):
        best = super(DistribGRASPPR, self).search(problem, maxiters)
        msg = wireformat.MessageDone()
        msg.send(self.comm, 0)
        return best



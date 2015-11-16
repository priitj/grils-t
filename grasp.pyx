#!/usr/bin/env python
# cython: profile=False

import random
import wireformat

ALPHA_GRANULARITY = 10
REACTIVE_DELTA = 10

class Item(object):
    """Generic solution item"""
    def __init__(self, idx, reward = 0.0, cost = 0.0):
        self.idx = idx
        self.reward = reward
        self.cost = cost


class Problem(object):
    """Generic knapsack instance"""
    def __init__(self, items = [], capacity = 0.0):
        self.items = items
        self.capacity = capacity
        self.nitems = len(items)

    def set_capacity(self, capacity):
        self.capacity = capacity

class Candidate(object):
    """Insertion candidate"""
    def __init__(self, item):
        self.item = item
        self.hval = None

    def __repr__(self):
        return "<C: %d>"%(self.item.idx)


class RCL(object):
    """Restricted candidate list. Items restricted by threshold"""
    def __init__(self, cand_gen, alpha):
        """Build the RCL"""
        # unfortunately we need two passes here, one to find the bounds
        # and the other to filter. So the moves generator isn't as useful
        # as it could be otherwise
        cl = list(cand_gen)
        if cl:
            minh, maxh = self.bounds(cl)
            threshold = minh + (1 - alpha) * (maxh - minh)
            if threshold > maxh:
                threshold = maxh
            self.rcl = [ cand for cand in cl if cand.hval >= threshold ]
        else:
            self.rcl = []

    def bounds(self, cl):
        """Find upper and lower bounds of the heuristic values"""
        maxh = cl[0].hval
        minh = cl[0].hval
        for cand in cl[1:]:
            if maxh < cand.hval:
                maxh = cand.hval
            if minh > cand.hval:
                minh = cand.hval
        return minh, maxh

    def pick_random(self):
        if not self.rcl:
            return None
        else:
            return random.choice(self.rcl)


class Solution(wireformat.Solution):
    """Sequence of items; with an attached score"""
    def reset(self):
        self.score = 0.0
        self.items = []
        self.cost = 0.0
        self.iters = 0
        self.idx = set()

    def get_size(self):
        """Accessor method for the number of solution items"""
        return len(self.items)

    def get_score(self):
        """Accessor method for score"""
        return self.score

    def get_cost(self):
        """Accessor method for cost"""
        return self.cost

    def get_iters(self):
        """Accessor method for the number of iterations"""
        return self.iters

    def set_iters(self, i):
        """Set number of iterations needed to reach the solution"""
        self.iters = i

    def get_items(self):
        """Accessor method to get the enumerated items"""
        return enumerate(self.items)

    def get_idx(self):
        """Accessor method to get the contents as set"""
        return self.idx

    def get_item(self, idx):
        """Get a single element of the solution"""
        return self.items[idx]

    def _encode(self):
        """Helper function to encode OP solutions"""
        self.encode(self.get_score(),
            self.get_cost(),
            self.get_iters(),
            [x[1].item.idx for x in self.get_items()])

    def send_solution(self, comm, dest):
        """Send the solution to the destination"""
        self._encode()
        super(Solution, self).send_solution(comm, dest)

    def send_guidingreq(self, comm, dest):
        """Request a guiding solution from the destination"""
        self._encode()
        super(Solution, self).send_guidingreq(comm, dest)

    def receive_solution(self, comm, source, problem):
        """Receive the solution from a source"""
        super(Solution, self).receive_solution(comm, source)
        score, cost, iters, items = self.decode()
        self.reset()
        self.score = score
        self.cost = cost
        self.iters = iters
        for position, idx in items:
            self.insert(self.mk_cand(problem.items[idx], position), False)

    # to be overriden according to the problem specifics
    def add_score(self, cand):
        """Compute the score after inserting the item"""
        return self.get_score() + cand.item.reward

    # to be overriden according to the problem specifics
    def add_cost(self, cand):
        """Compute the total cost after inserting the item"""
        return self.get_cost() + cand.item.cost

    # to be overriden according to the problem specifics
    def del_score(self, cand):
        """Compute the score after removing the item"""
        return self.get_score() - cand.item.reward

    # to be overriden according to the problem specifics
    def del_cost(self, cand):
        """Compute the total cost after removing the item"""
        return self.get_cost() - cand.item.cost

    # to be overriden according to the problem specifics
    def mk_cand(self, item, position=None):
        """Build an appropriate solution candidate"""
        return Candidate(item)

    # to be overriden according to the problem specifics
    def insert_cand(self, cand):
        self.items.append(cand)

    # to be overriden according to the problem specifics
    def remove_pos(self, idx):
        return self.items.pop(idx)

    # to be overriden according to the problem specifics
    def removables(self):
        """Return all elements that are not compulsory"""
        return self.get_items()

    def full(self, capacity):
        """Check cumulative cost against given capacity"""
        return self.get_cost() > capacity

    def insert(self, cand, update=True):
        """Add an item to the solution"""
        self.insert_cand(cand)
        if update:
            self.score = self.add_score(cand)
            self.cost = self.add_cost(cand)
        self.idx.add(cand.item.idx)

    def remove(self, idx, update=True):
        """Remove an item from the solution"""
        cand = self.remove_pos(idx)
        if update:
            self.score = self.del_score(cand)
            self.cost = self.del_cost(cand)
        self.idx.remove(cand.item.idx)
        return cand

    def contains(self, item):
        return item.idx in self.get_idx()

    def copy(self):
        """Create a copy of the solution"""
        clone = type(self)()
        clone.items = [self.mk_cand(c.item, i)
            for i, c in self.get_items()]
        clone.score = self.get_score()
        clone.cost = self.get_cost()
        clone.iters = self.get_iters()
        clone.idx = set(self.get_idx())
        return clone

    def verify(self):
        """Check the integrity of the solution"""
        prov_cost = 0.0
        prov_score = 0.0
        for i, cand in self.get_items():
            prov_cost += cand.item.cost
            prov_score += cand.item.reward
        if (abs(prov_cost - self.get_cost()) > 0.0001 or
            abs(prov_score - self.get_score()) > 0.0001):
            return False, "Total score or cost mismatch"
        else:
            return True, "OK"


class GRASP(object):
    def __init__(self):
        self.alpha = 0.3
        self.beta = 1.0
        self.problem = None

    def find_alpha(self, iters):
        """Returns the next value of alpha. Depending on the implementation
           this can be a constant, a random number with uniform distribution
           or a learned value (as in Reactive GRASP)."""
        return self.alpha

    def find_beta(self):
        return self.beta

    def make_rcl(self, solution, alpha):
        """Build the restricted candidate list."""
        return RCL(self.feasible_moves(solution), alpha)

    def feasible_moves(self, solution):
        """Generate feasible insert moves with heuristic values"""
        for cand in self.all_moves(solution):
            if self.feasible(cand, solution):
                # dynamic heuristic value
                cand.hval = self.hval(cand, solution)
                yield cand

    # to be overriden according to the problem specifics
    def available_items(self, solution):
        """All items for building the candidate list"""
        return self.problem.items

    # to be overriden according to the problem specifics
    def all_moves(self, solution):
        """Generate all possible moves"""
        for item in self.available_items(solution):
            yield solution.mk_cand(item)

    # to be overriden according to the problem specifics
    def feasible(self, cand, solution):
        """Check if a move produces a feasible solution"""
        return True

    # to be overriden according to the problem specifics
    def hval(self, cand, solution):
        """Heuristic value of the candidate move"""
        return cand.item.reward

    # to be overriden according to the problem specifics
    def mk_solution(self):
        """Initialize an empty solution"""
        return Solution()

    # to be overriden for parallel versions
    def init_best(self):
        """Initialize the best solution"""
        return self.mk_solution()

    # to be overriden for parallel versions
    def update_best(self, i, solution, best):
        """Update the best solution"""
        if solution > best:
            best = solution.copy()
            best.set_iters(i)
        return best

    # to be overriden for more sophisticated GRASP-s
    def update_stats(self, i, alpha, solution, best):
        """Collect statistics about the search"""
        pass

    # to be overriden for more sophisticated GRASP-s
    def local_search(self, solution):
        """Optimize to a local maximum"""
        return solution

    def search(self, problem, maxiters=1000):
        """Run the search procedure."""
        self.problem = problem
        best = self.init_best()
        solution = self.mk_solution()

        for i in xrange(maxiters):
            # construction step
            alpha = self.find_alpha(i)
            while not solution.full(self.problem.capacity):
                rcl = self.make_rcl(solution, alpha)
                cand = rcl.pick_random()
                if cand is None:
                    break # no moves left
                solution.insert(cand)

            solution = self.local_search(solution)
            best = self.update_best(i, solution, best)
            self.update_stats(i, alpha, solution, best)

            # perturbation step
            beta = self.find_beta()
            solution = self.perturb(i, solution, beta)

        return best

    def perturb(self, i, solution, beta):
        """Remove some items from the solution"""
        return self.mk_solution() # canonical GRASP - restart with an empty solution

class ReactiveAlpha(object):
    """Implements the Reactive GRASP technique for determining $\alpha$"""
    def __init__(self, nbins):
        self.nbins = nbins
        self.sums = [0.0] * nbins
        self.cnt = [0] * nbins
        self.q = [1.0] * nbins # by default, each alpha is equally promising
        step = 1.0 / nbins
        self.bounds = [ step * i for i in xrange(1, nbins+1) ]
        self.REACTIVE_DELTA = REACTIVE_DELTA

    def add_sample(self, alpha, score, mscore, iters=0):
        """Update the statistics for alpha values"""
        alpha_i = int(alpha * self.nbins + 0.5)
        self.sums[alpha_i] += score
        self.cnt[alpha_i] += 1
        q_sum = 0.0
        # update relative shares
        for i in xrange(self.nbins):
            count = self.cnt[i]
            if count > 0:
                self.q[i] = ((self.sums[i] / count) /
                                                mscore)**self.REACTIVE_DELTA
            q_sum += self.q[i]
        # update probability distribution
        bound = 0.0
        for i in xrange(self.nbins):
            bound += self.q[i] / q_sum
            self.bounds[i] = bound

    def random_alpha(self):
        """Get alpha using the reactive probability distribution"""
        r = random.random()
        for i in xrange(self.nbins):
            if r <= self.bounds[i]:
                return float(i) / self.nbins
        return 1.0


class ReactiveGRASP(GRASP):
    """Basic GRASP with learning the best $\alpha$ value added"""
    def __init__(self):
        super(ReactiveGRASP, self).__init__()
        self.ra = ReactiveAlpha(ALPHA_GRANULARITY)

    def find_alpha(self, i):
        if i == 0:
            return 0.0 # greedy construction
        else:
            return self.ra.random_alpha()

    def update_stats(self, i, alpha, solution, best):
        self.ra.add_sample(alpha,
            solution.get_score(),
            best.get_score(),
            i)


class IndependentGRASP(GRASP):
    """Independent thread parallel strategy"""
    def __init__(self, comm):
        super(IndependentGRASP, self).__init__()
        self.comm = comm

    def update_best(self, i, solution, best):
        if solution > best:
            solution.send_solution(self.comm, 0)
            best.receive_solution(self.comm, 0, self.problem)
        return best

    def search(self, problem, maxiters=1000):
        best = super(IndependentGRASP, self).search(problem, maxiters)
        # send signal that we're done
        msg = wireformat.MessageDone()
        msg.send(self.comm, 0)
        return best


class GRASP_Knapsack(IndependentGRASP):
    """finds 0-1 knapsack approximate solutions. For testing"""
    def feasible(self, cand, solution):
        if solution.contains(cand.item):
            return False
        elif cand.item.cost > self.problem.capacity - solution.get_cost():
            return False
        else:
            return True

    def hval(self, cand, solution):
        return cand.item.reward / cand.item.cost


if __name__ == "__main__":
    from mpi4py import MPI
    import monitor

    # p07 from http://web.archive.org/web/20140910220241/http://people.sc.fsu.edu/~jburkardt/datasets/knapsack_01/
    profits = map(float, [ 135, 139, 149, 150, 156, 163, 173, 184,
        192, 201, 210, 214, 221, 229, 240 ])
    weights = map(float, [ 70, 73, 77, 80, 82, 87, 90, 94,
        98, 106, 110, 113, 115, 118, 120 ])
    problem = Problem(
        [ Item(i, x[0], x[1]) for i, x in enumerate(zip(profits, weights)) ],
        750.0
    )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        # control process
        best = monitor.monitor_best(comm)
        best.pretty_print()
    else:
        # search process
        g = GRASP_Knapsack(comm)
        g.search(problem, 20)


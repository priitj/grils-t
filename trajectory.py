#!/usr/bin/env python
# cython: profile=False

import random
import wireformat
import grasp

ELIMINATION_TIME = 1
ELIM_SCORE_BIAS = 2.0


class TimeOut(object):
    """Keep track of eliminated items"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.kicked = set()
        self.queue = {}

    def is_kicked(self, item):
        """Check if an item is currently eliminated"""
        return (item.idx in self.kicked)

    def add_item(self, i, item, timer):
        """Add an eliminated item"""
        self.kicked.add(item.idx)
        release = i + timer
        if self.queue.has_key(release):
            self.queue[release].append(item.idx)
        else:
            self.queue[release] = [item.idx]

    def release_items(self, i):
        """Release items which have been eliminated long enough"""
        released = self.queue.get(i, [])
        if released:
            for idx in released:
                self.kicked.remove(idx)
            del self.queue[i]

class KickList(object):
    """Items to eliminate with a probability distribution"""
    def __init__(self, solution, removables):
        self.items = [c[1].item for c in removables]
        self.idx = [c[0] for c in removables]
        self.nitems = len(self.items)
        self.hvect = [solution.del_score(c[1]) for c in removables]
        self.weights = self.assign_weights(self.items, self.hvect, self.nitems)

    def assign_weights(self, items, hvect, cnt):
        """Compute the weights for elimination candidates"""
        weights = [1.0] * cnt
        if cnt > 0:
            lowest = 0
            for idx in xrange(1, cnt):
                if hvect[idx] < hvect[lowest]:
                    lowest = idx
            weights[lowest] = ELIM_SCORE_BIAS
        return weights

    def kick_count(self, beta):
        """Compute the number of items to be eliminated"""
        cnt = int((self.nitems * beta) + 0.5)
        if cnt < 1 and self.nitems > 0:
            cnt = 1
        return cnt

    def pick_item(self):
        """Select one item to be eliminated"""
        r = random.random()
        sum_weights = sum(self.weights)
        bound = 0.0
        picked = -1 # default to last item, in case of floating point mishaps
        for i in xrange(self.nitems):
            bound += self.weights[i] / sum_weights
            if r < bound:
                picked = i
                break
        self.weights[picked] = 0.0
        return (self.idx[picked], self.items[picked])

class TrajectoryGRASP(grasp.ReactiveGRASP):
    def __init__(self):
        super(TrajectoryGRASP, self).__init__()
        self.to = TimeOut()
        self.ELIMINATION_TIME = ELIMINATION_TIME

    def find_beta(self):
        """Beta with biased distribution."""
        #return random.random()
        return random.choice([1,0.5,0.25,0.13,0.063,0.031,0.016,0.008])

    def available_items(self, solution):
        for item in super(TrajectoryGRASP, self).available_items(solution):
            if not self.to.is_kicked(item):
                yield item

    def perturb(self, i, solution, beta):
        return self.do_perturb(i, solution, beta)

    def do_perturb(self, i, solution, beta):
        """Kick one or more items from the solution"""
        available = (self.problem.nitems - solution.get_size() -
            len(self.to.kicked))
        kl = KickList(solution, list(solution.removables()))
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
        self.to.release_items(i)
        return solution


class CoopGRASPT(TrajectoryGRASP):
    """Cooperative thread parallel strategy (trajectory rejoins)"""
    def __init__(self, comm):
        super(CoopGRASPT, self).__init__()
        self.comm = comm
        self.rejoin = None
        self.iterdelta = 0 # keep track of solution age

    def update_best(self, i, solution, best):
        if solution > best:
            ok, msg = solution.verify()
            if not ok:
                solution.pretty_print()
                raise RuntimeError("Broken solution detected: "+msg)
            solution.set_iters(self.iterdelta + i)
            solution.send_solution(self.comm, 0)
            best = solution.copy()
        else:
            # ask for the global best even if there was no improvement locally
            # the monitor should have a solution, but we're being extra
            # paranoid and include an empty one that can init the monitor
            msg = wireformat.MessagePoll()
            msg.send(self.comm, 0)
        remote_best = self.mk_solution()
        remote_best.receive_solution(self.comm, 0, self.problem)
        if remote_best > best:
            self.rejoin = remote_best
            best = remote_best
            self.iterdelta = best.get_iters() - i
        return best

    def perturb(self, i, solution, beta):
        if self.rejoin is not None:
            solution = self.rejoin.copy()
            self.rejoin = None
            self.to.reset() # timeout set no longer valid
        return super(CoopGRASPT, self).perturb(i, solution, beta)

    def search(self, problem, maxiters=1000):
        best = super(CoopGRASPT, self).search(problem, maxiters)
        msg = wireformat.MessageDone()
        msg.send(self.comm, 0)
        return best

class IndGRASPT(grasp.IndependentGRASP, TrajectoryGRASP):
    """Independent thread parallel strategy"""
    pass


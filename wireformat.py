#!/usr/bin/env python
# cython: profile=False

from mpi4py import MPI
import numpy

MAX_MSG = 400
MSG_FMT = numpy.float64
MSG_SOLU = 1
MSG_DONE = 2
MSG_POLL = 3
MSG_GUIDING = 4
MSG_POOLRSP = 5

class Message(object):
    def __init__(self):
        self.buf = numpy.empty(MAX_MSG, dtype=MSG_FMT)
        self.reset()

    def reset(self):
        pass

    def send(self, comm, dest, tag):
        """Send the buffer to the destination"""
        comm.Send(self.buf, dest=dest, tag=tag)

    def receive(self, comm, source, tag):
        """Receive the solution from a source"""
        status = MPI.Status()
        comm.Recv(self.buf, source=source, tag=tag, status=status)
        return status

    def receive_any(self, comm):
        """Receive any message from a source"""
        return self.receive(comm, MPI.ANY_SOURCE, MPI.ANY_TAG)

class MessagePoll(Message):
    def __init__(self):
        self.buf = numpy.array([0, 0, 0, 0], dtype=MSG_FMT)

    def send(self, comm, dest, tag=MSG_POLL):
        super(MessagePoll, self).send(comm, dest, tag)

class MessageDone(Message):
    def __init__(self):
        self.buf = numpy.array([0], dtype=MSG_FMT)

    def send(self, comm, dest, tag=MSG_DONE):
        super(MessageDone, self).send(comm, dest, tag)

class MessagePoolSuccess(Message):
    def __init__(self, success = False):
        self.buf = numpy.array([float(success)], dtype=MSG_FMT)

    def send(self, comm, dest, tag=MSG_POOLRSP):
        super(MessagePoolSuccess, self).send(comm, dest, tag)

    def receive(self, comm, source, tag=MSG_POOLRSP):
        super(MessagePoolSuccess, self).receive(comm, source, tag)
        return (self.buf[0] > 0.5)


class Solution(Message):
    """Sequence of items; with an attached score"""
    def __cmp__(self, other):
        """Directly compare two solution instances"""
        return cmp(self.get_score(), other.get_score())

    def reset(self):
        self.buf[0] = 0.0
        self.buf[1] = 0.0
        self.buf[2] = 0.0
        self.buf[3] = 0.0

    def get_size(self):
        """Accessor method for the number of solution items"""
        return int(self.buf[0])

    def get_score(self):
        """Accessor method for score"""
        return self.buf[1]

    def get_cost(self):
        """Accessor method for cost"""
        return self.buf[2]

    def get_iters(self):
        """Accessor method for the number of iterations"""
        return int(self.buf[3])

    def set_iters(self, i):
        """Set number of iterations needed to reach the solution"""
        self.buf[3] = float(i)

    def _items(self):
        count = int(self.buf[0])
        return map(int, self.buf[4:count+4])

    def get_items(self):
        """Accessor method to get the enumerated items"""
        return enumerate(self._items())

    def get_idx(self):
        """Accessor method to get the contents as set"""
        return set(self._items())

    def encode(self, score, cost, iters, idx):
        """Encode into a MPI-passable representation"""
        size = float(len(idx))
        self.buf = numpy.array([size, score, cost, iters] + idx, dtype=MSG_FMT)

    def decode(self):
        """Decode from a MPI-passable representation"""
        score = self.buf[1]
        cost = self.buf[2]
        iters = self.buf[3]
        items = enumerate(self._items())
        return score, cost, iters, items

    def send_solution(self, comm, dest):
        """Send the solution to the destination"""
        self.send(comm, dest, MSG_SOLU)

    def send_guidingreq(self, comm, dest):
        """Send the request for an appropriate guiding solution"""
        self.send(comm, dest, MSG_GUIDING)

    def receive_solution(self, comm, source):
        """Receive the solution from a source"""
        self.receive(comm, source, MSG_SOLU)

    def pretty_print(self):
        """(Mostly) human readable representation"""
        print(self.get_size())
        print(self.get_score())
        print(self.get_cost())
        print(self.get_iters())
        print(list(self.get_items()))

    def copy(self):
        """Create a copy of the solution"""
        clone = type(self)()
        clone.buf = numpy.copy(self.buf)
        return clone

    def minus(self, solution):
        """Return a list of item indexes that are not in the other solution"""
        A = self.get_idx()
        B = solution.get_idx()
        return list(A.difference(B))

    def intersect(self, solution):
        """Return a list of item indexes that intersect the other solution"""
        A = self.get_idx()
        B = solution.get_idx()
        return list(A.intersection(B))

    def verify(self):
        """Check the integrity of the solution"""
        return True, "OK"



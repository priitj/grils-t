#!/usr/bin/env python
# cython: profile=False

import wireformat
import relink
import timeit

def monitor_best(comm, log, auxdata={}):
    """Monitor process for keeping track of the best solution. Also supports
    the rejoining version of cooperative strategy"""
    running = set(range(1, comm.Get_size()))
    best = wireformat.Solution()
    incoming = wireformat.Solution()
    start = timeit.default_timer()
    while len(running) > 0:
        status = incoming.receive_any(comm)
        source = status.Get_source()
        tag = status.Get_tag()
        if tag == wireformat.MSG_DONE:
            running.remove(source)
        elif tag == wireformat.MSG_SOLU or tag == wireformat.MSG_POLL:
            if tag == wireformat.MSG_SOLU:
                if incoming > best:
                    log_improvement(log, incoming, auxdata, start)
                    best = incoming.copy()
            best.send_solution(comm, source)
    return best

class LoggingElitePool(relink.ElitePool):
    """Pool of elite solutions for the path relinking technique"""
    def __init__(self, maxsize, logfile, auxdata):
        super(LoggingElitePool, self).__init__(maxsize)
        self.logfile = logfile
        self.auxdata = auxdata
        self.start = timeit.default_timer()

    def add_solution(self, solution):
        """Add the solution with logging when the best is replaced"""
        if not self.pool or solution > self.pool[0]:
            log_improvement(self.logfile, solution, self.auxdata, self.start)
        return super(LoggingElitePool, self).add_solution(solution)

def monitor_pool(comm, log, auxdata={}):
    """Monitor process to keep the elites pool for cooperative
    path relinking based strategy"""
    running = set(range(1, comm.Get_size()))
    ep = LoggingElitePool(relink.POOL_SIZE + 2*comm.Get_size(), log, auxdata)
    incoming = wireformat.Solution()
    while len(running) > 0:
        status = incoming.receive_any(comm)
        source = status.Get_source()
        tag = status.Get_tag()
        if tag == wireformat.MSG_DONE:
            running.remove(source)
        elif tag == wireformat.MSG_GUIDING:
            guiding = ep.pick_guiding(incoming)
            guiding.send_solution(comm, source)
        elif tag == wireformat.MSG_SOLU:
            success = ep.add_solution(incoming.copy())
            resp = wireformat.MessagePoolSuccess(success)
            resp.send(comm, source)
    return ep.get_best()

def monitor_distpool(comm, log, auxdata={}):
    """Monitor process for distributing pool solutions. Also keeps track
       of the best solution."""
    workers = {}
    for i in xrange(1, comm.Get_size()):
        workers[i] = 0
    best = wireformat.Solution()
    incoming = wireformat.Solution()
    null = wireformat.Solution()
    start = timeit.default_timer()
    queue = []

    while len(workers) > 0:
        status = incoming.receive_any(comm)
        source = status.Get_source()
        tag = status.Get_tag()

        if tag == wireformat.MSG_DONE:
            del workers[source]
        elif tag == wireformat.MSG_SOLU or tag == wireformat.MSG_POLL:
            if tag == wireformat.MSG_SOLU: # add to queue
                if incoming.get_iters() > relink.POOL_SIZE:
                    newest = incoming.copy()
                    queue.append((source, newest))
                    #log.write("accepted "+str((source, len(queue)))+"\n")
                if incoming > best:
                    log_improvement(log, incoming, auxdata, start)
                    best = incoming.copy()

            # get from queue
            qpos = workers[source]
            sent = False
            while qpos < len(queue):
                if queue[qpos][0] != source:
                    queue[qpos][1].send_solution(comm, source)
                    workers[source] = qpos + 1
                    sent = True
                    break
                qpos += 1
            if not sent:
                null.send_solution(comm, source)
                workers[source] = qpos
            #log.write("sent "+str((sent, source, workers, len(queue)))+"\n")

    return best

def log_improvement(logfile, solution, auxdata, start):
    """Log the score and iteration with the test parameters"""
    if logfile is not None:
        elapsed = timeit.default_timer() - start
        logfile.write("%d\t%d\t%.8f\t%d\t%.8f\n"%(
            auxdata.get("aux1", 0),
            auxdata.get("aux2", 0),
            solution.get_score(),
            solution.get_iters(),
            elapsed))


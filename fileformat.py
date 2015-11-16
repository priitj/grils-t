#!/usr/bin/env python
# cython: profile=False

import math
import os.path

class OPReader(object):
    """Generic OP problem reader"""
    def __init__(self, filename):
        self.filename = filename
        self.data = None

    def _read(self):
        """Read the data on-demand"""
        if self.data is None:
            self.initdata()
            try:
                f = open(self.filename)
            except IOError:
                f = None
            if f is not None:
                self.readdata(f)
                f.close()

    # override this in an inheriting class
    def initdata(self):
        """Initialize the data available from the instance"""
        self.data = {
            "vertices" : 0,
            "start" : 0,
            "end" : 0,
            "distmatrix" : [],
            "scores" : []
        }

    # override this in an inheriting class
    def readdata(self, f):
        """Read the instance data from file"""
        raise NotImplementedError("File format reader not implemented")

    def get_vertices(self):
        """Return the number of vertices in the problem instance"""
        self._read()
        return self.data["vertices"]

    def get_start(self):
        """Return the index (from 0) of the start vertex"""
        self._read()
        return self.data["start"]

    def get_end(self):
        """Return the index (from 0) of the end vertex"""
        self._read()
        return self.data["end"]

    def get_distmatrix(self):
        """Return the 2-dimensional array of distances between vertices"""
        self._read()
        return self.data["distmatrix"]

    def get_scores(self):
        """Return the array of score vectors"""
        self._read()
        return self.data["scores"]


class GOPReader(OPReader):
    """Read the problems in the format published by J. Silberholz
    for the 2-PIA algorithm (http://josilber.scripts.mit.edu/gop.zip)"""
    def initdata(self):
        """Initialize a GOP problem instance data"""
        super(GOPReader, self).initdata()
        self.data["attributes"] = 0

    def readline(self, f):
        return f.readline().strip().split(" ")

    def readdata(self, f):
        """Read a GOP problem data"""
        vcnt, acnt, start, end = map(int, self.readline(f)[:4])
        self.data["vertices"] = vcnt
        self.data["attributes"] = acnt
        self.data["start"] = start
        self.data["end"] = end

        if start < 0 or start >= vcnt:
            raise IndexError("Invalid start vertex")

        if end < 0 or end >= vcnt:
            raise IndexError("Invalid start vertex")

        for i in xrange(vcnt):
            dist = map(float, self.readline(f))
            if len(dist) != vcnt:
                raise IndexError("Invalid number of distance items")
            self.data["distmatrix"].append(dist)

        for i in xrange(vcnt):
            scores = map(float, self.readline(f))
            if len(scores) != acnt:
                raise IndexError("Invalid score vector length")
            self.data["scores"].append(scores)

    def get_attributes(self):
        """Return the number of attributes"""
        self._read()
        return self.data["attributes"]

class TOPReader(OPReader):
    """TOP file format (http://www.mech.kuleuven.be/en/cib/op/)"""
    def initdata(self):
        """Initialize a TOP problem instance data"""
        super(TOPReader, self).initdata()
        self.data["tours"] = 1
        self.data["dlim"] = 1.0

    def readline(self, f):
        return [ p for p in
            f.readline().strip().replace("\t", " ").split(" ")
            if len(p) > 0 ]

    def euclid(self, p1, p2):
        """2-dimensional distance between (x1, y1) and (x2, y2)"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def readdata(self, f):
        """Read TOP problem data"""
        vcnt = int(self.readline(f)[1])
        m = int(self.readline(f)[1])
        dlim = float(self.readline(f)[1])
        self.data["vertices"] = vcnt
        self.data["start"] = 0
        self.data["end"] = vcnt - 1
        self.data["tours"] = m
        self.data["dlim"] = dlim

        coord = []
        for i in xrange(vcnt):
            x, y, score = map(float, self.readline(f))
            coord.append((x, y))
            self.data["scores"].append([score])

        for i in xrange(vcnt):
            dist = []
            for j in xrange(vcnt):
                if i == j:
                    dist.append(0.0)
                else:
                    dist.append(self.euclid(coord[i], coord[j]))
            self.data["distmatrix"].append(dist)

    def get_dlim(self):
        """Return the distance limit"""
        self._read()
        return self.data["dlim"]

    def get_tours(self):
        """Return the number of tours"""
        self._read()
        return self.data["tours"]


class TOPTWReader(TOPReader):
    """TOPTW file format (http://www.mech.kuleuven.be/en/cib/op/)"""
    def initdata(self):
        """Initialize a TOPTW problem instance data"""
        super(TOPTWReader, self).initdata()
        self.data["costs"] = []
        self.data["tw"] = []

    def rounddigits(self):
        if os.path.basename(self.filename)[:2] == "pr":
            return 2
        else:
            return 1

    def readdata(self, f):
        """Read TOPTW problem data"""
        vcnt = int(self.readline(f)[2])
        #dlim = float(self.readline(f)[1])
        self.data["vertices"] = vcnt
        self.data["start"] = 0
        self.data["end"] = 0
        self.data["tours"] = 1
        self.readline(f)

        coord = []
        for i in xrange(vcnt):
            ll = self.readline(f)
            x = float(ll[1])
            y = float(ll[2])
            cost = float(ll[3])
            score = float(ll[4])
            a = int(ll[6])
            ot = float(ll[7+a])
            ct = float(ll[8+a])
            coord.append((x, y))
            self.data["scores"].append([score])
            self.data["costs"].append(cost)
            self.data["tw"].append([(ot,ct)])

        rnddig = self.rounddigits()
        for i in xrange(vcnt):
            dist = []
            for j in xrange(vcnt):
                if i == j:
                    dist.append(0.0)
                else:
                    dist.append(round(self.euclid(coord[i], coord[j]), rnddig))
            self.data["distmatrix"].append(dist)

        self.data["dlim"] = self.data["tw"][0][0][1]

    def get_tuples(self):
        """Return zipped data about vertices"""
        self._read()
        return zip(self.data["scores"], self.data["costs"], self.data["tw"])


class MCTOPTWReader(TOPTWReader):
    """MCTOPTW file format (http://www.mech.kuleuven.be/en/cib/op/)"""
    def initdata(self):
        """Initialize a MCTOPTW problem instance data"""
        super(MCTOPTWReader, self).initdata()
        self.data["fees"] = []
        self.data["twmode"] = []
        self.data["types"] = []
        self.data["budget"] = 0.0
        self.data["maxn"] = (0,)*10

    def rounddigits(self):
        bn = os.path.basename(self.filename)
        if bn.split("-")[2][:2] == "pr":
            return 2
        else:
            return 1

    def readdata(self, f):
        """Read MCTOPTW problem data"""
        ll = self.readline(f)
        vcnt = int(ll[1])
        self.data["vertices"] = vcnt
        self.data["start"] = 0
        self.data["end"] = 0
        self.data["tours"] = int(ll[0])
        self.data["budget"] = float(ll[2])
        self.data["maxn"] = tuple(map(int, ll[3:13]))

        coord = []

        # first item (start/endpoint)
        ll = self.readline(f)
        x = float(ll[1])
        y = float(ll[2])
        cost = float(ll[3])
        score = float(ll[4])
        ot = float(ll[5])
        ct = float(ll[6])
        coord.append((x, y))
        self.data["scores"].append([score])
        self.data["costs"].append(cost)
        self.data["tw"].append([(ot,ct)]*4)
        self.data["fees"].append(0.0)
        self.data["twmode"].append(1)
        self.data["types"].append((0,)*10)

        for i in xrange(vcnt-1):
            ll = self.readline(f)
            x = float(ll[1])
            y = float(ll[2])
            cost = float(ll[3])
            score = float(ll[4])
            ot1 = float(ll[5])
            ot2 = float(ll[6])
            ot3 = float(ll[7])
            ot4 = float(ll[8])
            ct4 = float(ll[9])
            mode = int(ll[10])
            fee = float(ll[11])
            types = tuple(map(int, ll[12:22]))
            coord.append((x, y))
            self.data["scores"].append([score])
            self.data["costs"].append(cost)
            self.data["tw"].append([
                (ot1,ot2), (ot2,ot3), (ot3,ot4), (ot4,ct4)])
            self.data["fees"].append(fee)
            self.data["twmode"].append(mode)
            self.data["types"].append(types)

        rnddig = self.rounddigits()
        for i in xrange(vcnt):
            dist = []
            for j in xrange(vcnt):
                if i == j:
                    dist.append(0.0)
                else:
                    dist.append(round(self.euclid(coord[i], coord[j]), rnddig))
            self.data["distmatrix"].append(dist)

        self.data["dlim"] = self.data["tw"][0][0][1]

    def get_tuples(self):
        """Return zipped data about vertices"""
        self._read()
        return zip(self.data["scores"],
            self.data["costs"],
            self.data["tw"],
            self.data["fees"],
            self.data["twmode"],
            self.data["types"])

    def get_budget(self):
        """Return the distance limit"""
        self._read()
        return self.data["budget"]

    def get_types(self):
        """Return the number of tours"""
        self._read()
        return self.data["maxn"]


class TDOPReader(TOPReader):
    """TDOP file format (http://www.mech.kuleuven.be/en/cib/op/)"""
    # Note: TOP reader inherited for convenience, tours not needed here
    def __init__(self, filename, arcfilename):
        super(TDOPReader, self).__init__(filename)
        self.read_arctype(arcfilename)

    def read_arctype(self, arcfilename):
        try:
            f = open(arcfilename)
        except IOError:
            f = None
        if f is not None:
            self.readarcdata(f)
            f.close()

    def readarcdata(self, f):
        """Build arc type table"""
        self.arctype = []
        while 1:
            try:
                ll = self.readline(f)
            except:
                ll = []
            if not ll:
                break
            self.arctype.append(map(int, ll))

    def readdata(self, f):
        """Read TDOP problem data"""
        vcnt = int(self.readline(f)[1])
        self.readline(f) # skip m
        dlim = float(self.readline(f)[1])
        self.data["vertices"] = vcnt
        self.data["start"] = 0
        self.data["end"] = vcnt - 1
        self.data["arctype"] = []
        self.data["dlim"] = dlim

        coord = []
        for i in xrange(vcnt):
            x, y, score = map(float, self.readline(f))
            coord.append((x, y))
            self.data["scores"].append([score])

        for i in xrange(vcnt):
            dist = []
            for j in xrange(vcnt):
                if i == j:
                    dist.append(0.0)
                else:
                    dist.append(self.euclid(coord[i], coord[j])/5.0)
            self.data["distmatrix"].append(dist)
            self.data["arctype"].append(self.arctype[i])

    def get_arctype(self):
        """Return the n x n array of arc types between vertex pairs"""
        self._read()
        return self.data["arctype"]


"""The bli module provides the BaselineIterator class which handles mapping CASA antenna IDs to array indices.

Auxilliary functions have been rehoused here from utils.py since we shouldn't need CASA-specific modules for any of this.
"""

def invert_map(d):
    return dict((v,k) for (k,v) in d.iteritems())

def triangle_l(l):
    return [(l[i], l[j])
            for j in range(len(l))
            for i in range(j)]

def triangle(n):
    return [(i,j) for j in range(n) for i in range(j)]

def upper_triangle(n):
    return [(i,j) for j in range(n) for i in range(j+1, n)]

def ij(i, j):
    return (i,j) if i <= j else (j, i)


class BaselineIterator(object):
    """A class to handle mapping CASA antenna IDs to array indices used by the fringe fitting tools.

We allow users to specify arbitraty antenna IDs to use in fringe
fitting (these are the physical or 'p' indices) but within our analysis
we renumber them as 0..n used as array indices (these are the effective
or 'e' indices). We also build maps each way, so that the rest of the
code can use the functions p_to_e and e_to_p to handle conversions.
    """
    def __init__(self, antennas):
        self.antenna_list = sorted(antennas)
        self.n_antennas = len(antennas)
        self.baselines = triangle_l(self.antenna_list)
        self.e_baselines = triangle_l(range(self.n_antennas))
        self.ant_ind_map = dict(zip(self.antenna_list, range(self.n_antennas)))
        self.ant_inv_map = invert_map(self.ant_ind_map)
    def p_to_e(self, p):
        return self.ant_ind_map[p]
    def e_to_p(self, e):
        return self.ant_inv_map[e]
    @staticmethod
    def sort_e_index(i, j):
        return (i, j) if i < j else (j, i)
    def get_sign_and_indices(self, p1, p2):
        """Look up the e-indices of two antennas given their
p-indices. Swapping the order of antennas in a baseline changes the
sign of the phase of the visibility, so we return a sign value as
well."""
        if (p1, p2) in self.baselines:
            sgn, ind = 1, self.baselines.index((p1, p2))
        elif (p2, p1) in self.baselines:
            sgn, ind = -1, self.baselines.index((p2, p1))
        else:
            raise ValueError(
                "No baseline {}--{}(=p) (baselines, {} antennas, {})".format(
                    p1, p2, self.baselines, self.antenna_list))
        (i, j) = self.e_baselines[ind]
        return sgn, (i, j)
    def iter(self):
        return iter(zip(self.baselines, self.e_baselines))
    def iterate_e_baselines_to(self, ref_e_antenna):
        """We iterate over all e_antennas, skipping the reference antenna."""
        for j in range(self.n_antennas):
            if j == ref_e_antenna:
                continue
            else:
                yield j

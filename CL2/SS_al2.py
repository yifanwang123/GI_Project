import random

class PermutationGroup:
    def __init__(self, n):
        self.n = n
        self.permutations = []

    def add_permutation(self, perm):
        if len(perm) != self.n:
            raise ValueError("Permutation length must be equal to n")
        if not self.is_permutation(perm):
            raise ValueError("Invalid permutation")
        self.permutations.append(perm)

    def is_permutation(self, perm):
        if len(perm) != self.n:
            return False
        seen = [False] * self.n
        for p in perm:
            if p < 0 or p >= self.n or seen[p]:
                return False
            seen[p] = True
        return True

    def identity(self):
        return list(range(self.n))

def multiply_permutations(p1, p2):
    if len(p1) != len(p2):
        raise ValueError("Permutations must be of same length")
    return [p1[i] for i in p2]

def inverse_permutation(p):
    n = len(p)
    inv = [0] * n
    for i in range(n):
        inv[p[i]] = i
    return inv

class SchreierSims:
    def __init__(self, n):
        self.n = n
        self.group = PermutationGroup(n)
        self.base = []
        self.schreier_trees = []
        self.generators = []

    def random_schreier_sims(self, base):
        self.base = base
        self.schreier_trees = []
        for b in base:
            self.schreier_trees.append(self.orbit_and_transversal(b))

    def orbit_and_transversal(self, b):
        orbit = [b]
        transversal = {b: self.group.identity()}
        queue = [b]
        while queue:
            alpha = queue.pop(0)
            for perm in self.group.permutations:
                beta = perm[alpha]
                if beta not in orbit:
                    orbit.append(beta)
                    transversal[beta] = multiply_permutations(transversal[alpha], perm)
                    queue.append(beta)
        return (orbit, transversal)

    def add_permutation(self, perm):
        self.group.add_permutation(perm)
        self.generators.append(perm)

    def update_with_partition(self, partition):
        partition_blocks = partition.cell_dict
        new_permutations = []

        
        for color_id, nodes in partition_blocks.items():
            permutation = list(range(len(partition.cls)))
            for i in range(len(nodes)):
                permutation[nodes[i]] = nodes[(i + 1) % len(nodes)]
            new_permutations.append(permutation)

        for perm in new_permutations:
            self.add_permutation(perm)
        self.random_schreier_sims(self.base)

    def get_generators(self):
        return self.generators

def initialize_schreier_sims_with_partitions(partition1, partition2):
    n = len(partition1.cls)
    schreier_sims = SchreierSims(n)

    # Create permutations that map elements of partition1 to partition2
    permutation = [0] * n
    for i in range(n):
        permutation[partition1.cls[i]] = partition2.cls[i]

    # Add permutation to Schreier-Sims
    schreier_sims.add_permutation(permutation)
    schreier_sims.random_schreier_sims(list(range(n)))
    return schreier_sims

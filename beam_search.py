class node:
    def __init__(self, label=None):
        # [(n1, 2), (n2, 1), (n5, 9)]
        self.reachable_nodes = []
        self.cum_cost = 0

        self.closed_list = []
        self.opened_list = []
        self.label = label

    def get_top_n(self, n):
        sreach = self.reachable_nodes
        sreach.sort(key = lambda x: x[1])
        return sreach[:n]

    # pretty print reachables
    def pp_r(self, node):
        return [(i.label, j) for i, j in node.reachable_nodes]

def beam_search(self, node, beam_width: int = 1):
    for _ in range(0):
        print(i)
    return node


if __name__ == "__main__":
    # Initialize all nodes (A to S)
    A = node("A")
    B = node("B")
    C = node("C")
    D = node("D")
    E = node("E")
    F = node("F")
    G = node("G")
    H = node("H")
    I = node("I")
    J = node("J")
    K = node("K")
    L = node("L")
    M = node("M")
    N = node("N")
    S = node("S")

    # Define the edges based on arrow directions
    A.reachable_nodes = [(J, 19), (S, 26)]
    B.reachable_nodes = [(A, 2), (I, 20)]
    C.reachable_nodes = []
    D.reachable_nodes = [(G, 19)]
    E.reachable_nodes = [(B, 12), (H, 28)]
    F.reachable_nodes = [(G, 22), (M, 11), (L, 35)]
    G.reachable_nodes = []
    H.reachable_nodes = [(F, 5), (K, 33)]
    I.reachable_nodes = [(N, 7), (C, 31), (H, 8)]
    J.reachable_nodes = [(N, 30)]
    K.reachable_nodes = [(D, 3), (G, 14)]
    L.reachable_nodes = [(C, 9)]
    M.reachable_nodes = [(G, 16)]
    N.reachable_nodes = [(L, 18)]
    S.reachable_nodes = [(I, 6)]



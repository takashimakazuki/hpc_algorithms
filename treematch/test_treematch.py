import networkx as nx
import numpy as np
from unittest import TestCase, main as ut_main
import matplotlib.pyplot as plt


from treematch import (
    list_all_possible_groups,
    get_arity,
    extend_comm_matrix,
    graph_of_incompatibility,
    get_independent_set,
)


def gen_tree():
    G = nx.Graph()
    nx.add_path(G, ["d0_0", "d1_0"])
    nx.add_path(G, ["d0_0", "d1_1"])
    nx.add_path(G, ["d1_0", "d2_0"])
    nx.add_path(G, ["d1_0", "d2_1"])
    nx.add_path(G, ["d1_0", "d2_2"])
    nx.add_path(G, ["d1_1", "d2_3"])
    nx.add_path(G, ["d1_1", "d2_4"])
    nx.add_path(G, ["d1_1", "d2_5"])

    nx.add_path(G, ["d2_0", "d3_0"])
    nx.add_path(G, ["d2_0", "d3_1"])
    nx.add_path(G, ["d2_1", "d3_2"])
    nx.add_path(G, ["d2_1", "d3_3"])
    nx.add_path(G, ["d2_2", "d3_4"])
    nx.add_path(G, ["d2_2", "d3_5"])
    nx.add_path(G, ["d2_3", "d3_6"])
    nx.add_path(G, ["d2_3", "d3_7"])
    nx.add_path(G, ["d2_4", "d3_8"])
    nx.add_path(G, ["d2_4", "d3_9"])
    nx.add_path(G, ["d2_5", "d3_10"])
    nx.add_path(G, ["d2_5", "d3_11"])
    return G


comm_matrix = np.array(
    [
        [0, 1000, 10, 1, 100, 1, 1, 1],
        [1000, 0, 1000, 1, 1, 100, 1, 1],
        [10, 1000, 0, 1000, 1, 1, 100, 1],
        [1, 1, 1000, 0, 1, 1, 1, 100],
        [100, 1, 1, 1, 0, 1000, 10, 1],
        [1, 100, 1, 1, 1000, 0, 1000, 1],
        [1, 1, 100, 1, 10, 1000, 0, 1000],
        [1, 1, 1, 100, 1, 1, 1000, 0],
    ]
)


# def test_aggregate_comm_matrix():
#     comm_matrix = np.array([[1, 10, 1, 10], [1, 10, 1, 10], [1, 10, 1, 10], [1, 10, 1, 10]])
#     group = [set([0, 1]), set([2, 3])]
#     res = aggregate_comm_matrix(comm_matrix, group)
#     print(res)


class TestTreematch(TestCase):
    def test_get_arity(self):
        tree = gen_tree()
        k = get_arity(tree, 2)
        self.assertEqual(k, 2)  # Gのdepth=2の頂点から木の葉方向につながっている頂点の個数は2

    def test_list_all_possible_groups(self):
        tree = gen_tree()
        groups = list_all_possible_groups(tree, comm_matrix, 3)
        self.assertEqual(len(groups), 28)  # 8C2=28

    def test_extend_comm_matrix(self):
        """
        virturl processes: 4
        communucation matrix: 4x4
        current depth: 2
        arity of upper node: 3
        """
        tree = gen_tree()
        comm_matrix = np.array(
            [[1, 10, 1, 10], [1, 10, 1, 10], [1, 10, 1, 10], [1, 10, 1, 10]]
        )
        m = extend_comm_matrix(tree, comm_matrix, 2)
        self.assertEqual(m.shape, (6, 6))

    def test_graph_of_incompatibility(self):
        l = [
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 6),
            (1, 7),
            (2, 3),
            (2, 4),
            (2, 5),
            (2, 6),
            (2, 7),
            (3, 4),
            (3, 5),
            (3, 6),
            (3, 7),
            (4, 5),
            (4, 6),
            (4, 7),
            (5, 6),
            (5, 7),
            (6, 7),
        ]
        graph = graph_of_incompatibility(l)
        nx.draw(graph, with_labels=True)
        plt.show()
    
    def test_get_independent_set(self):
        graph = nx.Graph()
        nx.add_path(graph, ['0-1', '0-2'])
        nx.add_path(graph, ['0-1', '0-3'])
        nx.add_path(graph, ['0-2', '0-3'])
        nx.add_path(graph, ['1-2', '1-3'])
        nx.add_path(graph, ['1-2', '2-3'])
        nx.add_path(graph, ['1-3', '2-3'])
        nx.add_path(graph, ['0-1', '1-2'])
        nx.add_path(graph, ['0-1', '1-3'])
        nx.add_path(graph, ['0-2', '1-2'])
        nx.add_path(graph, ['0-2', '2-3'])
        nx.add_path(graph, ['0-3', '1-3'])
        nx.add_path(graph, ['0-3', '2-3'])
        indep_set = get_independent_set(graph, seed=0xbeaf)
        self.assertEqual(indep_set, [(2, 3), (0, 1)])
        


if __name__ == "__main__":
    ut_main()

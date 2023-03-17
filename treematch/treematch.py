import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import itertools

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

comm_matrix_4procs = np.array(
    [
        [0, 1, 100, 0],
        [1, 0, 100, 1],
        [100, 100, 0, 100],
        [0, 1, 100, 0],
    ]
)

comm_matrix_5procs = np.array(
    [
        [  0,   1, 100,   0,   0],
        [  1,   0, 100,   1,   1],
        [100, 100,   0, 100, 100],
        [  0,   1, 100,   0,   0],
        [  0,   1, 100,   0,   0],
    ]
)

comm_matrix_6procs = np.array(
    [
        [  0,   1, 100,   0,   0, 0],
        [  1,   0, 100,   1,   1, 0],
        [100, 100,   0, 100, 100, 0],
        [  0,   1, 100,   0,   0, 0],
        [  0,   1, 100,   0,   0, 0],
        [  0,   0,   0,   0,   0, 0],
    ]
)


def get_arity(tree, depth):
    if depth < 0:
        raise ValueError("depth must be equal or greater than 0")
    adj = tree.adj[f"d{depth}_0"]  # 隣接ノード

    # ルートノードの場合とそれ以外の場合でarityの計算方法を分ける
    k = len(adj) if depth == 0 else len(adj) - 1
    return k


def list_all_possible_groups(tree, matrix, depth):
    k = get_arity(tree, depth - 1)
    p = len(matrix)
    return list(itertools.combinations(np.arange(p), k))


def graph_of_incompatibility(l, comm_matrix):
    def to_str(val):
        if type(val) == 'numpy.int64':
            return val.astype('str')
        return str(val)

    graph = nx.Graph()
    result = []
    for i in range(len(l)):
        for j in range(i+1, len(l)):
            if len(set(l[i]).intersection(set(l[j]))) >= 1:
                result.append((l[i], l[j]))
                node1 = '-'.join([to_str(num) for num in l[i]])
                node2 = '-'.join([to_str(num) for num in l[j]])
                nx.add_path(graph, [node1, node2])
    
    # weight(グルーピングの評価値)の計算
    for node in graph.nodes:
        vprocs = list(map(int, node.split('-')))
        perm = list(itertools.permutations(vprocs, 2))

        comm_sum = sum([sum(comm_matrix[v]) for v in vprocs])
        comm_intergroup = sum([comm_matrix[p[0]][p[1]] for p in perm])
        # print(comm_sum, comm_intergroup, comm_sum - comm_intergroup)
        graph.nodes[node]['weight'] = comm_sum - comm_intergroup

    return graph


def get_independent_set(graph, seed=0xbeaf):
    ### NetworkX default algorithm
    # indep_set = nx.maximal_independent_set(graph, seed=seed)

    ### Smallest values first algorithm
    indep_set = []
    vertices = sorted([{'name': k, 'weight': w} for k, w in nx.get_node_attributes(graph, 'weight').items()], key=lambda v: v['weight'])
    while len(vertices) > 0:
        v = vertices.pop(0)
        indep_set.append(v['name'])
        adj = [k for k in graph.adj[v['name']].keys()]
        vertices = [v for v in vertices if v['name'] not in adj]

    return [tuple(map(int, s.split('-'))) for s in indep_set]


def group_procs(tree, matrix, depth):
    l = list_all_possible_groups(tree, matrix, depth)
    if len(l) == 1:
        return l

    graph = graph_of_incompatibility(l, matrix)
    return get_independent_set(graph, seed=3)


def extend_comm_matrix(tree, matrix, depth):
    p = len(matrix)
    k = get_arity(tree, depth - 1)
    padding = k - p % k
    return np.pad(matrix, [(0, padding), (0, padding)])


def aggregate_comm_matrix(matrix, group):
    def sum_mat(i, j):
        s = 0
        for el_i in group[i]:
            for el_j in group[j]:
                s += matrix[el_i][el_j]
        return s

    n = len(group)
    r = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                r[i][j] = 0
            else:
                r[i][j] = sum_mat(i, j)
    return r

def remove_virtual_group(groups, actual_groups_num):
    return [tuple([proc for proc in g if proc < actual_groups_num]) for g in groups]

def treematch(tree, matrix, depth):
    # groups[1]...groups[depth-1]に各階層でのグループ分けを格納する
    # groups[0]は利用しない
    groups = np.empty(depth, dtype=list)  # tuple[][]
    groups[0] = []

    for d in range(depth - 1, 0, -1):
        p = len(matrix)
        k = get_arity(tree, d - 1)
        padding = 0
        if p % k != 0:
            matrix = extend_comm_matrix(tree, matrix, d)
        print(matrix)
        groups[d] = remove_virtual_group(group_procs(tree, matrix, d), p)
        matrix = aggregate_comm_matrix(matrix, groups[d])

    return groups


def main():
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


    depth = 4
    groups = treematch(G, comm_matrix, depth)
    depth1_nodes = G.adj['d0_0']
    for vproc, node in zip(groups[1][0], depth1_nodes):
        G.nodes[node]['vproc'] = vproc
    
    for d in range(1, depth-1):
        # vprocがセットされている頂点の子頂点を求める
        # 子頂点に対して，vprocをセットする groupsから
        nodes = [node_tuple[0] for node_tuple in G.nodes.data('vproc') if node_tuple[1] is not None and str(node_tuple[0]).startswith(f'd{d}_')]
        for i, node in enumerate(nodes):
            # 深さdのある頂点について，その子頂点のvprocをセットする
            children = [n for n in G.adj[node] if str(n).startswith(f'd{d+1}_')]
            node_vproc = G.nodes[node]['vproc']
            child_vprocs = groups[d+1][node_vproc]

            for i, child_vproc in enumerate(child_vprocs):
                G.nodes[children[i]]['vproc'] = child_vproc

    

    pos = graphviz_layout(G, prog='dot')
    nx.draw(G, pos=pos, with_labels=True, node_size=1000, font_color='white')
    node_labels = nx.get_node_attributes(G,'vproc')
    node_labels_pos = dict([ (k, (v[0], v[1]-20)) for k, v in pos.items() ])
    nx.draw_networkx_labels(G, pos=node_labels_pos, labels=node_labels)
    plt.margins(0.1)
    plt.show()


if __name__ == "__main__":
    main()

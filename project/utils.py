from pygraphblas import Matrix


def is_graph_undirected(graph: Matrix):
    return graph.tril().transpose().iseq(graph.triu())

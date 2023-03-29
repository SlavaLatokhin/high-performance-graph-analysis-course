from typing import List

import pygraphblas as gb
from pygraphblas import Matrix

from project.utils import is_graph_undirected


def count_of_triangles_for_each_vertex(graph: Matrix) -> List[int]:
    assert is_graph_undirected(graph)
    graph_square = graph.mxm(graph, semiring=gb.INT64.PLUS_TIMES, mask=graph.S)
    result = graph_square.reduce_vector()
    result.assign_scalar(0, mask=result, desc=gb.descriptor.C)
    return [num // 2 for num in result.vals]

def count_triangles_cohen(graph: Matrix) -> int:
    assert is_graph_undirected(graph)
    result = graph.tril().mxm(graph.triu(), semiring=gb.INT64.PLUS_TIMES, mask=graph)
    return result.reduce_int() // 2

def count_triangles_sandia(graph: Matrix) -> int:
    assert is_graph_undirected(graph)
    triu = graph.triu()
    result = triu.mxm(triu, semiring=gb.INT64.PLUS_TIMES, mask=triu)
    return result.reduce_int()
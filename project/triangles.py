from typing import List

import pygraphblas as gb
from pygraphblas import Matrix

from project.utils import is_graph_undirected


def count_of_triangles_for_each_vertex(graph: Matrix) -> List[int]:
    """
    A function that calculates for each vertex of an undirected
    graph the number of triangles in which it participates.

    :param graph: adjacency boolean matrix of a graph.
    :return: a list where for each vertex it is indicated how many triangles it participates in.
    """
    assert is_graph_undirected(graph)
    graph_square = graph.mxm(graph, semiring=gb.INT64.PLUS_TIMES, mask=graph.S)
    result = graph_square.reduce_vector()
    result.assign_scalar(0, mask=result, desc=gb.descriptor.C)
    return [num // 2 for num in result.vals]


def count_triangles_cohen(graph: Matrix) -> int:
    """
    Kohen's algorithm. Calculates the number of triangles of an undirected graph.

    :param graph: adjacency boolean matrix of a graph.
    :return: the number of triangles in the graph.
    """
    assert is_graph_undirected(graph)
    result = graph.tril().mxm(graph.triu(), semiring=gb.INT64.PLUS_TIMES, mask=graph)
    return result.reduce_int() // 2


def count_triangles_sandia(graph: Matrix) -> int:
    """
    Sandia algorithm. Calculates the number of triangles of an undirected graph.

    :param graph: adjacency boolean matrix of a graph.
    :return: the number of triangles in the graph.
    """
    assert is_graph_undirected(graph)
    triu = graph.triu()
    result = triu.mxm(triu, semiring=gb.INT64.PLUS_TIMES, mask=triu)
    return result.reduce_int()

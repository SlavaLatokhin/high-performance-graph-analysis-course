from typing import List, Tuple
from math import inf

import pygraphblas as gb
from pygraphblas import Matrix


def bellman_ford_for_vertex(graph: Matrix, start_vertex: int) -> List[int]:
    """
    A function of finding the shortest paths in a directed graph from a given vertex.

    :param graph: adjacency boolean matrix of a graph.
    :param start_vertex: vertex from which the search for shortest paths begins.
    :return: a list where for each vertex the distance to it from the specified starting vertex is indicated.
    """
    return bellman_ford_for_vertexes(graph, [start_vertex])[0][1]


def bellman_ford_for_vertexes(
    graph: Matrix, start_vertexes: List[int]
) -> List[Tuple[int, List[int]]]:
    """
    A function for finding the shortest paths in a directed graph of several given vertexes.

    :param graph: adjacency boolean matrix of a graph.
    :param start_vertexes: a list of vertexes from which the search for shortest paths begins.
    :return: a list of pairs: a vertex, and an array, where for each
    vertex the distance to it from the specified one is specified.
    """
    ncols = graph.ncols
    for i in range(ncols):
        graph[i, i] = 0
    dist = Matrix.sparse(gb.FP64, nrows=len(start_vertexes), ncols=ncols)
    for i, v in enumerate(start_vertexes):
        dist[i, v] = 0
    for _ in range(ncols - 1):
        dist.mxm(graph, semiring=gb.FP64.MIN_PLUS, out=dist)

    if dist.isne(dist.mxm(graph, semiring=gb.FP64.MIN_PLUS)):
        raise RuntimeError("A negative weight cycle is found in the graph")
    return [
        (v, [dist.get(i, j, default=inf) for j in range(graph.ncols)])
        for i, v in enumerate(start_vertexes)
    ]


def floyd_warshall(graph: Matrix) -> List[Tuple[int, List[int]]]:
    """
    A function of finding the shortest paths in a directed graph for all pairs of vertexes.

    :param graph: adjacency boolean matrix of a graph.
    :return: a list of pairs: a vertex, and an array, where for each
    vertex the distance to it from the specified one is specified.
    """
    ncols = graph.ncols
    dist = Matrix.sparse(gb.FP64, nrows=ncols, ncols=ncols)
    dist += graph
    for i in range(ncols):
        dist[i, i] = 0
    for k in range(ncols):
        col, row = dist.extract_matrix(col_index=k), dist.extract_matrix(row_index=k)
        value = col.mxm(row, semiring=gb.FP64.MIN_PLUS)
        dist.eadd(value, gb.FP64.MIN, out=dist)

    for k in range(ncols):
        col, row = dist.extract_matrix(col_index=k), dist.extract_matrix(row_index=k)
        value = col.mxm(row, semiring=gb.FP64.MIN_PLUS)
        if dist.isne(dist.eadd(value, gb.FP64.MIN, out=dist)):
            raise RuntimeError("A negative weight cycle is found in the graph")
    return [
        (i, [dist.get(i, j, default=inf) for j in range(graph.ncols)])
        for i in range(ncols)
    ]

from typing import List, Tuple

import pygraphblas as gb
from pygraphblas import Vector, Matrix


def bfs(graph: Matrix, start_vertex: int) -> List[int]:
    """
    The function of traversing a directed graph in width from a given vertex.

    :param graph: adjacency boolean matrix of a graph.
    :param start_vertex: the vertex from which a traversal starts.
    :return: a list where for each vertex it is indicated at which step it is reachable.
    The starting vertex is reachable at the zero step, if the vertex is not reachable,
    then the value of the corresponding cell is -1.
    """
    step = 0
    front = Vector.sparse(gb.BOOL, graph.ncols)
    front[start_vertex] = True
    result = Vector.sparse(gb.INT64, graph.ncols)
    result[start_vertex] = step
    while front.nvals:
        step += 1
        front.vxm(graph, out=front, mask=result.S, desc=gb.descriptor.RSC)
        result.assign_scalar(step, mask=front.S)
    return [result.get(i, default=-1) for i in range(result.size)]


def msbfs(graph: Matrix, start_vertexes: List[int]) -> List[Tuple[int, List[int]]]:
    """
    The function of traversing a directed graph in width from several given.

    :param graph: adjacency boolean matrix of a graph.
    :param start_vertexes: a list of vertexes from which a traversal starts.
    :return: a list of pairs where first element is the start vertex and second
    element is a list where for each vertex it is indicated at which step it is reachable from start vertex.
    """
    front = Matrix.sparse(gb.INT64, len(start_vertexes), graph.ncols)
    parents = Matrix.sparse(gb.INT64, len(start_vertexes), graph.ncols)
    for i, vertex in enumerate(start_vertexes):
        front[i, vertex] = vertex
        parents[i, vertex] = -1
    while front.nvals:
        front.mxm(
            graph,
            semiring=gb.INT64.MIN_FIRST,
            out=front,
            mask=parents.S,
            desc=gb.descriptor.RSC,
        )
        parents.assign(front, mask=front.S)
        front.apply(gb.INT64.POSITIONJ, out=front, mask=front.S)
    return [
        (vertex, [parents.get(i, j, default=-2) for j in range(graph.ncols)])
        for i, vertex in enumerate(start_vertexes)
    ]

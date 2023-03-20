from typing import List, Tuple

import pygraphblas as gb
from pygraphblas import Vector, Matrix


def bfs(graph: Matrix, start_vertex: int) -> List[int]:
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

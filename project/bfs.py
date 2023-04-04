import pygraphblas as gb
from pygraphblas import types, Vector, Matrix


def bfs(graph: Matrix, start_vertex: int):
    step = 0
    front = Vector.sparse(types.BOOL, graph.ncols)
    front[start_vertex] = True
    result = Vector.sparse(types.INT64, graph.ncols)
    result[start_vertex] = step
    while front.nvals:
        step += 1
        front.vxm(graph, out=front, mask=result, desc=gb.descriptor.RSC)
        result.assign_scalar(step, mask=front)
    return [result.get(i, default=-1) for i in range(result.size)]

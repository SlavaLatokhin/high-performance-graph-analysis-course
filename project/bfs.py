import pygraphblas as gb
from pygraphblas import types, Vector, Matrix


def bfs(graph: Matrix, start_vertex: int):
    step = 0
    front = Vector.sparse(types.BOOL, graph.ncols)
    front[start_vertex] = True
    result = Vector.sparse(types.INT64, graph.ncols)
    result[start_vertex] = step

    visited = Vector.sparse(types.BOOL, graph.ncols)
    visited_nvals = visited.nvals
    visited[start_vertex] = True
    while visited.nvals != visited_nvals:
        step += 1
        front.vxm(graph, out=front)

        visited_nvals = visited.nvals
        visited.eadd(front, out=visited)
        mask = front.eadd(result, add_op=types.INT64.GT, mask=front)
        result.assign_scalar(step, mask=mask)
    return [result.get(i, default=-1) for i in range(result.size)]

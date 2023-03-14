import pytest
import pygraphblas as gb
from project.bfs import bfs


@pytest.mark.parametrize(
    "I, J, start_vertex, expected",
    [
        (
            [0, 1, 1, 3, 2, 5],
            [1, 2, 3, 2, 5, 4],
            0,
            [0, 1, 2, 2, 4, 3],
        ),
        (
            [0, 1, 1, 3, 2, 5],
            [1, 2, 3, 2, 5, 4],
            3,
            [-1, -1, 1, 0, 3, 2],
        ),
        (
            [0, 1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 0],
            3,
            [3, 4, 5, 6, 1, 2],
        ),
        (
            [],
            [],
            0,
            [0, -1, -1, -1, -1, -1],
        ),
    ],
)
def test_bfs(I, J, start_vertex, expected):
    size = len(expected)
    matrix = gb.Matrix.from_lists(I, J, nrows=size, ncols=size)
    actual = bfs(matrix, start_vertex)
    assert actual == expected

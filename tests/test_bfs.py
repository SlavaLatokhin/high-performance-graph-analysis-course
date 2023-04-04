import pytest
import pygraphblas as gb
from typing import List

from project.bfs import bfs, msbfs
from tests.utils import load_data


@pytest.mark.parametrize(
    "I, J, vertex_count, start_vertex, expected",
    load_data(
        "bfs_data",
        "test_bfs",
        lambda p: (p["I"], p["J"], p["vertex_count"], p["start_vertex"], p["expected"]),
    ),
)
def test_bfs(I, J, vertex_count: int, start_vertex: int, expected):
    matrix = gb.Matrix.from_lists(I, J, nrows=vertex_count, ncols=vertex_count)
    actual = bfs(matrix, start_vertex)
    assert actual == expected


@pytest.mark.parametrize(
    "I, J, vertex_count, start_vertexes, expected",
    load_data(
        "bfs_data",
        "test_msbfs",
        lambda p: (
            p["I"],
            p["J"],
            p["vertex_count"],
            p["start_vertexes"],
            p["expected"],
        ),
    ),
)
def test_msbfs(I, J, vertex_count: int, start_vertexes: List[int], expected):
    matrix = gb.Matrix.from_lists(I, J, nrows=vertex_count, ncols=vertex_count)
    expected_tuples = [(a, b) for [a, b] in expected]
    actual = msbfs(matrix, start_vertexes)
    assert actual == expected_tuples

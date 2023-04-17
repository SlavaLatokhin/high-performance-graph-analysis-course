import pytest
import pygraphblas as gb
from typing import List

from project.shortest_paths import (
    bellman_ford_for_vertexes,
    bellman_ford_for_vertex,
    floyd_warshall,
)
from tests.utils import load_data


@pytest.mark.parametrize(
    "I, J, V, vertex_count, start_vertexes, expected",
    load_data(
        "shortest_paths_data",
        "test_bellman_ford_for_vertexes",
        lambda p: (
            p["I"],
            p["J"],
            p["V"],
            p["vertex_count"],
            p["start_vertexes"],
            p["expected"],
        ),
    ),
)
def test_bellman_ford_for_vertexes(
    I, J, V, vertex_count: int, start_vertexes: List[int], expected
):
    matrix = gb.Matrix.from_lists(I, J, V, nrows=vertex_count, ncols=vertex_count)
    expected_tuples = [(a, [float(i) for i in b]) for [a, b] in expected]
    actual = bellman_ford_for_vertexes(matrix, start_vertexes)
    print(actual)
    assert actual == expected_tuples


@pytest.mark.parametrize(
    "I, J, V, vertex_count, start_vertex, expected",
    load_data(
        "shortest_paths_data",
        "test_bellman_ford_for_vertex",
        lambda p: (
            p["I"],
            p["J"],
            p["V"],
            p["vertex_count"],
            p["start_vertex"],
            p["expected"],
        ),
    ),
)
def test_bellman_ford_for_vertex(
    I, J, V, vertex_count: int, start_vertex: int, expected
):
    matrix = gb.Matrix.from_lists(I, J, V, nrows=vertex_count, ncols=vertex_count)
    expected_floats = [float(i) for i in expected]
    actual = bellman_ford_for_vertex(matrix, start_vertex)
    assert actual == expected_floats


@pytest.mark.parametrize(
    "I, J, V, vertex_count, expected",
    load_data(
        "shortest_paths_data",
        "test_floyd_warshall",
        lambda p: (p["I"], p["J"], p["V"], p["vertex_count"], p["expected"]),
    ),
)
def test_floyd_warshall(I, J, V, vertex_count: int, expected):
    matrix = gb.Matrix.from_lists(I, J, V, nrows=vertex_count, ncols=vertex_count)
    expected_tuples = [(a, [float(i) for i in b]) for [a, b] in expected]
    actual = floyd_warshall(matrix)
    assert actual == expected_tuples

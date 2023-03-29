import pytest
import pygraphblas as gb

from project.triangles import count_of_triangles_for_each_vertex, count_triangles_sandia, count_triangles_cohen
from tests.utils import load_data


@pytest.mark.parametrize(
    "I, J, vertex_count, expected",
    load_data(
        "triangles_data",
        "test_count_of_triangles",
        lambda p: (p["I"], p["J"], p["vertex_count"], p["expected"]),
    ),
)
def test_count_of_triangles_for_each_vertex(I, J, vertex_count: int, expected):
    matrix = gb.Matrix.from_lists(I, J, nrows=vertex_count, ncols=vertex_count)
    actual = count_of_triangles_for_each_vertex(matrix)
    assert actual == expected

@pytest.mark.parametrize(
    "I, J, vertex_count, expected",
    load_data(
        "triangles_data",
        "test_count_of_triangles",
        lambda p: (p["I"], p["J"], p["vertex_count"], p["expected"]),
    ),
)
def test_count_triangles_cohen(I, J, vertex_count: int, expected):
    matrix = gb.Matrix.from_lists(I, J, nrows=vertex_count, ncols=vertex_count)
    actual = count_triangles_cohen(matrix)
    assert actual == sum(expected) / 3

@pytest.mark.parametrize(
    "I, J, vertex_count, expected",
    load_data(
        "triangles_data",
        "test_count_of_triangles",
        lambda p: (p["I"], p["J"], p["vertex_count"], p["expected"]),
    ),
)
def test_count_triangles_sandia(I, J, vertex_count: int, expected):
    matrix = gb.Matrix.from_lists(I, J, nrows=vertex_count, ncols=vertex_count)
    actual = count_triangles_sandia(matrix)
    assert actual == sum(expected) / 3

from prior._coords import grid_cell_center_to_meters, grid_to_meters, meters_to_grid


def test_grid_to_meters_is_inverse_of_meters_to_grid():
    point = (12.25, 34.75)

    assert grid_to_meters(*meters_to_grid(*point)) == point


def test_grid_cell_center_to_meters_offsets_cell_indices():
    assert grid_cell_center_to_meters(3, 7) == (1.75, 3.75)

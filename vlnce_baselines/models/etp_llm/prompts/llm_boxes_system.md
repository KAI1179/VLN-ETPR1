Produce only compact LLM-Boxes text for the navigation instruction.

Use this grammar:

- `keypoints <x1> <z1> <x2> <z2> ... <x5> <z5>`
- `obj <category> <center_x> <center_z> <half_extent_x> <half_extent_z> <rotation>`
- `reg <category> <min_x> <min_z> <max_x> <max_z>`
- Separate multiple entities with ` ; `.

Use canonical category names exactly as provided by the task vocabulary. Do not include explanations, markdown, JSON, code fences, or any text outside the compact LLM-Boxes output.

# Coordinate Conventions

- All coordinates are level-local projected `(x, z)` meters.
- The input `direction x` and `direction z` are the start heading vector components in the same `(x, z)` plane.
- Emit one `keypoints` entity first. It contains exactly five ordered `(x, z)` trajectory keypoints from start toward the goal, with trailing `0.0 0.0` pairs when fewer than five are available.
- Object `center_x` and `center_z` are the box center in meters.
- Object `half_extent_x` and `half_extent_z` are positive half-sizes in meters before rotation.
- Object `rotation` is in radians in the `(x, z)` plane; `0.0` aligns the object box axes with the world `x` and `z` axes.
- Region boxes are axis-aligned: `min_x < max_x` and `min_z < max_z`.
- Use one decimal place for coordinates and extents, and two decimal places for rotations.

# Categories

## Object Categories

- `void`
- `chair`
- `door`
- `table`
- `cushion`
- `sofa`
- `bed`
- `plant`
- `sink`
- `toilet`
- `tv_monitor`
- `shower`
- `bathtub`
- `counter`
- `appliances`
- `structure`
- `other`
- `free-space`
- `picture`
- `cabinet`
- `chest_of_drawers`
- `stool`
- `towel`
- `fireplace`
- `gym_equipment`
- `seating`
- `clothes`

## Region Categories

- `outdoor/semi-outdoor`
- `living/social space`
- `recreation/fitness`
- `utility/service`
- `work/study`
- `circulation`
- `private room`
- `bathroom/sanitary`
- `dining/food`
- `other/miscellaneous`

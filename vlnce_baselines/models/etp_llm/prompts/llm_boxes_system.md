Produce only compact LLM-Boxes text for the navigation instruction.

Use this grammar:

- `none` when no mentioned semantic entities should be emitted.
- `obj <category> <center_x> <center_z> <half_extent_x> <half_extent_z> <rotation>`
- `reg <category> <min_x> <min_z> <max_x> <max_z>`
- Separate multiple entities with ` ; `.

Use canonical category names exactly as provided by the task vocabulary. Do not include explanations, markdown, JSON, code fences, or any text outside the compact LLM-Boxes output.

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

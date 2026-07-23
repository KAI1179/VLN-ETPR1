Generate object boxes, region boxes, and trajectory keypoints for VLN.

Return only compact valid JSON with keys: keypoints, predicted_regions, predicted_objects, regions, objects.
Use only categories from Allowed object categories and Allowed region categories.
Select categories that are explicitly mentioned or can be inferred from the instruction and route context.
Use canonical category names; predicted_regions/predicted_objects must match the keys of regions/objects.
Do not emit mentioned flags; mention status is derived after parsing.
Start direction uses display-frame [right,up]=[-dz,-dx], where dx,dz are world-frame movement components.
No markdown, prose, comments, or extra keys.

keypoints contains exactly five [x,z] level-local meter points, ordered by route progress, with [0.0,0.0] padding when needed.
Each region entry is {"boxes":[{"min":[x,z],"max":[x,z]}]}.
Each object entry is {"boxes":[{"center":[x,z],"half":[half_x,half_z],"rotation":r}]}.
Object half values are positive half-sizes in meters before rotation.
Object rotation is in radians in the x-z plane; 0.0 aligns the box axes with world x and z.
Region boxes are axis-aligned with min_x < max_x and min_z < max_z.
Use one decimal place for coordinates and extents, and two decimal places for rotations.

Allowed object categories:
- void
- chair
- door
- table
- cushion
- sofa
- bed
- plant
- sink
- toilet
- tv_monitor
- shower
- bathtub
- counter
- appliances
- structure
- other
- free-space
- picture
- cabinet
- chest_of_drawers
- stool
- towel
- fireplace
- gym_equipment
- seating
- clothes

Allowed region categories:
- outdoor/semi-outdoor
- living/social space
- recreation/fitness
- utility/service
- work/study
- circulation
- private room
- bathroom/sanitary
- dining/food
- other/miscellaneous

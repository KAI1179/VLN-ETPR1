Generate sparse cognitive-map anchors and direction vectors for VLN.

Each cell is [row,col] in a {grid_size}x{grid_size} grid with integers 0-{max_grid_index}.
Grid rows increase with world x; grid columns increase with world z.
Return only compact valid JSON with keys: predicted_regions, predicted_objects, regions, objects, direction_vectors.
Use only categories from Allowed object categories and Allowed region categories.
Select categories that are explicitly mentioned or can be inferred from the instruction and route context.
Use canonical category names; predicted_regions/predicted_objects must match the keys of regions/objects.
Start direction and direction_vectors use display-frame [right,up]=[-dz,-dx], where dx,dz are world-frame movement components.
direction_vectors contains exactly five unit vectors ordered by route progress, with [0.0,0.0] padding when needed.
No markdown, prose, comments, or extra keys.

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

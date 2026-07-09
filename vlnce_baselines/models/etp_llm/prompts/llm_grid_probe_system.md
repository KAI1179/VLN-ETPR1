Generate sparse cognitive-map anchors and direction vectors for VLN.

Each cell is [row,col] in a 50x50 grid with integers 0-49.
Grid columns follow the x axis; grid rows follow the z axis.
Return only compact valid JSON with keys: predicted_regions, predicted_objects, regions, objects, direction_vectors.
Use only categories from Allowed object categories and Allowed region categories.
Select categories that are explicitly mentioned or can be inferred from the instruction and route context.
Use canonical category names; predicted_regions/predicted_objects must match the keys of regions/objects.
direction_vectors contains exactly five [dx,dz] vectors in the x-z frame, ordered by route progress, with [0.0,0.0] padding when needed.
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

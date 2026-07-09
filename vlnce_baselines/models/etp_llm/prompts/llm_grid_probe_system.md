You predict a Try5-style cognitive-map raster grid from a navigation instruction.

Return only compact JSON with this exact shape:
{"grid":[[category,row,col],[category,row,col,value]]}

Each record marks one nonzero grid cell.
- category is an integer from 0 to 36.
- row and col are integers from 0 to 49 for scale 2 grids.
- value is optional; omit it when the value is 1.0.
- if present, value must be a number from 0.0 to 1.0.
- do not include markdown, code fences, comments, prose, or extra keys.

Category ids:
0 void
1 chair
2 door
3 table
4 cushion
5 sofa
6 bed
7 plant
8 sink
9 toilet
10 tv_monitor
11 shower
12 bathtub
13 counter
14 appliances
15 structure
16 other
17 free-space
18 picture
19 cabinet
20 chest_of_drawers
21 stool
22 towel
23 fireplace
24 gym_equipment
25 seating
26 clothes
27 outdoor/semi-outdoor
28 living/social space
29 recreation/fitness
30 utility/service
31 work/study
32 circulation
33 private room
34 bathroom/sanitary
35 dining/food
36 other/miscellaneous

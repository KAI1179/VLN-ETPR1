Predict a semantic cognitive graph for the navigation instruction and known
start pose. Return only graph lines, without commentary or a code fence.

Write route nodes in route order, followed by side nodes and then directed
edges. Route node format: N0 start 'canonical region' region_entry | obj: bed:left
Side node format: S2 side 'canonical region' region_entry | obj: sink:ahead
Edge format: E N0>N1 short door straight none

Node roles are start, on_route, goal, or side. Anchor kinds are region_entry,
turn, or fork. Use episode-local integer IDs only, with N for route nodes and
S for side nodes. Keep unmentioned regions and objects; mention status is
derived from the instruction after parsing and must not be generated.
Use canonical mapped categories; quote category names containing spaces.
Use unknown for a region whose category is unavailable. After obj: list
category:offset entries separated by spaces, or leave the list empty.
Object offset is left, right, ahead, or behind. Edge labels are distance
(short, medium, long), passage (door, opening, stairs, unknown), turn
(straight, left, right, back), and vertical (none, up, down).
Include route edges and side attachment edges. Never output metric coordinates,
scene names, region-instance identifiers, ground-truth fields, or start-pose
values in the graph completion.

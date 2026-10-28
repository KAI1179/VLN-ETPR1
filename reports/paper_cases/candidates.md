# Paper qualitative-case shortlist

CPU screening only. No model inference, simulator or GPU is used.

## Definitions and provenance

- **paired_outcome:** ours success=1, zeromap success=0
- **excluded_episodes:** ['92', '1392']
- **exclusion_source:** Explicit CLI-supplied IDs; this script does not verify membership in attn_case.pdf. See README.md for confirmation/provisional status.
- **map_quality:** JSON and schema both valid; rank pooled mentioned-category F1 descending, no arbitrary F1 cutoff
- **mentioned_F1:** 2*(objectTP+regionTP)/(objectPred+regionPred+objectTarget+regionTarget); empty denominator=0
- **mentions:** original GT mention flags and category-presence metrics, not LLM mentioned flags
- **route_complexity:** GT n_turns>=2 AND distinct MP3D GT region instances n_rooms>=2; not an instruction-semantic assertion
- **turn_feature:** dense GT direction cosine<0.8, consecutive triggers merged; original episode_features.py
- **lexical_cues:** \b(?:turn|bear|veer)\s+(?:to\s+the\s+)?(?:left|right)\b|\b(?:take|make)\s+(?:a\s+|the\s+)?(?:left|right)(?:\s+turn)?\b
- **length:** 8<=ours high_level_steps<=15; not low-level steps or trajectory coordinate count
- **ranking:** mentioned pooled F1, GT turns, GT rooms, closeness to10 high-level steps, then episode ID
- **metric_units:** SR and SPL fractions; NE metres; source low-level and high-level step counts kept separately

```json
{
  "ours": "/home/xukai/code/ETP-R1-snapshot/ETP-R1/results/ours_val_unseen_episodes.jsonl",
  "zeromap": "/home/xukai/code/ETP-R1-snapshot/ETP-R1/results/zeromap_val_unseen_episodes.jsonl",
  "features": "/data/xukai/checkpoint/etp-r1-snapshot-data-logs/logs/analysis/features_val_unseen.json",
  "validity": "/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/map_quality/t0_val_unseen_episodes.jsonl",
  "presence": "/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/map_quality/t1_val_unseen_episodes.jsonl",
  "feature_script": "/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/analysis/episode_features.py",
  "quality_script": "/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/analysis/map_quality/t1_presence.py"
}
```

## Filter counts

```json
{
  "paired_episodes": 1839,
  "ours_success_zero_failure": 222,
  "after_explicit_exclusions": 221,
  "valid_json_and_schema": 220,
  "GT_turns_ge2_and_GT_regions_ge2": 111,
  "ours_high_steps_8_to15": 85
}
```

## Top candidates

| Episode | Mentioned F1 (pooled / object / region) | JSON / schema | GT turns / distinct regions / crossings | Lexical turn cues | High-level steps (ours / zero) | Low-level steps (ours / zero) | Ours SR / SPL / NE | Zero SR / SPL / NE |
|---|---|---|---|---|---|---|---|---|
| 532 | 1.0000 / 1.0000 / 1.0000 | valid / valid | 4 / 4 / 3 | 0 | 10 / 9 | 80.0 / 78.0 | 1 / 0.9298 / 1.962m | 0 / 0.0000 / 3.507m |
| 456 | 1.0000 / 1.0000 / 1.0000 | valid / valid | 4 / 3 / 2 | 2 | 11 / 8 | 122.0 / 65.0 | 1 / 0.5590 / 0.901m | 0 / 0.0000 / 18.486m |
| 1455 | 1.0000 / 1.0000 / 1.0000 | valid / valid | 4 / 3 / 2 | 1 | 9 / 9 | 90.0 / 88.0 | 1 / 0.8416 / 2.396m | 0 / 0.0000 / 4.480m |
| 691 | 1.0000 / 1.0000 / 1.0000 | valid / valid | 4 / 2 / 1 | 1 | 9 / 9 | 62.0 / 65.0 | 1 / 0.9002 / 0.788m | 0 / 0.0000 / 3.295m |
| 1444 | 1.0000 / 1.0000 / 1.0000 | valid / valid | 4 / 2 / 1 | 0 | 11 / 15 | 194.0 / 287.0 | 1 / 0.2833 / 2.438m | 0 / 0.0000 / 4.452m |
| 326 | 1.0000 / 1.0000 / 1.0000 | valid / valid | 3 / 5 / 4 | 0 | 10 / 15 | 73.0 / 160.0 | 1 / 0.8732 / 1.300m | 0 / 0.0000 / 25.742m |
| 1043 | 1.0000 / 0.0000 / 1.0000 | valid / valid | 3 / 5 / 4 | 2 | 11 / 13 | 94.0 / 269.0 | 1 / 0.9046 / 2.913m | 0 / 0.0000 / 4.338m |
| 52 | 1.0000 / 1.0000 / 1.0000 | valid / valid | 3 / 4 / 3 | 1 | 10 / 9 | 73.0 / 70.0 | 1 / 1.0000 / 2.641m | 0 / 0.0000 / 3.368m |
| 666 | 1.0000 / 1.0000 / 0.0000 | valid / valid | 3 / 4 / 3 | 1 | 11 / 10 | 77.0 / 67.0 | 1 / 1.0000 / 2.018m | 0 / 0.0000 / 3.609m |
| 1549 | 1.0000 / 1.0000 / 0.0000 | valid / valid | 3 / 4 / 3 | 2 | 9 / 7 | 58.0 / 43.0 | 1 / 1.0000 / 2.034m | 0 / 0.0000 / 4.353m |
| 603 | 1.0000 / 1.0000 / 0.0000 | valid / valid | 3 / 4 / 3 | 2 | 8 / 6 | 60.0 / 37.0 | 1 / 0.9775 / 0.798m | 0 / 0.0000 / 3.058m |
| 351 | 1.0000 / 1.0000 / 1.0000 | valid / valid | 3 / 4 / 3 | 0 | 13 / 13 | 119.0 / 323.0 | 1 / 0.7551 / 2.531m | 0 / 0.0000 / 5.005m |
| 132 | 1.0000 / 1.0000 / 1.0000 | valid / valid | 3 / 4 / 4 | 2 | 14 / 15 | 178.0 / 150.0 | 1 / 0.2944 / 1.686m | 0 / 0.0000 / 6.308m |
| 438 | 1.0000 / 1.0000 / 1.0000 | valid / valid | 3 / 4 / 3 | 1 | 15 / 15 | 253.0 / 324.0 | 1 / 0.2676 / 0.713m | 0 / 0.0000 / 7.829m |
| 314 | 1.0000 / 1.0000 / 0.0000 | valid / valid | 3 / 3 / 2 | 1 | 10 / 8 | 119.0 / 90.0 | 1 / 0.5127 / 1.162m | 0 / 0.0000 / 5.376m |
| 1432 | 1.0000 / 1.0000 / 1.0000 | valid / valid | 3 / 3 / 2 | 0 | 10 / 10 | 59.0 / 99.0 | 1 / 0.9271 / 0.380m | 0 / 0.0000 / 3.754m |
| 1609 | 1.0000 / 1.0000 / 1.0000 | valid / valid | 3 / 3 / 2 | 0 | 10 / 15 | 75.0 / 142.0 | 1 / 0.8298 / 0.438m | 0 / 0.0000 / 5.418m |
| 1085 | 1.0000 / 1.0000 / 1.0000 | valid / valid | 3 / 3 / 2 | 3 | 8 / 15 | 58.0 / 394.0 | 1 / 0.5703 / 0.691m | 0 / 0.0000 / 4.333m |
| 315 | 1.0000 / 1.0000 / 1.0000 | valid / valid | 3 / 3 / 2 | 0 | 13 / 7 | 250.0 / 81.0 | 1 / 0.2424 / 2.260m | 0 / 0.0000 / 3.361m |
| 116 | 1.0000 / 1.0000 / 1.0000 | valid / valid | 3 / 3 / 2 | 2 | 14 / 15 | 241.0 / 358.0 | 1 / 0.1974 / 0.229m | 0 / 0.0000 / 3.300m |

## Full instructions for layout review

### Episode 532

Walk away from the couch, through the doorway and up the staircase on the left. Continue past the first landing, up the second set of stairs and stop.

Lexical turn cues: []. Text-semantic review remains pending.

### Episode 456

Walk down the hall leading to the cabinet. At the cabinet take a right and enter the bedroom. In the room take a left and enter the bathroom on the far left. Stop on the rug in front of the sink.

Lexical turn cues: ['take a right', 'take a left']. Text-semantic review remains pending.

### Episode 1455

Walk into the hall and turn right. Walk through the entryway on the left and across the atrium. Walk up to the tenth step and stop.

Lexical turn cues: ['turn right']. Text-semantic review remains pending.

### Episode 691

Walk up two flights of stairs and continue onto the long rug in the hallway. Stop in the middle of the long rug and turn left. Enter the room and wait on the pink rug next to the lamp that is sitting on a cabinet.

Lexical turn cues: ['turn left']. Text-semantic review remains pending.

### Episode 1444

Head inside through the sliding glass doors on the other end of the deck. Stop in the living room and wait near the arm chair.

Lexical turn cues: []. Text-semantic review remains pending.

### Episode 326

Walk forward into the dining room and through the door at the far right. Walk between the two couches and past the fireplace on the left. Go through the double glass doors and wait near the couch.

Lexical turn cues: []. Text-semantic review remains pending.

### Episode 1043

Turn left then walk down the hallway.  Then turn right and enter the den.  Then walk through the bath room and enter the exercise room.

Lexical turn cues: ['Turn left', 'turn right']. Text-semantic review remains pending.

### Episode 52

Take a right at the large clock and travel straight through the kitchen, when you see the thermostat on the wall you are right next to the washroom.

Lexical turn cues: ['Take a right']. Text-semantic review remains pending.

### Episode 666

Exit the room and head right. When you get to frame of the teddy bear with balloons that says "Welcome" in the wall turn right. Head straight into that room and wait by the bed.

Lexical turn cues: ['turn right']. Text-semantic review remains pending.

### Episode 1549

Turn around and go passed the couch, then turn left and take an immediate right passed the painting. Turn left and go straight through the first door on the left.

Lexical turn cues: ['turn left', 'Turn left']. Text-semantic review remains pending.

### Episode 603

Walk straight ahead until you reach the bed. Turn left and walk out the door to the left. Once out, turn right and then enter the door on your right and stop once you enter.

Lexical turn cues: ['Turn left', 'turn right']. Text-semantic review remains pending.

### Episode 351

Walk down the stairs towards and past the atrium.  Head towards the kitchen and wait in the dining area.

Lexical turn cues: []. Text-semantic review remains pending.

### Episode 132

Going down the stairs take a right and go down the next set of stairs and at the bottom of the stairs take a left into the dining room stopping just inside the doorway.

Lexical turn cues: ['take a right', 'take a left']. Text-semantic review remains pending.

### Episode 438

Walk down the stairs and go straight down the hallway. Turn left into the room and wait near the dining table.

Lexical turn cues: ['Turn left']. Text-semantic review remains pending.

### Episode 314

Walk to the oven, turn left, walk past the couch, enter the room past the fireplace, wait at the grey couch.

Lexical turn cues: ['turn left']. Text-semantic review remains pending.

### Episode 1432

Walk past the couch and left through the room into the hallway. Walk past the microwave out of the room and stop.

Lexical turn cues: []. Text-semantic review remains pending.

### Episode 1609

Walk down both sets of stairs, through the doorway and wait by the rug.

Lexical turn cues: []. Text-semantic review remains pending.

### Episode 1085

Exit the bathroom. Turn left and then turn right. Turn left and go into the other bathroom. Wait near the sink.

Lexical turn cues: ['Turn left', 'turn right', 'Turn left']. Text-semantic review remains pending.

### Episode 315

Walk out of the kitchen, past the oven, past the sofas towards the fireplace.  Stop at the doorway facing towards the next sitting room facing the dining room table.

Lexical turn cues: []. Text-semantic review remains pending.

### Episode 116

you are in a closet, walk forward, turn right to walk out of closet into a bed room. turn right and see bath room, walk into bathroom and wait near the tub and sink.

Lexical turn cues: ['turn right', 'turn right']. Text-semantic review remains pending.

## Next action

Choose two episodes after reviewing the instructions and trajectories. Rerun only those episodes for panoramic/topological assets; this CPU screen does not render them.

# Paper qualitative-case shortlist

CPU screening only. No model inference, simulator or GPU is used.

## Definitions and provenance

- **split:** val_seen
- **paired_outcome:** ours success=1, zeromap success=0
- **excluded_episodes:** []
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
  "ours": "/home/xukai/code/ETP-R1-snapshot/ETP-R1/results/ours_val_seen_episodes.jsonl",
  "zeromap": "/home/xukai/code/ETP-R1-snapshot/ETP-R1/results/zeromap_val_seen_episodes.jsonl",
  "features": "/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/paper_cases/val_seen/features_val_seen.json",
  "validity": "/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/map_quality/t0_val_seen_episodes.jsonl",
  "presence": "/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/map_quality/t1_val_seen_episodes.jsonl",
  "feature_script": "/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/analysis/episode_features.py",
  "quality_script": "/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/analysis/map_quality/t1_presence.py"
}
```

## Filter counts

```json
{
  "paired_episodes": 778,
  "ours_success_zero_failure": 66,
  "after_explicit_exclusions": 66,
  "valid_json_and_schema": 65,
  "GT_turns_ge2_and_GT_regions_ge2": 29,
  "ours_high_steps_8_to15": 22
}
```

## Top candidates

| Episode | Mentioned F1 (pooled / object / region) | JSON / schema | GT turns / distinct regions / crossings | Lexical turn cues | High-level steps (ours / zero) | Low-level steps (ours / zero) | Ours SR / SPL / NE | Zero SR / SPL / NE |
|---|---|---|---|---|---|---|---|---|
| 485 | 1.0000 / 1.0000 / 1.0000 | valid / valid | 5 / 3 / 2 | 1 | 10 / 7 | 85.0 / 65.0 | 1 / 0.9389 / 0.928m | 0 / 0.0000 / 5.291m |
| 306 | 1.0000 / 1.0000 / 0.0000 | valid / valid | 5 / 2 / 1 | 0 | 9 / 12 | 75.0 / 129.0 | 1 / 0.5781 / 2.508m | 0 / 0.0000 / 5.303m |
| 307 | 1.0000 / 1.0000 / 0.0000 | valid / valid | 5 / 2 / 1 | 0 | 15 / 11 | 141.0 / 100.0 | 1 / 0.3017 / 1.799m | 0 / 0.0000 / 4.882m |
| 24 | 1.0000 / 1.0000 / 1.0000 | valid / valid | 4 / 3 / 2 | 2 | 15 / 15 | 195.0 / 232.0 | 1 / 0.5048 / 1.411m | 0 / 0.0000 / 9.547m |
| 170 | 1.0000 / 1.0000 / 1.0000 | valid / valid | 3 / 6 / 5 | 0 | 9 / 5 | 75.0 / 43.0 | 1 / 0.7669 / 1.793m | 0 / 0.0000 / 5.311m |
| 661 | 1.0000 / 0.0000 / 1.0000 | valid / valid | 3 / 5 / 4 | 3 | 10 / 7 | 66.0 / 44.0 | 1 / 0.8561 / 0.488m | 0 / 0.0000 / 3.024m |
| 131 | 1.0000 / 1.0000 / 1.0000 | valid / valid | 3 / 5 / 4 | 1 | 11 / 10 | 147.0 / 146.0 | 1 / 0.6429 / 1.761m | 0 / 0.0000 / 5.670m |
| 473 | 1.0000 / 1.0000 / 1.0000 | valid / valid | 3 / 5 / 6 | 1 | 13 / 11 | 88.0 / 100.0 | 1 / 0.9528 / 1.523m | 0 / 0.0000 / 12.477m |
| 101 | 1.0000 / 1.0000 / 1.0000 | valid / valid | 3 / 5 / 4 | 1 | 15 / 15 | 203.0 / 337.0 | 1 / 0.3875 / 1.977m | 0 / 0.0000 / 14.963m |
| 249 | 1.0000 / 1.0000 / 1.0000 | valid / valid | 3 / 4 / 3 | 4 | 9 / 9 | 72.0 / 69.0 | 1 / 0.9922 / 0.979m | 0 / 0.0000 / 3.799m |
| 400 | 1.0000 / 1.0000 / 1.0000 | valid / valid | 3 / 3 / 2 | 0 | 10 / 9 | 82.0 / 72.0 | 1 / 1.0000 / 2.510m | 0 / 0.0000 / 3.040m |
| 74 | 1.0000 / 1.0000 / 1.0000 | valid / valid | 3 / 2 / 1 | 1 | 9 / 9 | 58.0 / 87.0 | 1 / 1.0000 / 2.199m | 0 / 0.0000 / 5.226m |
| 75 | 1.0000 / 0.0000 / 1.0000 | valid / valid | 3 / 2 / 1 | 1 | 11 / 11 | 115.0 / 95.0 | 1 / 0.6331 / 2.292m | 0 / 0.0000 / 3.119m |
| 371 | 1.0000 / 1.0000 / 1.0000 | valid / valid | 2 / 5 / 5 | 1 | 13 / 9 | 71.0 / 47.0 | 1 / 0.9848 / 0.613m | 0 / 0.0000 / 4.890m |
| 121 | 1.0000 / 1.0000 / 1.0000 | valid / valid | 2 / 4 / 3 | 1 | 8 / 6 | 53.0 / 35.0 | 1 / 1.0000 / 1.787m | 0 / 0.0000 / 4.199m |
| 539 | 1.0000 / 1.0000 / 1.0000 | valid / valid | 2 / 4 / 4 | 0 | 8 / 6 | 56.0 / 50.0 | 1 / 1.0000 / 1.140m | 0 / 0.0000 / 3.179m |
| 450 | 1.0000 / 1.0000 / 1.0000 | valid / valid | 2 / 4 / 5 | 2 | 13 / 9 | 198.0 / 78.0 | 1 / 0.4219 / 1.152m | 0 / 0.0000 / 3.083m |
| 59 | 1.0000 / 1.0000 / 1.0000 | valid / valid | 2 / 3 / 2 | 1 | 9 / 7 | 75.0 / 57.0 | 1 / 0.9430 / 0.790m | 0 / 0.0000 / 4.228m |
| 746 | 1.0000 / 1.0000 / 0.0000 | valid / valid | 2 / 3 / 2 | 1 | 11 / 6 | 145.0 / 94.0 | 1 / 0.6060 / 0.355m | 0 / 0.0000 / 9.198m |
| 417 | 1.0000 / 1.0000 / 1.0000 | valid / valid | 2 / 2 / 1 | 0 | 8 / 6 | 48.0 / 33.0 | 1 / 1.0000 / 2.855m | 0 / 0.0000 / 4.608m |

## Full instructions for layout review

### Episode 485

Walk out of the room and turn left. Go past the wall sculpture and veer a little left. Keep going and wait in the doorway to the left of the mirror.

Lexical turn cues: ['turn left']. Text-semantic review remains pending.

### Episode 306

Walk around the right side of the table, through the pillars into the larger room.  Now go around the left side of the long table and stop in the middle of the room with the ping pong table to your right.

Lexical turn cues: []. Text-semantic review remains pending.

### Episode 307

Head around the table and go the main area left of the long table.  Go to the middle of the room next to the ping pong table and stop.

Lexical turn cues: []. Text-semantic review remains pending.

### Episode 24

Walk out of the bathroom and take a right. Then take a left. Take another right and wait at the top of the stairs.

Lexical turn cues: ['take a right', 'take a left']. Text-semantic review remains pending.

### Episode 170

Walk up the stairs.

Lexical turn cues: []. Text-semantic review remains pending.

### Episode 661

Walk into the office and turn left. Walk into the bedroom and turn left. Walk down the hall and turn left. Walk into the bathroom and stop.

Lexical turn cues: ['turn left', 'turn left', 'turn left']. Text-semantic review remains pending.

### Episode 131

Leave the closet and take a right into the hallway. In the hall walk straight and stick left passing a cabinet on your left. Once past the cabinet go into the first room on your left.

Lexical turn cues: ['take a right']. Text-semantic review remains pending.

### Episode 473

With the dining table on your right and ahead of you, exit the room through the opening to your left, then continue down the hallway to your right. At the end of the hallway, turn left and continue forward. Just before the end of the hallway with the pictures of a young girl and boy on the wall, take a dogleg to the left and enter the bedroom. Walk past the foot of the bed and enter the bathroom to the right of the head of the bed.

Lexical turn cues: ['turn left']. Text-semantic review remains pending.

### Episode 101

Walk past stove. Make right through open black door. Walk past outdoor dining room. Walk up stairs. Wait at black rocking chair.

Lexical turn cues: ['Make right']. Text-semantic review remains pending.

### Episode 249

Turn right and exit into the bedroom. Turn left and walk to the wall and then turn left. Walk to the end of that area and turn right out the last door. Once out the door walk across the small room and enter the door to the far left. Stop once you enter the bedroom.

Lexical turn cues: ['Turn right', 'Turn left', 'turn left', 'turn right']. Text-semantic review remains pending.

### Episode 400

Follow the stanchions and the red carpet straight through the doors into a narrow hall that bring you to another large room.  Step onto the cream and rose rug and wait.

Lexical turn cues: []. Text-semantic review remains pending.

### Episode 74

Walk to the hallway with a painting on the wall. Turn left keep walking keeping left going down the stairs waiting on the first few steps of the stairs.

Lexical turn cues: ['Turn left']. Text-semantic review remains pending.

### Episode 75

Walk out of bedroom into hallway and turn left. Walk down hall to stairway. Walk down stairway and wait on bottom landing.

Lexical turn cues: ['turn left']. Text-semantic review remains pending.

### Episode 371

Exit the door on the left then turn slightly left to go straight down the hallway. Go up the steps then turn left and go out the door on the left and stop.

Lexical turn cues: ['turn left']. Text-semantic review remains pending.

### Episode 121

Walk out of the kitchen into the dining room. Walk towards the doorway to the sitting room. Veer right to walk out of the sitting room to the foyer. Stop next to the blue chair.

Lexical turn cues: ['Veer right']. Text-semantic review remains pending.

### Episode 539

Go to your right and go past the kitchen and go to the left of the bar. Just past the bar, go into the bedroom on the left. GO a few steps into the room and wait there.

Lexical turn cues: []. Text-semantic review remains pending.

### Episode 450

enter the room then turn left walk straight then turn right enter the second door wait in the bathroom.

Lexical turn cues: ['turn left', 'turn right']. Text-semantic review remains pending.

### Episode 59

Walk through the door by the sink into the middle of the next room. Turn right and walk down the hallway and enter the third door on your right.

Lexical turn cues: ['Turn right']. Text-semantic review remains pending.

### Episode 746

Walk towards the small table in the center of the room. Enter the doorway on the right. Turn right when you get to the fountain and enter the door.

Lexical turn cues: ['Turn right']. Text-semantic review remains pending.

### Episode 417

Walk past the stairs and down the hall to the bedroom. Wait in the doorway, next to the bed.

Lexical turn cues: []. Text-semantic review remains pending.

## Next action

Choose two episodes after reviewing the instructions and trajectories. Rerun only those episodes for panoramic/topological assets; this CPU screen does not render them.

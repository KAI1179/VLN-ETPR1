# Map-faithful successful episodes: 9 pass / 1820 successful episodes scored (12 without a raster map)

Pass counts under relaxed thresholds: err<=30/cov>=0.8: 9, err<=30/cov>=0.7: 9, err<=45/cov>=0.8: 11, err<=45/cov>=0.7: 11

Filters: SPL>=0.85, K_gt>=3, |K_pred-K_gt|<=1, max dir err<=30.0 deg, cov>=0.8, prec>=0.6

| # | split | ep | scene | SPL | nDTW | path m | K pred/gt | dir err (deg) | turn sides | cov | prec | instruction |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | val_seen | 432 | sKLMLpTHeUy | 1.00 | 0.95 | 6.6 | 2/3 | [1, 4] | 1/1 | 1.00 | 0.92 | Go straight through the bathroom through the door on the right toward the bedroom. Stop at the foot of the bed. |
| 2 | val_unseen | 616 | X7HyMhZNoso | 1.00 | 0.95 | 9.3 | 2/3 | [3, 16] | 1/1 | 1.00 | 0.92 | Go up the steps next to the fridge. Then turn right toward the bathroom. Stop at the top of the steps. |
| 3 | val_seen | 431 | sKLMLpTHeUy | 1.00 | 0.95 | 6.6 | 3/3 | [1, 4, 27] | 2/2 | 1.00 | 0.92 | walk away from closet, past sinks, turn right into bedroom, stop in front of doorway on right. |
| 4 | val_seen | 302 | 1LXtFkjw3qL | 1.00 | 0.92 | 13.4 | 3/4 | [4, 4, 4] | 2/2 | 1.00 | 0.83 | Exit the bathroom toward the room and pass the black curtain, then pass the green couch and go out the white door on the right and stop at the top of the stairs. |
| 5 | val_unseen | 303 | zsNo4HB9uLZ | 0.95 | 0.94 | 10.4 | 3/4 | [18, 16, 12] | 2/2 | 1.00 | 0.86 | Walk down the hallway and take a left. At the end of the hallway take a left and a right at the next hall and enter the bedroom on your left. |
| 6 | val_unseen | 1695 | 8194nk5LbLH | 1.00 | 0.89 | 6.4 | 2/3 | [0, 1] | 1/1 | 1.00 | 0.63 | Continue up the stairs and wait inside the gym. |
| 7 | val_seen | 42 | 29hnd4uzFmX | 1.00 | 0.94 | 6.8 | 2/3 | [9, 2] | 1/1 | 1.00 | 0.68 | Walk forward through the kitchen and enter the empty room that is the third door from the left. Stop in the room next to the door. |
| 8 | val_seen | 315 | sKLMLpTHeUy | 1.00 | 0.94 | 9.9 | 2/3 | [28, 13] | 1/1 | 1.00 | 0.81 | Walk towards the dining table, walk through the entryway into the kitchen, turn right, and wait. |
| 9 | val_seen | 732 | aayBHfsNo7d | 0.93 | 0.85 | 14.7 | 2/3 | [11, 0] | 1/1 | 0.94 | 0.61 | Walk toward the staircase and to the left of the stairs. Turn left by the table and chairs and stop in front of the desk. |

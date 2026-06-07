# TODO

## Prior

- [x] In R2R, duplicated ids seem to only happen on `val_unseen` and `val_seen`. Need to check on RxR, and find if we can exploit it.

    ```python
    ids = dict()
    for episode in VLNCEEpisodeEntry.iter_from("R2R"):
        if episode.episode_id in ids:
            ids[episode.episode_id].add(episode.split)
        else:
            ids[episode.episode_id] = set()
    for id, splits in ids.items():
        if len(splits) > 1:
            print(f"Duplicated id: {id}, {splits}")
    ```

    Verdict: Cannot exploit. R2R has 1,836 duplicated episode ids and RxR has 11,006 duplicated episode ids across `train`, `val_seen`, and `val_unseen`. Treat `dataset + source + episode_id` as the canonical episode key.

- [x] Fix: cognitive grid map save, distinguish split (id in diff splits may collide). If prev todo handled, can be skipped.
- [x] Reuse magnum where possible, instead of impl algos ourselves.
- [x] Replace five direction vectors with trajectory keypoints.
- [x] Reorganize module layout.
- [x] SemanticBoxes final API
    - Remove `SceneSemanticBoxes.first_encountered_level`.
    - Remove `SceneSemanticBoxes.to_cognitive_map`.
    - Remove id from bounding boxes.
    - Add class `RelevantSemanticBoxes`.
        - Returned by `SceneSemanticBoxes.relevant_to`.
        - Basically `LevelSemanticBoxes`, plus `ground_truth_trajectory`,
          `trajectory_keypoints`, `instruction`, and `start_direction_vector`.
    - Overview
        - `SceneSemanticBoxes.from_scene_id` -> `SceneSemanticBoxes` (collection of `LevelSemanticBoxes`)
        - `SceneSemanticBoxes` -`relevant_to`-> `RelevantSemanticBoxes`
        - `RelevantSemanticBoxes` -`to_cognitive_map`-> `CognitiveGridMap`
    - Reason: By adding a separate type and utilizing type checking, we can reduce bugs.
- [x] Keep `GroundTruthGridMap` for inspection / visualization only. Remove direct production construction paths.
- [x] `VLNCEEpisodeEntry.role` seems stale. Consider removing it and relevant attrs / functions, then fix callers.
- [x] API section in readme

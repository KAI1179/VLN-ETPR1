# TODO

## Prior

- [ ] In R2R, duplicated ids seem to only happen on `val_unseen` and `val_seen`. Need to check on RxR, and find if we can exploit it.

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

- [ ] Fix: cognitive grid map save, distinguish split (id in diff splits may collide). If prev todo handled, can be skipped.
- [ ] Reuse magnum where possible, instead of impl algos ourselves.
- [ ] Remove direction vectors. Replace with reference paths.
- [ ] Reorganize module layout.
- [ ] SemanticBoxes API refactor (1)
    - Remove `SceneSemanticBoxes.first_encountered_level`.
    - Make `SceneSemanticBoxes.relevant_to` return `LevelSemanticBoxes` (uses same logic as first_encountered_level).
    - Move `to_cognitive_map` from `SceneSemanticBoxes` to `LevelSemanticBoxes`. Removed unused logic caused by our transition.
    - Remove id from bounding boxes.
- [ ] SemanticBoxes API refactor (2)
    - Add class `RelatedSemanticBoxes` (propose better name if any)
        - Returned by `SceneSemanticBoxes.relevant_to`
        - Basically just LevelSemanticBoxes, but has additional attr `reference_path`, `instruction` & `start_direction_vector`
    - Overview
        - `SceneSemanticBoxes.from_scene_id` -> `SceneSemanticBoxes` (collection of `LevelSemanticBoxes`)
        - `SceneSemanticBoxes` -`relevant_to`-> `RelatedSemanticBoxes`
        - `RelatedSemanticBoxes` -`to_cognitive_map`-> `CognitiveGridMap`
        - `SceneSemanticBoxes` -?-> `[GroundTruthGridMap]` (for inspection and visualization only)
        - Remove the path to generate `GroundTruthGridMap` / `[GroundTruthGridMap]` directly
    - Shortcuts can be kept, like `SceneSemanticBoxes.to_cognitive_map`
    - Reason: By adding a separate type and utilizing type checking, we can reduce bugs.

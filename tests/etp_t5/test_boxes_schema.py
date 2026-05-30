import json
import math

import pytest

import prior.bbox as bbox
from vlnce_baselines.models.etp_t5.boxes_schema import (
    ObjectBoxSpec,
    RegionBoxSpec,
    T5BoxesSpec,
    T5BoxesValidationError,
    build_t5_boxes_input,
    parse_t5_boxes_json,
    relevant_semantic_boxes_to_spec,
    spec_to_json,
    spec_to_relevant_semantic_boxes,
    write_prediction_artifact,
)


def _empty_level():
    return bbox.LevelSemanticBoxes(
        objects=[[] for _ in range(bbox.OBJECT_CATEGORIES)],
        regions=[[] for _ in range(bbox.REGION_CATEGORIES)],
        range_y=[None, None],
    )


def test_relevant_semantic_boxes_to_spec_sorts_and_rounds_without_runtime_fields():
    level = _empty_level()
    level.objects[3] = [
        bbox.OBB2D(
            center=(4.44, 1.24),
            half_extents=(0.24, 0.75),
            rotation=1.236,
            mentioned=True,
        )
    ]
    level.objects[1] = [
        bbox.OBB2D(
            center=(2.04, 3.06),
            half_extents=(0.54, 0.64),
            rotation=0.777,
            mentioned=False,
        ),
        bbox.OBB2D(
            center=(1.96, 8.01),
            half_extents=(1.0, 2.0),
            rotation=0.0,
            mentioned=True,
        ),
    ]
    level.regions[7] = [bbox.AABB2D(min=(0.04, 1.05), max=(2.06, 3.04))]
    level.regions[1] = [bbox.AABB2D(min=(5.0, 6.0), max=(7.0, 9.0))]
    relevant = bbox.RelevantSemanticBoxes(
        level_idx=2,
        level=level,
        instruction="go past the table",
        reference_path=[(0.0, 0.0)],
        start_direction_vector=(0.0, 1.0),
    )

    payload = json.loads(spec_to_json(relevant_semantic_boxes_to_spec(relevant)))

    assert payload == {
        "objects": [
            {
                "category": "chair",
                "center": [2.0, 3.1],
                "half_extents": [0.5, 0.6],
                "rotation": 0.78,
            },
            {
                "category": "chair",
                "center": [2.0, 8.0],
                "half_extents": [1.0, 2.0],
                "rotation": 0.0,
            },
            {
                "category": "table",
                "center": [4.4, 1.2],
                "half_extents": [0.2, 0.8],
                "rotation": 1.24,
            },
        ],
        "regions": [
            {
                "category": "bathroom/sanitary",
                "min": [0.0, 1.1],
                "max": [2.1, 3.0],
            },
            {
                "category": "living/social space",
                "min": [5.0, 6.0],
                "max": [7.0, 9.0],
            },
        ],
    }
    assert "mentioned" not in json.dumps(payload)
    assert "confidence" not in json.dumps(payload)


def test_build_t5_boxes_input_includes_metadata_but_not_scene_id():
    payload = json.loads(
        build_t5_boxes_input(
            "R2R",
            "Turn left at the chair.",
            start_position=(1.24, 2.96),
            start_direction=(0.123, -0.987),
        )
    )

    assert payload == {
        "dataset": "R2R",
        "start_position": [1.2, 3.0],
        "start_direction": [0.12, -0.99],
        "instruction": "Turn left at the chair.",
    }
    assert "scene" not in json.dumps(payload).lower()


def test_build_t5_boxes_input_accepts_3d_start_position_as_xz_projection():
    prompt_from_2d = json.loads(
        build_t5_boxes_input(
            "RxR",
            "Go ahead.",
            start_position=(1.24, 2.96),
            start_direction=(0.0, 1.0),
        )
    )
    prompt_from_3d = json.loads(
        build_t5_boxes_input(
            "RxR",
            "Go ahead.",
            start_position=(1.24, 99.0, 2.96),
            start_direction=(0.0, 1.0),
        )
    )

    assert prompt_from_2d["start_position"] == [1.2, 3.0]
    assert prompt_from_3d["start_position"] == [1.2, 3.0]


def test_parse_t5_boxes_json_accepts_strict_schema():
    spec = parse_t5_boxes_json(
        json.dumps(
            {
                "objects": [
                    {
                        "category": "chair",
                        "center": [1, 2],
                        "half_extents": [0.5, 0.75],
                        "rotation": 1.25,
                    }
                ],
                "regions": [
                    {
                        "category": "living/social space",
                        "min": [0, 0],
                        "max": [3, 4],
                    }
                ],
            }
        )
    )

    assert spec.objects == (
        ObjectBoxSpec(
            category="chair",
            center=(1.0, 2.0),
            half_extents=(0.5, 0.75),
            rotation=1.25,
        ),
    )
    assert spec.regions == (
        RegionBoxSpec(
            category="living/social space",
            min=(0.0, 0.0),
            max=(3.0, 4.0),
        ),
    )
    assert isinstance(spec.objects, tuple)
    assert isinstance(spec.regions, tuple)


@pytest.mark.parametrize(
    "payload",
    [
        "{not json",
        {"objects": [], "regions": [], "scene_id": "abc"},
        {"objects": [{"category": "alien", "center": [1, 2], "half_extents": [1, 1], "rotation": 0}], "regions": []},
        {"objects": [{"category": "chair", "center": [1], "half_extents": [1, 1], "rotation": 0}], "regions": []},
        {"objects": [{"category": "chair", "center": [1, 2], "half_extents": [0, 1], "rotation": 0}], "regions": []},
        {"objects": [{"category": "chair", "center": [1, 2], "half_extents": [1, 1]}], "regions": []},
        {"objects": [{"category": "chair", "center": [True, 2], "half_extents": [1, 1], "rotation": 0}], "regions": []},
        {"objects": [{"category": "chair", "center": [math.inf, 2], "half_extents": [1, 1], "rotation": 0}], "regions": []},
        {"objects": [{"category": "chair", "center": [1, 2], "half_extents": [1, 1], "rotation": False}], "regions": []},
        {"objects": [], "regions": [{"category": "unknown", "min": [0, 0], "max": [1, 1]}]},
        {"objects": [], "regions": [{"category": "circulation", "min": [0, 0], "max": [1, math.nan]}]},
        {"objects": [], "regions": [{"category": "circulation", "min": [0, 0], "max": [0, 1]}]},
    ],
)
def test_parse_t5_boxes_json_rejects_invalid_predictions(payload):
    text = payload if isinstance(payload, str) else json.dumps(payload)

    with pytest.raises(T5BoxesValidationError):
        parse_t5_boxes_json(text)


def test_spec_to_relevant_semantic_boxes_indexes_categories_and_derives_mentions():
    spec = T5BoxesSpec(
        objects=[
            ObjectBoxSpec(
                category="chair",
                center=(1.0, 2.0),
                half_extents=(0.5, 0.6),
                rotation=0.25,
            ),
            ObjectBoxSpec(
                category="table",
                center=(3.0, 4.0),
                half_extents=(1.0, 1.1),
                rotation=0.0,
            ),
        ],
        regions=[
            RegionBoxSpec(
                category="living/social space",
                min=(0.0, 0.0),
                max=(5.0, 6.0),
            )
        ],
    )

    relevant = spec_to_relevant_semantic_boxes(
        spec,
        instruction="Walk to the chair.",
        level_idx=4,
        reference_path=[(9.0, 8.0)],
        start_direction_vector=(1.0, 0.0),
        range_y=[0.0, 2.0],
        category_extractor=lambda instruction: ({1}, set()),
    )

    assert relevant.level_idx == 4
    assert relevant.instruction == "Walk to the chair."
    assert relevant.reference_path == [(9.0, 8.0)]
    assert relevant.start_direction_vector == (1.0, 0.0)
    assert relevant.level.range_y == [0.0, 2.0]
    assert relevant.level.objects[1] == [
        bbox.OBB2D(
            center=(1.0, 2.0),
            half_extents=(0.5, 0.6),
            rotation=0.25,
            mentioned=True,
        )
    ]
    assert relevant.level.objects[3][0].mentioned is False
    assert relevant.level.regions[1][0].mentioned is False


def test_spec_to_relevant_semantic_boxes_rejects_scalar_reference_points():
    spec = T5BoxesSpec(objects=[], regions=[])

    with pytest.raises(T5BoxesValidationError):
        spec_to_relevant_semantic_boxes(
            spec,
            instruction="Go ahead.",
            level_idx=0,
            reference_path=[1.0],
            start_direction_vector=(0.0, 1.0),
        )


def test_write_prediction_artifact_writes_valid_or_invalid_shape(tmp_path):
    spec = T5BoxesSpec(
        objects=[
            ObjectBoxSpec(
                category="chair",
                center=(1.0, 2.0),
                half_extents=(0.5, 0.5),
                rotation=0.0,
            )
        ],
        regions=[],
    )

    write_prediction_artifact(tmp_path, "valid-example", valid_spec=spec)
    write_prediction_artifact(
        tmp_path,
        "invalid-example",
        invalid_text="{bad",
        error=T5BoxesValidationError("malformed JSON"),
    )

    valid_payload = json.loads((tmp_path / "valid-example.json").read_text())
    invalid_payload = json.loads((tmp_path / "invalid-example.json").read_text())

    assert valid_payload == json.loads(spec_to_json(spec))
    assert invalid_payload == {"raw_text": "{bad", "error": "malformed JSON"}
    assert "raw_text" not in valid_payload
    assert "objects" not in invalid_payload


def test_write_prediction_artifact_rejects_ambiguous_arguments(tmp_path):
    spec = T5BoxesSpec(objects=[], regions=[])

    with pytest.raises(ValueError):
        write_prediction_artifact(
            tmp_path,
            "ambiguous",
            valid_spec=spec,
            invalid_text="{bad",
        )

    with pytest.raises(ValueError):
        write_prediction_artifact(tmp_path, "missing")


def test_write_prediction_artifact_sanitizes_example_id_path_components(tmp_path):
    write_prediction_artifact(
        tmp_path,
        "../nested/../../escape",
        invalid_text="{bad",
        error=T5BoxesValidationError("malformed JSON"),
    )

    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    assert files[0].parent == tmp_path
    assert files[0].name != "escape.json"
    assert not (tmp_path.parent / "escape.json").exists()

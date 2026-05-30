from typing import Any, cast

import pytest

import prior.bbox as bbox
from vlnce_baselines.models.etp_t5.boxes_schema import (
    ObjectBoxSpec,
    RegionBoxSpec,
    T5BoxesSpec,
    T5BoxesValidationError,
    build_t5_boxes_input,
    parse_t5_boxes_text,
    spec_to_t5_boxes_text,
    spec_to_relevant_semantic_boxes,
    write_prediction_artifact,
)


def _empty_level():
    return bbox.LevelSemanticBoxes(
        objects=[[] for _ in range(bbox.OBJECT_CATEGORIES)],
        regions=[[] for _ in range(bbox.REGION_CATEGORIES)],
        range_y=[None, None],
    )


def test_build_t5_boxes_input_includes_metadata_but_not_scene_id():
    prompt = build_t5_boxes_input(
        "R2R",
        "Turn left at the chair.",
        start_position=(1.24, 2.96),
        start_direction=(0.123, -0.987),
    )

    assert prompt == (
        "dataset R2R | start x = 1.2 | start z = 3.0 | "
        "direction x = 0.12 | direction z = -0.99 | "
        "instruction Turn left at the chair."
    )
    assert "scene" not in prompt.lower()
    assert "{" not in prompt
    assert "}" not in prompt


def test_build_t5_boxes_input_accepts_3d_start_position_as_xz_projection():
    prompt_from_2d = build_t5_boxes_input(
        "RxR",
        "Go ahead.",
        start_position=(1.24, 2.96),
        start_direction=(0.0, 1.0),
    )
    prompt_from_3d = build_t5_boxes_input(
        "RxR",
        "Go ahead.",
        start_position=(1.24, 99.0, 2.96),
        start_direction=(0.0, 1.0),
    )

    assert "start x = 1.2 | start z = 3.0" in prompt_from_2d
    assert "start x = 1.2 | start z = 3.0" in prompt_from_3d


def test_spec_to_t5_boxes_text_round_trips_objects_and_regions():
    spec = T5BoxesSpec(
        objects=[
            ObjectBoxSpec(
                category="table",
                center=(3.04, 4.06),
                half_extents=(0.54, 0.66),
                rotation=0.251,
            ),
            ObjectBoxSpec(
                category="chair",
                center=(1.0, 2.0),
                half_extents=(0.5, 0.5),
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

    text = spec_to_t5_boxes_text(spec)

    assert text == (
        "[ object chair | center x = 1.0 | center z = 2.0 | "
        "half x = 0.5 | half z = 0.5 | rotation = 0.0 ] "
        "[ object table | center x = 3.0 | center z = 4.1 | "
        "half x = 0.5 | half z = 0.7 | rotation = 0.25 ] "
        "[ region living/social space | min x = 0.0 | min z = 0.0 | "
        "max x = 5.0 | max z = 6.0 ]"
    )
    assert parse_t5_boxes_text(text) == T5BoxesSpec(
        objects=(
            ObjectBoxSpec("chair", (1.0, 2.0), (0.5, 0.5), 0.0),
            ObjectBoxSpec("table", (3.0, 4.1), (0.5, 0.7), 0.25),
        ),
        regions=(RegionBoxSpec("living/social space", (0.0, 0.0), (5.0, 6.0)),),
    )


def test_spec_to_t5_boxes_text_serializes_empty_spec_as_none():
    text = spec_to_t5_boxes_text(T5BoxesSpec(objects=[], regions=[]))

    assert text == "none"
    assert parse_t5_boxes_text(text) == T5BoxesSpec(objects=(), regions=())


@pytest.mark.parametrize(
    "text",
    [
        "not parseable",
        "[ object alien | center x = 1 | center z = 2 | half x = 1 | half z = 1 | rotation = 0 ]",
        "[ object chair | center x = 1 | center z = 2 | half x = 0 | half z = 1 | rotation = 0 ]",
        "[ object chair | center x = nope | center z = 2 | half x = 1 | half z = 1 | rotation = 0 ]",
        "[ object chair | center x = 1 | center z = 2 | half x = 1 | half z = 1 ]",
        "[ object chair | center x = 1 | center z = 2 | half x = 1 | half z = 1 | rotation = 0 | confidence = 1 ]",
        "[ region circulation | min x = 0 | min z = 0 | max x = 0 | max z = 1 ]",
        "[ region unknown | min x = 0 | min z = 0 | max x = 1 | max z = 1 ]",
        "[ region circulation | min x = 0 | min z = 0 | max x = 1 | max z = 1 | score = 1 ]",
    ],
)
def test_parse_t5_boxes_text_rejects_invalid_predictions(text):
    with pytest.raises(T5BoxesValidationError):
        parse_t5_boxes_text(text)


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
    invalid_reference_path = cast(Any, [1.0])

    with pytest.raises(T5BoxesValidationError):
        spec_to_relevant_semantic_boxes(
            spec,
            instruction="Go ahead.",
            level_idx=0,
            reference_path=invalid_reference_path,
            start_direction_vector=(0.0, 1.0),
        )


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


def test_write_prediction_artifact_writes_t5_text_files(tmp_path):
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
        invalid_text="[ object alien ]",
        error=T5BoxesValidationError("unknown object category: 'alien'"),
    )

    valid_text = (tmp_path / "valid-example.txt").read_text()
    invalid_text = (tmp_path / "invalid-example.txt").read_text()

    assert valid_text == (
        "[ object chair | center x = 1.0 | center z = 2.0 | "
        "half x = 0.5 | half z = 0.5 | rotation = 0.0 ]\n"
    )
    assert invalid_text == (
        "[ object alien ]\n"
        "\n"
        "# error: unknown object category: 'alien'\n"
    )
    assert not (tmp_path / "valid-example.json").exists()
    assert not (tmp_path / "invalid-example.json").exists()


def test_write_prediction_artifact_sanitizes_example_id_path_components(tmp_path):
    write_prediction_artifact(
        tmp_path,
        "../nested/../../escape",
        invalid_text="{bad",
        error=T5BoxesValidationError("unparsed text outside entities"),
    )

    files = list(tmp_path.glob("*.txt"))
    assert len(files) == 1
    assert files[0].parent == tmp_path
    assert files[0].name != "escape.txt"
    assert not (tmp_path.parent / "escape.txt").exists()

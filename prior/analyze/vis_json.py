"""Visualizes given JSON file. Supported formats:

- Exported JSON (`RelevantSemanticBoxes.save_json`)
- Flattened / imagined JSON (with fields `reference_path`, `objects` and `regions` only, fills in fabricated values for the rest)
"""

from pydantic import ValidationError
from ..bbox import RelevantSemanticBoxes
from tap import Tap
from json import load


class VisualizationArgs(Tap):
    json_path: str
    """Input path to the json file."""
    png_path: str
    """Output path to the generated png."""


def reconstruct(data: dict) -> RelevantSemanticBoxes:
    fixed = {
        "level_idx": data.get("level_idx") or 0,
        "level": {
            "objects": data["objects"],
            "regions": data["regions"],
            "range_y": data.get("range_y") or [None, None],
            "offset_x": data.get("offset_x") or 0,
            "offset_z": data.get("offset_z") or 0,
        },
        "instruction": "Move down the stairs and then turn left. Move forward and then immediately turn left into the living room. Continue forward and stop in front of the piano. ",
        "reference_path": data["reference_path"],
        "start_direction_vector": data.get("start_direction_vector") or [1, 0],
    }
    return RelevantSemanticBoxes.model_validate(fixed)


def main():
    args = VisualizationArgs(underscores_to_dashes=True).parse_args()
    with open(args.json_path) as f:
        data: dict = load(f)

    try:
        relevant_boxes = RelevantSemanticBoxes.model_validate(data)
    except ValidationError:
        relevant_boxes = reconstruct(data)

    grid_map = relevant_boxes.to_cognitive_map()
    grid_map.visualize(args.png_path)


if __name__ == "__main__":
    main()

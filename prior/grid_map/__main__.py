"""Visualizes given npz file."""

from typing import Literal
from ..grid_map import BaseGridMap, CognitiveGridMap
from tap import Tap


class VisualizationArgs(Tap):
    npz_path: str
    """Input path to the npz file."""
    png_path: str
    """Output path to the generated png."""
    type: Literal["base", "cognitive"] = "cognitive"
    """Map type."""


def main():
    args = VisualizationArgs(underscores_to_dashes=True).parse_args()
    if args.type == "base":
        grid_map = BaseGridMap.load(args.npz_path)
    elif args.type == "cognitive":
        grid_map = CognitiveGridMap.load(args.npz_path)
    grid_map.visualize(args.png_path)


if __name__ == "__main__":
    main()

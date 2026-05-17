"""Test the code."""

from . import (
    ANNOTATION_FILES,
    AnnotationEntry,
    ConnectivityEntry,
)


def main():
    annotation = AnnotationEntry.iter_from(ANNOTATION_FILES[3])
    entry = next(annotation)
    print(entry)
    print(entry.instruction)
    connectivity_map = ConnectivityEntry.map_for("E9uDoFAP3SH")
    print(connectivity_map["6ce4614650fd4294852d7fbeb89ef6be"])


if __name__ == "__main__":
    main()

from __future__ import annotations
import _magnum.scenegraph
import magnum as mn
import magnum as magnum
__all__: list[str] = ['GreedyFollowerError', 'InvalidAttachedObject', 'assert_obj_valid', 'magnum', 'mn']
class GreedyFollowerError(RuntimeError):
    pass
class InvalidAttachedObject(RuntimeError):
    pass
def assert_obj_valid(obj: _magnum.scenegraph.AbstractFeature3D) -> None:
    ...

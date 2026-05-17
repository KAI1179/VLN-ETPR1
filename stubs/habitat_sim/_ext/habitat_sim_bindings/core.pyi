from __future__ import annotations
import _magnum
import typing
__all__: list[str] = ['LoggingContext', 'orthonormalize_rotation_shear']
class LoggingContext:
    @staticmethod
    def current() -> LoggingContext:
        ...
    @staticmethod
    def reinitialize_from_env() -> None:
        ...
    @property
    def sim_is_quiet(self) -> bool:
        ...
@typing.overload
def orthonormalize_rotation_shear(arg0: _magnum.Matrix4) -> _magnum.Matrix4:
    ...
@typing.overload
def orthonormalize_rotation_shear(arg0: _magnum.Matrix4d) -> _magnum.Matrix4d:
    ...
_logging_context: LoggingContext  # value = <habitat_sim._ext.habitat_sim_bindings.core.LoggingContext object>

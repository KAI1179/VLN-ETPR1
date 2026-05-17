"""
EGL-based platform integration
"""
from __future__ import annotations
__all__: list[str] = ['WindowlessApplication']
class WindowlessApplication:
    """
    Windowless EGL application
    """
    class Configuration:
        """
        Configuration
        """
        def __init__(self) -> None:
            ...
    def __init__(self, configuration: WindowlessApplication.Configuration = ...) -> None:
        """
        Constructor
        """
    def exec(self) -> int:
        """
        Execute application
        """

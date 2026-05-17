"""
A simple Google-style logging wrapper.

Taken from https://github.com/benley/python-glog and adapted
"""
from __future__ import annotations
from habitat_sim._ext.habitat_sim_bindings.core import LoggingContext
import logging as logging
from logging import LogRecord
__all__: list[str] = ['DEBUG', 'ERROR', 'FATAL', 'HabitatSimFormatter', 'INFO', 'LogRecord', 'LoggingContext', 'WARN', 'WARNING', 'format_message', 'handler', 'logger', 'logging']
class HabitatSimFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ...
def format_message(record: logging.LogRecord) -> str:
    ...
DEBUG: int = 10
ERROR: int = 40
FATAL: int = 50
INFO: int = 20
WARN: int = 30
WARNING: int = 30
handler: logging.StreamHandler  # value = <StreamHandler <stderr> (NOTSET)>
logger: logging.Logger  # value = <Logger /home/vscode/.conda/envs/prior/lib/python3.9/site-packages/habitat_sim-0.3.3-py3.9-linux-x86_64.egg/habitat_sim/logging.py (ERROR)>

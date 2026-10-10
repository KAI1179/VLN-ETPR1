#!/usr/bin/env python3
"""Experiment entry point: existing cache generator restricted to RxR English.

One visible GPU per invocation; existing R2R generator defaults are unchanged.
"""
import sys
from vlnce_baselines.models.etp_llm import llm_grid_navigation_cache as cache


if __name__ == '__main__':
    cache.GRID_VLNCE_DATASETS = ('RxR',)
    cache.main(sys.argv[1:] + ['--scope', 'navigation-full', '--parallel-workers', '1'])

"""
GRPO (Group Relative Policy Optimization) Implementation

This module implements the core GRPO pipeline for training language models
with mechanistic signal integration for hallucination mitigation.
"""

from .pipeline import GRPOPipeline
from .optimizer import GRPOOptimizer
from .group_selector import GroupSelector

__all__ = ["GRPOPipeline", "GRPOOptimizer", "GroupSelector"]

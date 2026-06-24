"""Nexus Vision Pipeline Package

This package provides a framework for building and running machine learning pipelines.
It includes classes for defining stages, submitting jobs, and managing job statuses.

Version: 0.1.0
"""
__version__ = "0.1.0"

from .pipeline import JobStatus, Job, Stage, DeadLetterQueue, Pipeline

# Additional imports and code...

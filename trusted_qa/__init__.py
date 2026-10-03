"""Draft independent QA runner for the Mool Katha trusted repository.

This package is deliberately self contained: a future trusted job must never
import Python modules from the producer-writable checkout it is inspecting.
"""

from .common import QaHold

__all__ = ["QaHold"]

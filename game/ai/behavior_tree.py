"""Tiny behaviour-tree toolkit (plan phase 7.3).

Nodes tick with a shared context object and return SUCCESS / FAILURE /
RUNNING.  Composites (Selector, Sequence) plus leaf Condition/Action cover
every mob behaviour; trees are assembled in game/mobs.py.
"""

from __future__ import annotations

from enum import Enum
from typing import Callable


class Status(Enum):
    SUCCESS = 1
    FAILURE = 2
    RUNNING = 3


class Node:
    def tick(self, ctx) -> Status:  # noqa: D401
        raise NotImplementedError


class Selector(Node):
    """First non-failing child wins."""

    def __init__(self, *children: Node) -> None:
        self.children = children

    def tick(self, ctx) -> Status:
        for child in self.children:
            status = child.tick(ctx)
            if status != Status.FAILURE:
                return status
        return Status.FAILURE


class Sequence(Node):
    """Fails on the first failing child; RUNNING pauses the sequence."""

    def __init__(self, *children: Node) -> None:
        self.children = children

    def tick(self, ctx) -> Status:
        for child in self.children:
            status = child.tick(ctx)
            if status != Status.SUCCESS:
                return status
        return Status.SUCCESS


class Condition(Node):
    def __init__(self, predicate: Callable[[object], bool]) -> None:
        self.predicate = predicate

    def tick(self, ctx) -> Status:
        return Status.SUCCESS if self.predicate(ctx) else Status.FAILURE


class Action(Node):
    def __init__(self, fn: Callable[[object], Status]) -> None:
        self.fn = fn

    def tick(self, ctx) -> Status:
        return self.fn(ctx)

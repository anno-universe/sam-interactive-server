"""Typed prompt data objects for *interactive* inference.

An interactive model (SAM / SAM2 / MedSAM …) is driven by a sequence of prompts:
a box, positive or negative click points, a mask, or free text. On the wire each
prompt is a plain ``dict`` tagged with a ``"type"`` key plus type-specific fields
— the shape the Anno backend persists in ``InteractiveInferencePrompt`` and the
frontend replays on commit. These classes give the *inference-server author* a
typed view of that same wire shape so ``predict()`` can iterate over concrete
objects instead of raw dicts.

Serialization mirrors the geometry DOs in :mod:`~anno_sdk.types`: every class
exposes a ``prompt_type`` string and a :meth:`to_dict` that emits the tagged
dict; :func:`parse_prompt` / :func:`parse_prompts` invert it.

Wire shapes
-----------
* ``box``            → ``{"type": "box", "x", "y", "width", "height"}``
* ``positive_point`` → ``{"type": "positive_point", "x", "y"}``
* ``negative_point`` → ``{"type": "negative_point", "x", "y"}``
* ``mask``           → ``{"type": "mask", "points": [[x, y], ...]}`` (may also carry ``"rle"``)
* ``text``           → ``{"type": "text", "text": ...}``
"""

from __future__ import annotations

from dataclasses import dataclass

#: Prompt type constants (canonical home; re-exported by :mod:`anno_sdk.interactive`).
PROMPT_BOX = "box"
PROMPT_POSITIVE_POINT = "positive_point"
PROMPT_NEGATIVE_POINT = "negative_point"
PROMPT_MASK = "mask"
PROMPT_TEXT = "text"

#: The full set of prompt types this contract defines.
PROMPT_TYPES = frozenset(
    {PROMPT_BOX, PROMPT_POSITIVE_POINT, PROMPT_NEGATIVE_POINT, PROMPT_MASK, PROMPT_TEXT}
)


@dataclass
class BoxPrompt:
    """A bounding-box prompt."""

    x: float
    y: float
    width: float
    height: float

    TYPE = PROMPT_BOX

    @property
    def prompt_type(self) -> str:
        return self.TYPE

    def to_dict(self) -> dict:
        return {
            "type": self.TYPE,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, data: dict) -> BoxPrompt:
        return cls(
            x=data["x"],
            y=data["y"],
            width=data["width"],
            height=data["height"],
        )


@dataclass
class PositivePointPrompt:
    """A positive (include) click point."""

    x: float
    y: float

    TYPE = PROMPT_POSITIVE_POINT

    @property
    def prompt_type(self) -> str:
        return self.TYPE

    def to_dict(self) -> dict:
        return {"type": self.TYPE, "x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, data: dict) -> PositivePointPrompt:
        return cls(x=data["x"], y=data["y"])


@dataclass
class NegativePointPrompt:
    """A negative (exclude) click point."""

    x: float
    y: float

    TYPE = PROMPT_NEGATIVE_POINT

    @property
    def prompt_type(self) -> str:
        return self.TYPE

    def to_dict(self) -> dict:
        return {"type": self.TYPE, "x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, data: dict) -> NegativePointPrompt:
        return cls(x=data["x"], y=data["y"])


@dataclass
class MaskPrompt:
    """A mask prompt, carried as polygon ``points`` and/or an ``rle`` payload.

    At least one of ``points`` / ``rle`` is expected; both are optional so the
    service can pick whichever encoding it supports. ``None`` fields are dropped
    from :meth:`to_dict`.
    """

    points: list[list[float]] | None = None
    rle: dict | str | None = None

    TYPE = PROMPT_MASK

    @property
    def prompt_type(self) -> str:
        return self.TYPE

    def to_dict(self) -> dict:
        out: dict = {"type": self.TYPE}
        if self.points is not None:
            out["points"] = self.points
        if self.rle is not None:
            out["rle"] = self.rle
        return out

    @classmethod
    def from_dict(cls, data: dict) -> MaskPrompt:
        return cls(points=data.get("points"), rle=data.get("rle"))


@dataclass
class TextPrompt:
    """A free-text prompt."""

    text: str

    TYPE = PROMPT_TEXT

    @property
    def prompt_type(self) -> str:
        return self.TYPE

    def to_dict(self) -> dict:
        return {"type": self.TYPE, "text": self.text}

    @classmethod
    def from_dict(cls, data: dict) -> TextPrompt:
        return cls(text=data["text"])


#: Union of every concrete prompt type.
Prompt = BoxPrompt | PositivePointPrompt | NegativePointPrompt | MaskPrompt | TextPrompt

#: Dispatch table from wire ``"type"`` string to the concrete class.
_PROMPT_REGISTRY: dict[str, type] = {
    cls.TYPE: cls
    for cls in (BoxPrompt, PositivePointPrompt, NegativePointPrompt, MaskPrompt, TextPrompt)
}


def parse_prompt(data: dict) -> Prompt:
    """Turn a single wire ``dict`` into its typed :data:`Prompt`.

    Raises :class:`ValueError` if the ``"type"`` key is missing or unknown.
    """
    try:
        prompt_type = data["type"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"prompt is missing a 'type' key: {data!r}") from exc
    cls = _PROMPT_REGISTRY.get(prompt_type)
    if cls is None:
        raise ValueError(f"unknown prompt type {prompt_type!r}")
    return cls.from_dict(data)


def parse_prompts(items: list[dict]) -> list[Prompt]:
    """Turn a list of wire dicts into typed :data:`Prompt` objects."""
    return [parse_prompt(item) for item in items]

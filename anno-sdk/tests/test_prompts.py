"""Tests for the typed interactive prompt DOs (prompts.py)."""

from __future__ import annotations

import pytest

from anno_sdk import (
    BoxPrompt,
    MaskPrompt,
    NegativePointPrompt,
    PositivePointPrompt,
    TextPrompt,
    parse_prompt,
    parse_prompts,
)


class TestPromptRoundtrip:
    def test_box(self) -> None:
        p = BoxPrompt(x=10, y=20, width=100, height=50)
        assert p.prompt_type == "box"
        assert p.to_dict() == {"type": "box", "x": 10, "y": 20, "width": 100, "height": 50}
        assert BoxPrompt.from_dict(p.to_dict()) == p
        assert parse_prompt(p.to_dict()) == p

    def test_positive_point(self) -> None:
        p = PositivePointPrompt(x=1, y=2)
        assert p.to_dict() == {"type": "positive_point", "x": 1, "y": 2}
        assert parse_prompt(p.to_dict()) == p

    def test_negative_point(self) -> None:
        p = NegativePointPrompt(x=3, y=4)
        assert p.to_dict() == {"type": "negative_point", "x": 3, "y": 4}
        assert parse_prompt(p.to_dict()) == p

    def test_text(self) -> None:
        p = TextPrompt(text="the cat")
        assert p.to_dict() == {"type": "text", "text": "the cat"}
        assert parse_prompt(p.to_dict()) == p

    def test_mask_with_points(self) -> None:
        p = MaskPrompt(points=[[0, 0], [1, 1], [2, 0]])
        assert p.to_dict() == {"type": "mask", "points": [[0, 0], [1, 1], [2, 0]]}
        assert "rle" not in p.to_dict()  # None optionals dropped
        assert parse_prompt(p.to_dict()) == p

    def test_mask_with_rle(self) -> None:
        p = MaskPrompt(rle={"counts": "abc", "size": [10, 10]})
        d = p.to_dict()
        assert d == {"type": "mask", "rle": {"counts": "abc", "size": [10, 10]}}
        assert "points" not in d
        assert parse_prompt(d) == p


class TestParse:
    def test_parse_prompts_mixed(self) -> None:
        wire = [
            {"type": "box", "x": 1, "y": 2, "width": 3, "height": 4},
            {"type": "positive_point", "x": 5, "y": 6},
            {"type": "text", "text": "hi"},
        ]
        prompts = parse_prompts(wire)
        assert [type(p).__name__ for p in prompts] == [
            "BoxPrompt",
            "PositivePointPrompt",
            "TextPrompt",
        ]
        assert [p.to_dict() for p in prompts] == wire

    def test_parse_prompt_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown prompt type"):
            parse_prompt({"type": "lasso", "points": []})

    def test_parse_prompt_missing_type_raises(self) -> None:
        with pytest.raises(ValueError, match="missing a 'type'"):
            parse_prompt({"x": 1, "y": 2})

    def test_parse_prompts_empty(self) -> None:
        assert parse_prompts([]) == []

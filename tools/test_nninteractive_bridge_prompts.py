"""Prompt-order contract tests for the external nnInteractive bridge."""

from __future__ import annotations

import nninteractive_bridge as bridge


IDENTITY_MAPPING = {
    "platform_to_mimics_axes": [0, 1, 2],
    "platform_to_mimics_flips": [False, False, False],
}


class RecordingSession:
    def __init__(self) -> None:
        self.calls = []

    def add_point_interaction(
        self, point, include_interaction, run_prediction=True
    ):
        self.calls.append(
            ("point", tuple(point), bool(include_interaction), run_prediction)
        )

    def add_scribble_interaction(
        self,
        crop,
        include_interaction,
        run_prediction=True,
        interaction_bbox=None,
    ):
        self.calls.append(
            (
                "scribble",
                bool(include_interaction),
                run_prediction,
                interaction_bbox,
            )
        )


def test_point_set_runs_every_prompt_in_annotator_order():
    session = RecordingSession()
    count = bridge._apply_point_set(
        session,
        {
            "interaction_type": "point_set",
            "coordinates": "mimics",
            "points": [
                {"point": [2, 3, 4], "include_interaction": True},
                {"point": [8, 9, 10], "include_interaction": False},
                {"point": [12, 13, 14], "include_interaction": True},
            ],
        },
        mimics_shape=[20, 20, 20],
        platform_shape=[20, 20, 20],
        buffer_mapping=IDENTITY_MAPPING,
    )
    assert count == 3
    assert session.calls == [
        ("point", (2, 3, 4), True, True),
        ("point", (8, 9, 10), False, True),
        ("point", (12, 13, 14), True, True),
    ]


def test_scribble_set_predicts_after_each_scribble():
    session = RecordingSession()
    count = bridge._apply_scribble_set(
        session,
        {
            "interaction_type": "scribble_set",
            "coordinates": "mimics",
            "scribbles": [
                {
                    "polyline_points": [[4, 4, 7], [8, 8, 7]],
                    "include_interaction": True,
                },
                {
                    "polyline_points": [[12, 12, 7], [15, 15, 7]],
                    "include_interaction": False,
                },
            ],
        },
        mimics_shape=[20, 20, 20],
        platform_shape=[20, 20, 20],
        buffer_mapping=IDENTITY_MAPPING,
    )
    assert count == 2
    assert [call[1:3] for call in session.calls] == [
        (True, True),
        (False, True),
    ]

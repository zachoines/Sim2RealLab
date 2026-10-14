"""Two-axis translate plans from a live planner service.

Opt in by setting ``STRAFER_LIVE_PLANNER_URL`` to a running service, for
example ``http://127.0.0.1:8200`` after ``make serve-planner``. Without it
the module is skipped; with it, an unhealthy service fails the module.
"""

from __future__ import annotations

import os
import uuid

import pytest

from strafer_autonomy.clients.planner_client import HttpPlannerClient, HttpPlannerClientConfig
from strafer_autonomy.executor.mission_runner import DEFAULT_AVAILABLE_SKILLS
from strafer_autonomy.schemas import PlannerRequest

_URL = os.environ.get("STRAFER_LIVE_PLANNER_URL", "")

pytestmark = [
    pytest.mark.live_planner,
    pytest.mark.skipif(not _URL, reason="STRAFER_LIVE_PLANNER_URL is not set"),
]


@pytest.fixture(scope="module")
def client() -> HttpPlannerClient:
    client = HttpPlannerClient(HttpPlannerClientConfig(base_url=_URL, max_retries=0))
    health = client.health()
    assert health.get("model_loaded") is True, health
    return client


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("move 1.712 meters forward and 1.372 meters right", (1.712, -1.372)),
        ("move 1.712 meters forward and 1.372 meters left", (1.712, 1.372)),
        ("move 1.712 meters backward and 1.372 meters right", (-1.712, -1.372)),
        pytest.param(
            "move 1.712 meters backward and 1.372 meters left",
            (-1.712, 1.372),
            marks=pytest.mark.xfail(
                strict=True,
                reason="planner-translate-direction-schema: the planner returns this lateral sign wrong",
            ),
        ),
    ],
)
def test_two_axis_translate_sign_and_magnitude(client, command, expected):
    request = PlannerRequest(
        request_id=uuid.uuid4().hex,
        raw_command=command,
        available_skills=DEFAULT_AVAILABLE_SKILLS,
    )
    plan = client.plan_mission(request)
    assert plan.mission_type == "translate"
    assert [step.skill for step in plan.steps] == ["translate"]
    args = plan.steps[0].args
    assert (args["dx_m"], args["dy_m"]) == pytest.approx(expected, abs=1e-3)

from __future__ import annotations

import pooltool.constants as c
from pooltool.objects.ball.datatypes import BallState
from pooltool.system.datatypes import System


class ShotBallFilter:
    def __init__(self):
        self.enabled = False
        self.target_ball_id: str | None = None
        self.hidden_ball_ids: set[str] = set()
        self._saved_states: dict[str, BallState] = {}

    @property
    def applied(self) -> bool:
        return bool(self._saved_states)

    def configure(self, enabled: bool, target_ball_id: str | None) -> None:
        self.enabled = enabled
        self.target_ball_id = target_ball_id if enabled else None

    def visible_ball_ids(self, system: System) -> set[str]:
        if not self.enabled or self.target_ball_id is None:
            return set(system.balls)

        return {
            ball_id
            for ball_id in (system.cue.cue_ball_id, self.target_ball_id)
            if ball_id in system.balls
        }

    def hidden_ids(self, system: System) -> set[str]:
        if not self.enabled or self.target_ball_id is None:
            return set()

        visible = self.visible_ball_ids(system)
        return {
            ball_id
            for ball_id, ball in system.balls.items()
            if ball_id not in visible and ball.state.s != c.pocketed
        }

    def apply_to_system(self, system: System) -> None:
        self.restore_system(system)
        self.hidden_ball_ids = self.hidden_ids(system)
        self._saved_states = {
            ball_id: system.balls[ball_id].state.copy()
            for ball_id in self.hidden_ball_ids
        }

        for ball_id in self.hidden_ball_ids:
            state = system.balls[ball_id].state.copy()
            state.s = c.pocketed
            state.rvw[1] = 0.0
            state.rvw[2] = 0.0
            system.balls[ball_id].state = state

    def restore_system(self, system: System) -> set[str]:
        restored = set()
        for ball_id, state in self._saved_states.items():
            if ball_id in system.balls:
                system.balls[ball_id].state = state.copy()
                restored.add(ball_id)

        self._saved_states = {}
        if not self.enabled:
            self.hidden_ball_ids = set()
        return restored


shot_filter = ShotBallFilter()

import numpy as np

import pooltool.constants as c
from pooltool.ani.shot_filter import ShotBallFilter
from pooltool.objects.ball.datatypes import Ball
from pooltool.system.datatypes import System


def test_shot_filter_visible_ball_ids():
    system = System.example()
    ball_ids = list(system.balls)
    target_ball_id = next(
        ball_id for ball_id in ball_ids if ball_id != system.cue.cue_ball_id
    )

    shot_filter = ShotBallFilter()
    assert shot_filter.visible_ball_ids(system) == set(system.balls)

    shot_filter.configure(enabled=True, target_ball_id=target_ball_id)

    assert shot_filter.visible_ball_ids(system) == {
        system.cue.cue_ball_id,
        target_ball_id,
    }


def test_shot_filter_apply_and_restore_system():
    system = System.example()
    system.balls["2"] = Ball.create("2", xy=(system.table.w / 2, system.table.l / 4))
    ball_ids = list(system.balls)
    target_ball_id = next(
        ball_id for ball_id in ball_ids if ball_id != system.cue.cue_ball_id
    )
    hidden_ball_id = next(
        ball_id
        for ball_id in ball_ids
        if ball_id not in {system.cue.cue_ball_id, target_ball_id}
    )
    original_state = system.balls[hidden_ball_id].state.copy()

    shot_filter = ShotBallFilter()
    shot_filter.configure(enabled=True, target_ball_id=target_ball_id)
    shot_filter.apply_to_system(system)

    assert shot_filter.applied
    assert hidden_ball_id in shot_filter.hidden_ball_ids
    assert system.balls[hidden_ball_id].state.s == c.pocketed
    assert np.all(system.balls[hidden_ball_id].state.rvw[1:] == 0)

    restored = shot_filter.restore_system(system)

    assert restored == shot_filter.hidden_ball_ids
    assert not shot_filter.applied
    assert system.balls[hidden_ball_id].state == original_state
    assert system.balls[hidden_ball_id].state is not original_state

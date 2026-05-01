#! /usr/bin/env python

import numpy as np
from panda3d.core import LineSegs, Point2, Point3

import pooltool.ani.tasks as tasks
import pooltool.constants as c
from pooltool.ani.action import Action
from pooltool.ani.camera import cam
from pooltool.ani.collision import cue_avoid
from pooltool.ani.constants import (
    elevate_sensitivity,
    english_sensitivity,
    max_elevate,
    max_english,
    max_stroke_speed,
    min_camera,
    min_stroke_speed,
    power_sensitivity,
    rotate_fine_sensitivity_x,
    rotate_sensitivity_x,
)
from pooltool.ani.globals import Global
from pooltool.ani.hud import hud
from pooltool.ani.modes.datatypes import BaseMode, Mode
from pooltool.ani.mouse import MouseMode, mouse
from pooltool.ani.scene import visual
from pooltool.ani.shot_filter import shot_filter
from pooltool.config import settings
from pooltool.ptmath.utils import norm2d, tip_contact_offset
from pooltool.system.datatypes import multisystem

CUE_DRAG_HOTZONE_LENGTH = 0.9
CUE_DRAG_HOTZONE_RADIUS = 0.12
MOUSE_WHEEL_ZOOM = 0.08
TARGET_BALL_HOTZONE_PAD = 0.08
TARGET_MARKER_RADIUS = 0.035
AIM_DASH_LENGTH = 0.075
AIM_DASH_GAP = 0.045
AIM_GUIDE_Z_OFFSET = 0.006


class AimGuide:
    def __init__(self, mode):
        self.mode = mode
        self.target_ball_id: str | None = None
        self.target_point: np.ndarray | None = None
        self.node = None

    def destroy(self):
        self.clear()
        hud.update_target_ball(None)

    def clear(self):
        if self.node is not None and not self.node.isEmpty():
            self.node.removeNode()
        self.node = None

    def validate_target(self):
        if self.target_ball_id not in self.candidate_ball_ids():
            self.target_ball_id = None
            self.target_point = None

        hud.update_target_ball(self.target_ball_id)

    def has_target(self):
        self.validate_target()
        return self.target_ball_id is not None and self.target_point is not None

    def prompt_for_target(self):
        self.target_ball_id = None
        self.target_point = None
        hud.update_target_ball(None)
        self.clear()

    def candidate_ball_ids(self):
        cue_ball_id = multisystem.active.cue.cue_ball_id

        return [
            ball_id
            for ball_id in visual.balls
            if ball_id in visual.balls
            and ball_id != cue_ball_id
            and visual.balls[ball_id]._ball.state.s != c.pocketed
            and not visual.balls[ball_id].get_node("pos").isHidden()
        ]

    def default_target_point(self, ball_id):
        if ball_id is None:
            return None

        cue_pos = self.cue_ball_position()
        ball_pos = self.ball_position(ball_id)
        direction = ball_pos[:2] - cue_pos[:2]
        length = np.linalg.norm(direction)
        if length < 1e-9:
            point_xy = ball_pos[:2]
        else:
            radius = visual.balls[ball_id]._ball.params.R
            point_xy = ball_pos[:2] - direction / length * radius

        return np.array([point_xy[0], point_xy[1], ball_pos[2]], dtype=float)

    def cue_ball_position(self):
        cue_ball_id = multisystem.active.cue.cue_ball_id
        if cue_ball_id not in visual.balls:
            return None
        return self.ball_position(cue_ball_id)

    def ball_position(self, ball_id):
        return np.array(visual.balls[ball_id].get_node("pos").getPos(Global.render))

    def select_from_cursor(self):
        ball_id = self.ball_under_cursor()
        if ball_id is None:
            return False

        target_changed = ball_id != self.target_ball_id
        self.target_ball_id = ball_id
        if target_changed or self.target_point is None:
            self.target_point = self.default_target_point(ball_id)
        self.update_target_point_from_cursor()
        hud.update_target_ball(self.target_ball_id)
        return True

    def ball_under_cursor(self):
        if not mouse.mouse.hasMouse():
            return None

        cursor = np.array([mouse.get_x(), mouse.get_y()])
        best_ball_id = None
        best_distance = np.inf

        for ball_id in self.candidate_ball_ids():
            ball = visual.balls[ball_id]
            center = self.mode.project_to_screen(ball.get_node("pos").getPos(Global.render))
            if center is None:
                continue

            radius_world = ball._ball.params.R
            edge_world = ball.get_node("pos").getPos(Global.render) + Point3(
                radius_world, 0, 0
            )
            edge = self.mode.project_to_screen(edge_world)
            radius_screen = (
                np.linalg.norm(edge - center) if edge is not None else TARGET_BALL_HOTZONE_PAD
            )
            hotzone = max(TARGET_BALL_HOTZONE_PAD, radius_screen * 1.8)
            distance = np.linalg.norm(cursor - center)

            if distance <= hotzone and distance < best_distance:
                best_ball_id = ball_id
                best_distance = distance

        return best_ball_id

    def update_target_point_from_cursor(self):
        if self.target_ball_id is None:
            return

        target_ball = visual.balls[self.target_ball_id]
        ball_pos = self.ball_position(self.target_ball_id)
        cursor_point = self.mode.mouse_table_point(z=ball_pos[2])
        if cursor_point is None:
            return

        offset = np.array([cursor_point.x - ball_pos[0], cursor_point.y - ball_pos[1]])
        radius = target_ball._ball.params.R
        distance = np.linalg.norm(offset)
        if distance > radius:
            offset = offset / distance * radius

        self.target_point = np.array(
            [ball_pos[0] + offset[0], ball_pos[1] + offset[1], ball_pos[2]],
            dtype=float,
        )

    def point_cue_at_target(self):
        cue_pos = self.cue_ball_position()
        if cue_pos is None or self.target_point is None:
            return

        direction = cue_pos[:2] - self.target_point[:2]
        if np.linalg.norm(direction) < 1e-9:
            return

        phi = np.degrees(np.arctan2(direction[1], direction[0])) % 360
        self.mode.set_cue_phi(phi)

    def update(self, target_drag=False):
        self.validate_target()
        if self.target_ball_id is None or self.target_point is None:
            self.clear()
            return

        if target_drag:
            self.update_target_point_from_cursor()
            self.point_cue_at_target()

        self.draw()

    def draw(self):
        cue_pos = self.cue_ball_position()
        if cue_pos is None or self.target_point is None:
            self.clear()
            return

        target = np.array(self.target_point, dtype=float)
        endpoint = self.table_edge_point(cue_pos, target)
        if endpoint is None:
            self.clear()
            return

        self.clear()
        drawer = LineSegs()
        drawer.setThickness(3)
        drawer.setColor(1.0, 0.85, 0.14, 1)

        start = np.array([cue_pos[0], cue_pos[1], cue_pos[2] + AIM_GUIDE_Z_OFFSET])
        end = np.array([endpoint[0], endpoint[1], cue_pos[2] + AIM_GUIDE_Z_OFFSET])
        self.draw_dashed_line(drawer, start, end)

        drawer.setColor(0.05, 0.7, 1.0, 1)
        marker = np.array(
            [
                target[0],
                target[1],
                target[2] + visual.balls[self.target_ball_id]._ball.params.R + 0.01,
            ]
        )
        self.draw_marker(drawer, marker)

        self.node = Global.render.find("scene").find("table").attachNewNode(
            drawer.create()
        )
        self.node.setDepthTest(False)
        self.node.setDepthWrite(False)

    def draw_dashed_line(self, drawer, start, end):
        vector = end - start
        length = np.linalg.norm(vector)
        if length < 1e-9:
            return

        direction = vector / length
        distance = 0.0
        while distance < length:
            segment_start = start + direction * distance
            segment_end = start + direction * min(distance + AIM_DASH_LENGTH, length)
            drawer.moveTo(*segment_start)
            drawer.drawTo(*segment_end)
            distance += AIM_DASH_LENGTH + AIM_DASH_GAP

    def draw_marker(self, drawer, center):
        r = TARGET_MARKER_RADIUS
        drawer.moveTo(center[0] - r, center[1], center[2])
        drawer.drawTo(center[0] + r, center[1], center[2])
        drawer.moveTo(center[0], center[1] - r, center[2])
        drawer.drawTo(center[0], center[1] + r, center[2])

        points = np.linspace(0, 2 * np.pi, 25)
        for i in range(1, len(points)):
            prev = points[i - 1]
            curr = points[i]
            drawer.moveTo(center[0] + r * np.cos(prev), center[1] + r * np.sin(prev), center[2])
            drawer.drawTo(center[0] + r * np.cos(curr), center[1] + r * np.sin(curr), center[2])

    def table_edge_point(self, start, through):
        direction = through[:2] - start[:2]
        length = np.linalg.norm(direction)
        if length < 1e-9:
            return None

        direction = direction / length
        bounds = self.table_bounds()
        t_values = []
        for axis, lower, upper in ((0, bounds[0], bounds[1]), (1, bounds[2], bounds[3])):
            if abs(direction[axis]) < 1e-9:
                continue
            for boundary in (lower, upper):
                t = (boundary - start[axis]) / direction[axis]
                if t > 0:
                    other_axis = 1 - axis
                    other_value = start[other_axis] + direction[other_axis] * t
                    other_lower = bounds[2] if other_axis == 1 else bounds[0]
                    other_upper = bounds[3] if other_axis == 1 else bounds[1]
                    if other_lower - 1e-6 <= other_value <= other_upper + 1e-6:
                        t_values.append(t)

        if not t_values:
            return None

        endpoint_xy = start[:2] + direction * min(t_values)
        return np.array([endpoint_xy[0], endpoint_xy[1], start[2]], dtype=float)

    def table_bounds(self):
        segments = multisystem.active.table.cushion_segments.linear.values()
        xs = []
        ys = []
        for segment in segments:
            xs.extend([segment.p1[0], segment.p2[0]])
            ys.extend([segment.p1[1], segment.p2[1]])

        return min(xs), max(xs), min(ys), max(ys)


class AimMode(BaseMode):
    name = Mode.aim
    keymap = {
        Action.rotate_cue_left: False,
        Action.rotate_cue_right: False,
        Action.fine_control: False,
        Action.adjust_head: False,
        Action.quit: False,
        Action.stroke: False,
        Action.view: False,
        Action.camera_drag: False,
        Action.cue_drag: False,
        Action.target_drag: False,
        Action.toggle_other_balls: False,
        Action.exec_shot: False,
        Action.power: False,
        Action.elevation: False,
        Action.english: False,
        Action.cam_save: False,
        Action.cam_load: False,
        Action.show_help: False,
        Action.pick_ball: False,
        Action.call_shot: False,
        Action.ball_in_hand: False,
        Action.prev_shot: False,
        Action.introspect: False,
        Action.scroll_up: False,
        Action.scroll_down: False,
    }

    def __init__(self):
        super().__init__()

        # In this state, the cue sticks to the cue_avoid.min_theta
        self.magnet_theta = True
        # if cue angle is within this many degrees from cue_avoid.min_theta, it sticks
        # to cue_avoid.min_theta
        self.magnet_threshold = 0.2
        self.aim_guide = AimGuide(self)

    def enter(self, load_prev_cam=False):
        mouse.mode(MouseMode.ABSOLUTE)
        hud.show_aim_controls()

        if not visual.cue.has_focus:
            ball_id = multisystem.active.cue.cue_ball_id
            visual.cue.init_focus(visual.balls[ball_id])
        else:
            visual.cue.match_ball_position()

        visual.cue.show_nodes(ignore=("cue_cseg",))
        visual.cue.get_node("cue_stick").setX(0)
        self.aim_guide.update()
        self.aim_guide.point_cue_at_target()
        self.configure_ball_filter()
        self.apply_ball_visibility()

        # Fixate the camera onto the cueing ball
        cueing_ball_id = multisystem.active.cue.cue_ball_id
        cam.move_fixation(visual.balls[cueing_ball_id].get_node("pos").getPos())

        if load_prev_cam:
            cam.load_saved_state(Mode.aim)

        self.register_keymap_event("escape", Action.quit, True)
        self.register_keymap_event("f", Action.fine_control, True)
        self.register_keymap_event("f-up", Action.fine_control, False)
        self.register_keymap_event("t", Action.adjust_head, True)
        self.register_keymap_event("t-up", Action.adjust_head, False)
        tasks.register_event("mouse1", self._start_mouse_drag)
        tasks.register_event("mouse1-up", self._stop_mouse_drag)
        tasks.register_event("toggle-other-balls", self.toggle_other_balls)
        self.register_keymap_event("j", Action.rotate_cue_left, True)
        self.register_keymap_event("j-up", Action.rotate_cue_left, False)
        self.register_keymap_event("k", Action.rotate_cue_right, True)
        self.register_keymap_event("k-up", Action.rotate_cue_right, False)
        self.register_keymap_event("s", Action.stroke, True)
        self.register_keymap_event("v", Action.view, True)
        self.register_keymap_event("1", Action.cam_save, True)
        self.register_keymap_event("2", Action.cam_load, True)
        self.register_keymap_event("h", Action.show_help, True)
        self.register_keymap_event("q", Action.pick_ball, True)
        self.register_keymap_event("c", Action.call_shot, True)
        self.register_keymap_event("g", Action.ball_in_hand, True)
        self.register_keymap_event("b", Action.elevation, True)
        self.register_keymap_event("b-up", Action.elevation, False)
        self.register_keymap_event("e", Action.english, True)
        self.register_keymap_event("e-up", Action.english, False)
        self.register_keymap_event("x", Action.power, True)
        self.register_keymap_event("x-up", Action.power, False)
        self.register_keymap_event("space", Action.exec_shot, True)
        self.register_keymap_event("space-up", Action.exec_shot, False)
        self.register_keymap_event("p-up", Action.prev_shot, True)
        self.register_keymap_event("i", Action.introspect, True)
        self.register_keymap_event("i-up", Action.introspect, False)
        self.register_keymap_event("wheel_up", Action.scroll_up, True)
        self.register_keymap_event("wheel_down", Action.scroll_down, True)

        if settings.gameplay.cue_collision:
            tasks.add(cue_avoid.collision_task, "collision_task")

        tasks.add(self.aim_task, "aim_task")
        tasks.add(self.shared_task, "shared_task")

    def exit(self):
        self.aim_guide.destroy()
        hud.hide_aim_controls()

        if not shot_filter.applied and multisystem.active is not None:
            shot_filter.configure(enabled=False, target_ball_id=None)
            self.apply_ball_visibility()

        tasks.remove("aim_task")
        tasks.remove("shared_task")

        if settings.gameplay.cue_collision:
            tasks.remove("collision_task")

        cam.store_state(Mode.aim, overwrite=True)

    def aim_task(self, task):
        if self.keymap[Action.view]:
            Global.mode_mgr.change_mode(Mode.view, enter_kwargs=dict(move_active=True))
            return task.done
        elif self.keymap[Action.stroke]:
            Global.mode_mgr.change_mode(Mode.stroke)
        elif self.keymap[Action.pick_ball]:
            Global.mode_mgr.change_mode(Mode.pick_ball)
        elif self.keymap[Action.call_shot]:
            Global.mode_mgr.change_mode(Mode.call_shot)
        elif self.keymap[Action.ball_in_hand]:
            Global.mode_mgr.change_mode(Mode.ball_in_hand)
        elif self.keymap[Action.scroll_up] or self.keymap[Action.scroll_down]:
            self.zoom_from_wheel()
        elif self.keymap[Action.adjust_head]:
            cam.rotate_via_mouse(theta_only=True)
            self.cue_avoidance()
        elif self.keymap[Action.elevation]:
            self.aim_elevate_cue()
        elif self.keymap[Action.english]:
            self.apply_english()
        elif self.keymap[Action.power]:
            self.aim_apply_power()
        elif self.keymap[Action.target_drag]:
            self.aim_guide.update(target_drag=True)
            self.configure_ball_filter()
            self.apply_ball_visibility()
        elif self.keymap[Action.cue_drag]:
            self.drag_cue()
        elif self.keymap[Action.camera_drag]:
            cam.rotate_via_mouse(fine_control=self.keymap[Action.fine_control])
            self.keep_camera_above_cue()
        elif self.keymap[Action.exec_shot]:
            self.keymap[Action.exec_shot] = False
            if not self.aim_guide.has_target():
                self.aim_guide.prompt_for_target()
                self.configure_ball_filter()
                self.apply_ball_visibility()
            elif Global.game.shot_constraints.can_shoot():
                Global.mode_mgr.mode_stroked_from = Mode.aim
                visual.cue.set_object_state_as_render_state(skip_V0=True)
                self.configure_ball_filter()
                shot_filter.apply_to_system(multisystem.active)
                multisystem.active.strike()
                Global.mode_mgr.change_mode(Mode.calculate)
        elif self.keymap[Action.prev_shot]:
            self.keymap[Action.prev_shot] = False
            if len(multisystem) > 1:
                visual.switch_to_shot(multisystem.active_index - 1)
                self._update_hud()
                Global.mode_mgr.change_mode(Mode.shot)
                return task.done
        elif self.keymap[Action.rotate_cue_left] or self.keymap[Action.rotate_cue_right]:
            self.rotate_cue_from_keyboard()
        else:
            mouse.track()

        return task.cont

    def rotate_cue_from_keyboard(self):
        keyboard_rotation = 0.1 if self.keymap[Action.fine_control] else 2
        if self.keymap[Action.rotate_cue_left]:
            self.rotate_cue_by(keyboard_rotation)
        elif self.keymap[Action.rotate_cue_right]:
            self.rotate_cue_by(-keyboard_rotation)

    def rotate_cue_by(self, delta_phi):
        phi = (multisystem.active.cue.phi + delta_phi) % 360
        self.set_cue_phi(phi)

    def set_cue_phi(self, phi):
        multisystem.active.cue.set_state(phi=phi)
        visual.cue.set_render_state_as_object_state()
        self.cue_avoidance()
        self._update_hud()

    def drag_cue(self):
        target = self.mouse_table_point()

        if target is not None:
            ball_pos = visual.cue.follow.get_node("pos").getPos(Global.render)
            dx = ball_pos.x - target.x
            dy = ball_pos.y - target.y
            if np.hypot(dx, dy) > 1e-4:
                phi = np.degrees(np.arctan2(dy, dx)) % 360
                self.set_cue_phi(phi)
                mouse.track()
                return

        sensitivity = (
            rotate_fine_sensitivity_x
            if self.keymap[Action.fine_control]
            else rotate_sensitivity_x
        )
        with mouse:
            self.rotate_cue_by(-sensitivity * mouse.get_dx())

    def zoom_from_wheel(self):
        if self.keymap[Action.scroll_up]:
            cam.zoom(MOUSE_WHEEL_ZOOM)
            self.keymap[Action.scroll_up] = False

        if self.keymap[Action.scroll_down]:
            cam.zoom(-MOUSE_WHEEL_ZOOM)
            self.keymap[Action.scroll_down] = False

    def keep_camera_above_cue(self):
        theta = -visual.cue.get_node("cue_stick_focus").getR()
        if cam.theta < theta + min_camera:
            cam.rotate(theta=theta + min_camera)

    def _start_mouse_drag(self):
        if self.aim_guide.select_from_cursor():
            self.keymap[Action.target_drag] = True
            self.aim_guide.point_cue_at_target()
            self.aim_guide.update()
            self.configure_ball_filter()
            self.apply_ball_visibility()
        elif self.cursor_over_cue():
            self.keymap[Action.cue_drag] = True
        else:
            self.keymap[Action.camera_drag] = True

        mouse.track()

    def _stop_mouse_drag(self):
        self.keymap[Action.camera_drag] = False
        self.keymap[Action.cue_drag] = False
        self.keymap[Action.target_drag] = False
        mouse.track()

    def toggle_other_balls(self):
        if shot_filter.enabled:
            shot_filter.configure(enabled=False, target_ball_id=None)
        elif self.aim_guide.has_target():
            shot_filter.configure(
                enabled=True,
                target_ball_id=self.aim_guide.target_ball_id,
            )
        else:
            self.aim_guide.prompt_for_target()

        self.apply_ball_visibility()

    def configure_ball_filter(self):
        if not shot_filter.enabled:
            shot_filter.configure(enabled=False, target_ball_id=None)
            return

        if self.aim_guide.has_target():
            shot_filter.configure(
                enabled=True,
                target_ball_id=self.aim_guide.target_ball_id,
            )
        else:
            shot_filter.configure(enabled=False, target_ball_id=None)

    def apply_ball_visibility(self):
        visible_ball_ids = shot_filter.visible_ball_ids(multisystem.active)

        for ball_id, ball_render in visual.balls.items():
            if ball_id in visible_ball_ids:
                ball_render.show_nodes()
            else:
                ball_render.hide_nodes()

        hud.update_other_balls_button(shot_filter.enabled)

    def cursor_over_cue(self):
        distance = self.screen_distance_to_cue()
        return distance is not None and distance <= CUE_DRAG_HOTZONE_RADIUS

    def screen_distance_to_cue(self):
        if not mouse.mouse.hasMouse():
            return None

        cue_stick = visual.cue.get_node("cue_stick")
        ball_radius = visual.cue.follow._ball.params.R
        tip = Global.render.getRelativePoint(cue_stick, Point3(ball_radius, 0, 0))
        butt = Global.render.getRelativePoint(
            cue_stick, Point3(ball_radius + CUE_DRAG_HOTZONE_LENGTH, 0, 0)
        )

        screen_tip = self.project_to_screen(tip)
        screen_butt = self.project_to_screen(butt)
        if screen_tip is None or screen_butt is None:
            return None

        cursor = np.array([mouse.get_x(), mouse.get_y()])
        segment = screen_butt - screen_tip
        segment_length_sq = np.dot(segment, segment)
        if segment_length_sq == 0:
            return np.linalg.norm(cursor - screen_tip)

        t = np.clip(np.dot(cursor - screen_tip, segment) / segment_length_sq, 0, 1)
        closest = screen_tip + t * segment
        return np.linalg.norm(cursor - closest)

    def project_to_screen(self, point):
        projected = Point2()
        camera_point = cam.node.getRelativePoint(Global.render, point)
        if not cam.lens.project(camera_point, projected):
            return None

        return np.array([projected.x, projected.y])

    def mouse_table_point(self, z=None):
        if not mouse.mouse.hasMouse():
            return None

        near_point = Point3()
        far_point = Point3()
        if not cam.lens.extrude(
            Point2(mouse.get_x(), mouse.get_y()), near_point, far_point
        ):
            return None

        near = Global.render.getRelativePoint(cam.node, near_point)
        far = Global.render.getRelativePoint(cam.node, far_point)
        direction = far - near
        if abs(direction.z) < 1e-6:
            return None

        if z is None:
            z = visual.cue.follow.get_node("pos").getZ(Global.render)

        t = (z - near.z) / direction.z
        return near + direction * t

    def cue_avoidance(self):
        _, _, theta, *_ = visual.cue.get_render_state()

        if (theta < cue_avoid.min_theta) or self.magnet_theta:
            theta = cue_avoid.min_theta
            system_cue = multisystem.active.cue
            system_cue.set_state(theta=theta)
            system_cue_ball = multisystem.active.balls[system_cue.cue_ball_id]
            visual.cue.set_render_state_as_object_state()
            hud.update_cue(system_cue, system_cue_ball)

        if cam.theta < theta + min_camera:
            cam.rotate(theta=theta + min_camera)

    def aim_apply_power(self):
        with mouse:
            dy = mouse.get_dy()

        V0 = multisystem.active.cue.V0 + dy * power_sensitivity
        if V0 < min_stroke_speed:
            V0 = min_stroke_speed
        if V0 > max_stroke_speed:
            V0 = max_stroke_speed

        multisystem.active.cue.set_state(V0=V0)
        self._update_hud()

    def aim_elevate_cue(self):
        cue = visual.cue.get_node("cue_stick_focus")

        with mouse:
            delta_elevation = mouse.get_dy() * elevate_sensitivity

        old_elevation = -cue.getR()
        new_elevation = max(0, min(max_elevate, old_elevation + delta_elevation))

        if cue_avoid.min_theta >= new_elevation - self.magnet_threshold:
            # user set theta to minimum value, resume cushion tracking
            self.magnet_theta = True
            new_elevation = cue_avoid.min_theta
        else:
            # theta has been modified by the user, so no longer tracks the cushion
            self.magnet_theta = False

        cue.setR(-new_elevation)

        if cam.theta < (new_elevation + min_camera):
            cam.rotate(theta=new_elevation + min_camera)

        multisystem.active.cue.set_state(theta=new_elevation)
        self._update_hud()

    def apply_english(self):
        with mouse:
            dx, dy = mouse.get_dx(), mouse.get_dy()

        cue = visual.cue.get_node("cue_stick")
        cue_focus = visual.cue.get_node("cue_stick_focus")

        R = visual.cue.follow._ball.params.R

        delta_y, delta_z = dx * english_sensitivity, dy * english_sensitivity

        # y corresponds to side spin, z to top/bottom spin
        new_y = cue.getY() + delta_y
        new_z = cue.getZ() + delta_z

        cue_axis_offset = (
            np.array([-new_y, new_z]) / R
        )  # components normalized to ball radius
        contact_point_offset = tip_contact_offset(
            cue_axis_offset, multisystem.active.cue.specs.tip_radius, R
        )

        norm = norm2d(contact_point_offset)
        if norm > max_english:
            limit_scaling_factor = max_english / norm
            new_y *= limit_scaling_factor
            new_z *= limit_scaling_factor
            cue_axis_offset *= limit_scaling_factor
            contact_point_offset *= limit_scaling_factor

        cue.setY(new_y)
        cue.setZ(new_z)

        # if application of english increases min_theta beyond current elevation,
        # increase elevation
        if (
            self.magnet_theta
            or cue_avoid.min_theta >= -cue_focus.getR() - self.magnet_threshold
        ):
            cue_focus.setR(-cue_avoid.min_theta)

        if cam.theta < (new_theta := -cue_focus.getR() + min_camera):
            cam.rotate(theta=new_theta)

        multisystem.active.cue.set_state(
            a=contact_point_offset[0],
            b=contact_point_offset[1],
            theta=-cue_focus.getR(),
        )

        self._update_hud()

    def _update_hud(self) -> None:
        """Update HUD with current system's cue and cue ball"""
        system_cue = multisystem.active.cue
        hud.update_cue(system_cue, multisystem.active.balls[system_cue.cue_ball_id])

import numpy as np
from robosuite.utils.mjcf_utils import new_element, new_site
from robosuite.utils.placement_samplers import SequentialCompositeSampler

from libero.libero.envs.bddl_base_domain import BDDLBaseDomain, register_problem
from libero.libero.envs.objects import *
from libero.libero.envs.objects.drawing_objects import blank_board_pixels, flatten_strokes, sample_drawing_pixels
from libero.libero.envs.predicates import *
from libero.libero.envs.regions import *
from libero.libero.envs.utils import rectangle2xyrange


@register_problem
class Libero_Drawing_Copy(BDDLBaseDomain):
    def __init__(self, bddl_file_name, *args, **kwargs):
        self.workspace_name = "main_table"
        self.visualization_sites_list = []
        self.table_full_size = (1.15, 1.35, 0.05)
        self.table_offset = (0, 0, 0.90)
        self.z_offset = 0.01 - self.table_full_size[2]
        self.reference_curve = None
        self.reference_strokes = []
        self.reference_sketch_type = "unknown"
        self.reference_image = None
        self._reference_texture_dirty = False
        self.draw_canvas = None
        self._last_draw_uv = None
        self._last_ink_geom_uv = None
        self._draw_segment_count = 0
        self._episode_seed = 0
        self._texture_id_cache = {}
        self.scripted_ink_mode = False
        self.inking_enabled = False
        kwargs.update({"robots": [f"Mounted{robot_name}" for robot_name in kwargs["robots"]]})
        kwargs.update({"workspace_offset": self.table_offset})
        kwargs.update({"arena_type": "table"})
        kwargs.setdefault("scene_xml", "scenes/libero_tabletop_base_style.xml")
        kwargs.setdefault(
            "scene_properties",
            {"floor_style": "light-gray", "wall_style": "light-gray-plaster"},
        )
        super().__init__(bddl_file_name, *args, **kwargs)

    def _load_fixtures_in_arena(self, mujoco_arena):
        for fixture_category in list(self.parsed_problem["fixtures"].keys()):
            if fixture_category == "table":
                continue
            for fixture_instance in self.parsed_problem["fixtures"][fixture_category]:
                self.fixtures_dict[fixture_instance] = get_object_fn(fixture_category)(
                    name=fixture_instance,
                    joints=None,
                )

    def _load_objects_in_arena(self, mujoco_arena):
        for category_name, object_names in self.parsed_problem["objects"].items():
            for object_name in object_names:
                self.objects_dict[object_name] = get_object_fn(category_name)(
                    name=object_name,
                    joints=[dict(type="free", damping="0.02")],
                )

    def _load_sites_in_arena(self, mujoco_arena):
        object_sites_dict = {}
        for object_region_name, region in self.parsed_problem["regions"].items():
            if "main_table" not in object_region_name:
                continue
            ranges = region["ranges"][0]
            zone_size = ((ranges[2] - ranges[0]) / 2, (ranges[3] - ranges[1]) / 2)
            zone_centroid_xy = ((ranges[2] + ranges[0]) / 2, (ranges[3] + ranges[1]) / 2)
            target_zone = TargetZone(
                name=object_region_name,
                rgba=region["rgba"],
                zone_size=zone_size,
                zone_centroid_xy=zone_centroid_xy,
            )
            object_sites_dict[object_region_name] = target_zone
            mujoco_arena.table_body.append(
                new_site(
                    name=target_zone.name,
                    pos=target_zone.pos,
                    quat=target_zone.quat,
                    rgba=target_zone.rgba,
                    size=target_zone.size,
                    type="box",
                )
            )
        self.object_sites_dict = object_sites_dict

    def _load_model(self):
        super()._load_model()
        eef_body = self.model.worldbody.find(".//body[@name='gripper0_eef']")
        if eef_body is not None:
            eef_body.append(
                new_site(
                    name="gripper0_vertical_brush_grasp_site",
                    pos=[0.0, 0.0, 0.0],
                    quat=[0.02008686, 0.70664746, 0.70699550, -0.02007697],
                    rgba=[0, 0, 1, 0],
                    size=[0.006],
                )
            )
        self.model.equality.append(
            new_element(
                "weld",
                name="drawing_brush_gripper_weld",
                body1="gripper0_eef",
                body2="soft_brush_1_main",
                solref=[0.004, 1],
                solimp=[0.985, 0.999, 0.0005],
                active=False,
            )
        )

    def _add_placement_initializer(self):
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        regions = self.parsed_problem["regions"]
        for state in self.parsed_problem["initial_state"]:
            if state[0] != "on" or state[2] not in regions:
                continue
            object_name = state[1]
            region_name = state[2]
            region = regions[region_name]
            if region["target"] != "main_table":
                continue
            obj = self.fixtures_dict.get(object_name, self.objects_dict.get(object_name))
            if obj is None:
                continue
            x_ranges, y_ranges = rectangle2xyrange(region["ranges"])
            sampler = MultiRegionRandomSampler(
                f"{object_name}_sampler",
                mujoco_objects=obj,
                x_ranges=x_ranges,
                y_ranges=y_ranges,
                rotation=region["yaw_rotation"],
                rotation_axis="z",
                z_offset=self.z_offset,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=False,
                reference_pos=self.workspace_offset,
            )
            self.placement_initializer.append_sampler(sampler)

    def _check_success(self):
        brush = self.object_states_dict.get("soft_brush_1")
        draw_board = self.object_states_dict.get("drawing_board_blank_1")
        if brush is None or draw_board is None:
            return False
        return bool(eval_predicate_fn("on", brush, draw_board))

    def _setup_camera(self, mujoco_arena):
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.5886131746834771, 0.0, 1.4903500240372423],
            quat=[0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349],
        )
        mujoco_arena.set_camera(
            camera_name="frontview",
            pos=[1.0, 0.0, 1.45],
            quat=[0.56, 0.43, 0.43, 0.56],
        )
        mujoco_arena.set_camera(camera_name="overheadview", pos=[0.0, 0.0, 2.2], quat=[0.7071068, 0.0, 0.0, 0.7071068])

    def reset(self):
        super().reset()
        self._episode_seed += 1
        self._reset_drawing_textures()
        self.sim.forward()
        self._update_observables(force=True)
        return self._get_observations()

    def set_drawing_seed(self, seed: int):
        self._episode_seed = int(seed)
        self._reset_drawing_textures()
        self.sim.forward()
        self._update_observables(force=True)

    def _post_process(self):
        super()._post_process()
        if self.reference_image is not None and self._reference_texture_dirty:
            self._set_texture_rgb("drawing_board_reference_tex", self.reference_image)
            self._reference_texture_dirty = False
        if self.draw_canvas is not None and not self.scripted_ink_mode and self.inking_enabled:
            self._update_brush_ink()
        elif self.draw_canvas is not None and not self.scripted_ink_mode:
            self._last_draw_uv = None
            self._last_ink_geom_uv = None

    def _reset_drawing_textures(self):
        reference, strokes, sketch_type = sample_drawing_pixels(self._episode_seed)
        self.reference_strokes = [np.asarray(stroke, dtype=np.float32) for stroke in strokes if len(stroke)]
        self.reference_curve = flatten_strokes(self.reference_strokes)
        self.reference_sketch_type = str(sketch_type)
        self.reference_image = reference
        self._reference_texture_dirty = True
        self.draw_canvas = blank_board_pixels()
        self._last_draw_uv = None
        self._last_ink_geom_uv = None
        self._draw_segment_count = 0
        self._set_texture_rgb("drawing_board_reference_tex", reference)
        self._set_texture_rgb("drawing_board_blank_tex", self.draw_canvas)
        self._render_stroke_geoms("drawing_board_reference_1", "ref_ink", self.reference_strokes)
        self._hide_ink_geoms("draw_ink")

    def _set_texture_rgb(self, texture_name: str, image: np.ndarray):
        tex_id = self._texture_id_cache.get(texture_name)
        if tex_id is None:
            tex_id = self._find_texture_id(texture_name)
            if tex_id is not None:
                self._texture_id_cache[texture_name] = tex_id
        if tex_id is None:
            return
        h = int(self.sim.model.tex_height[tex_id])
        w = int(self.sim.model.tex_width[tex_id])
        adr = int(self.sim.model.tex_adr[tex_id])
        img = image
        if img.shape[:2] != (h, w):
            import cv2

            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        tex = getattr(self.sim.model, "tex_rgb", None)
        if tex is not None:
            tex[adr : adr + h * w] = img.reshape(-1, 3)
            self._upload_texture(tex_id)
            return
        tex_data = self.sim.model.tex_data
        nchannel = int(self.sim.model.tex_nchannel[tex_id])
        if nchannel == 3:
            tex_data[adr : adr + h * w * 3] = img.reshape(-1)
        elif nchannel == 4:
            alpha = np.full((h, w, 1), 255, dtype=np.uint8)
            rgba = np.concatenate([img.astype(np.uint8), alpha], axis=-1)
            tex_data[adr : adr + h * w * 4] = rgba.reshape(-1)
        else:
            gray = img.mean(axis=-1).astype(np.uint8)
            tex_data[adr : adr + h * w] = gray.reshape(-1)
        self._upload_texture(tex_id)

    def _find_texture_id(self, texture_name: str):
        names = self._texture_names()
        if texture_name in names:
            return names.index(texture_name)
        for idx, name in enumerate(names):
            if name.endswith(texture_name):
                return idx
        return None

    def _texture_names(self) -> list[str]:
        model = self.sim.model
        if hasattr(model, "texture_names"):
            return list(model.texture_names)
        names = []
        raw_names = getattr(model, "names", b"")
        for idx in range(int(getattr(model, "ntex", 0))):
            adr = int(model.name_texadr[idx])
            end = raw_names.find(b"\x00", adr)
            if end < 0:
                end = len(raw_names)
            names.append(raw_names[adr:end].decode("utf-8"))
        return names

    def _upload_texture(self, tex_id: int):
        for ctx_name in ("render_context", "_render_context_offscreen", "viewer"):
            ctx = getattr(self.sim, ctx_name, None)
            if ctx is not None and hasattr(ctx, "upload_texture"):
                try:
                    ctx.upload_texture(tex_id)
                    return
                except Exception:
                    pass
        for ctx in getattr(self.sim, "render_contexts", []):
            if hasattr(ctx, "upload_texture"):
                try:
                    ctx.upload_texture(tex_id)
                    return
                except Exception:
                    pass

    def _update_brush_ink(self):
        tip = self.brush_tip_world_pos()
        board_pos = self.sim.data.body_xpos[self.obj_body_id["drawing_board_blank_1"]]
        rel = tip - board_pos
        surface_z = board_pos[2] + 0.018
        z_rel = tip[2] - surface_z
        if z_rel < -0.04 or z_rel > 0.006:
            self._last_draw_uv = None
            self._last_ink_geom_uv = None
            return
        u = rel[0] / 0.38 + 0.5
        v = rel[1] / 0.28 + 0.5
        if not (0.04 <= u <= 0.96 and 0.04 <= v <= 0.96):
            self._last_draw_uv = None
            self._last_ink_geom_uv = None
            return
        uv = np.array([u, v], dtype=np.float32)
        last_uv = None if self._last_draw_uv is None else self._last_draw_uv.copy()
        last_geom_uv = None if self._last_ink_geom_uv is None else self._last_ink_geom_uv.copy()
        self._draw_to_canvas(uv, last_uv)
        if last_geom_uv is None:
            self._append_ink_point("draw_ink", uv)
            self._last_ink_geom_uv = uv
        elif float(np.linalg.norm(uv - last_geom_uv)) >= 0.008:
            self._append_ink_segment("draw_ink", last_geom_uv, uv)
            self._last_ink_geom_uv = uv
        self._last_draw_uv = uv
        self._set_texture_rgb("drawing_board_blank_tex", self.draw_canvas)

    def brush_tip_world_pos(self) -> np.ndarray:
        body_id = self.obj_body_id["soft_brush_1"]
        center = np.array(self.sim.data.body_xpos[body_id])
        mat = np.array(self.sim.data.body_xmat[body_id]).reshape(3, 3)
        return center + mat @ np.array([0.0, 0.0, -0.018])

    def force_draw_tip(self, tip_world: np.ndarray):
        board_pos = self.sim.data.body_xpos[self.obj_body_id["drawing_board_blank_1"]]
        rel = np.asarray(tip_world, dtype=np.float64) - board_pos
        uv = np.array([rel[0] / 0.38 + 0.5, rel[1] / 0.28 + 0.5], dtype=np.float32)
        if not (0.04 <= uv[0] <= 0.96 and 0.04 <= uv[1] <= 0.96):
            return
        last_uv = None if self._last_draw_uv is None else self._last_draw_uv.copy()
        self._draw_to_canvas(uv, last_uv)
        if last_uv is None:
            self._append_ink_point("draw_ink", uv)
        else:
            self._append_ink_segment("draw_ink", last_uv, uv)
        self._last_draw_uv = uv
        self._set_texture_rgb("drawing_board_blank_tex", self.draw_canvas)

    def _draw_to_canvas(self, uv: np.ndarray, last_uv: np.ndarray | None):
        import cv2

        if not self.draw_canvas.flags.writeable:
            self.draw_canvas = self.draw_canvas.copy()
        h, w = self.draw_canvas.shape[:2]
        p = np.round(uv * np.array([w - 1, h - 1], dtype=np.float32)).astype(np.int32)
        if last_uv is None:
            cv2.circle(self.draw_canvas, tuple(p), 4, (8, 8, 8), -1, lineType=cv2.LINE_AA)
        else:
            q = np.round(last_uv * np.array([w - 1, h - 1], dtype=np.float32)).astype(np.int32)
            cv2.line(self.draw_canvas, tuple(q), tuple(p), (8, 8, 8), 7, lineType=cv2.LINE_AA)

    def _render_curve_geoms(self, board_name: str, ink_prefix: str, curve: np.ndarray):
        self._render_stroke_geoms(board_name, ink_prefix, [curve])

    def _render_stroke_geoms(self, board_name: str, ink_prefix: str, strokes: list[np.ndarray]):
        self._hide_ink_geoms(ink_prefix)
        if not strokes or self._ink_geom_id(ink_prefix, 0) is None:
            return
        if ink_prefix.startswith("ref_"):
            self._render_reference_dot_geoms(ink_prefix, strokes)
            return
        segments: list[tuple[np.ndarray, np.ndarray]] = []
        dots: list[np.ndarray] = []
        for stroke in strokes:
            curve = np.asarray(stroke, dtype=np.float32)
            if len(curve) == 1:
                dots.append(curve[0])
            elif len(curve) > 1:
                segments.extend((np.asarray(a), np.asarray(b)) for a, b in zip(curve[:-1], curve[1:]))

        capacity = self._ink_geom_capacity(ink_prefix)
        count = 0
        if segments:
            if len(segments) > capacity:
                pick = np.linspace(0, len(segments) - 1, capacity).round().astype(int)
                selected_segments = [segments[int(i)] for i in pick]
            else:
                selected_segments = segments
            for a, b in selected_segments:
                if not self._set_ink_segment(ink_prefix, count, a, b):
                    return
                count += 1
        for uv in dots:
            if count >= capacity or not self._set_ink_dot(ink_prefix, count, uv):
                return
            count += 1

    def _render_reference_dot_geoms(self, ink_prefix: str, strokes: list[np.ndarray]):
        points: list[np.ndarray] = []
        for stroke in strokes:
            curve = np.asarray(stroke, dtype=np.float32)
            if len(curve) == 1:
                points.append(curve[0])
            elif len(curve) > 1:
                for a, b in zip(curve[:-1], curve[1:]):
                    dense = self._dense_uv_segment(a, b, step=0.010)
                    points.extend(np.asarray(p, dtype=np.float32) for p in dense[:-1])
                points.append(curve[-1])
        capacity = self._ink_geom_capacity(ink_prefix)
        if not points or capacity <= 0:
            return
        if len(points) > capacity:
            pick = np.linspace(0, len(points) - 1, capacity).round().astype(int)
            points = [points[int(i)] for i in pick]
        for idx, uv in enumerate(points[:capacity]):
            if not self._set_ink_dot(ink_prefix, idx, uv):
                return

    def _hide_ink_geoms(self, ink_prefix: str):
        idx = 0
        while True:
            geom_id = self._ink_geom_id(ink_prefix, idx)
            if geom_id is None:
                break
            self.sim.model.geom_rgba[geom_id] = np.array([0.01, 0.01, 0.01, 0.0])
            idx += 1

    def _append_ink_point(self, ink_prefix: str, uv: np.ndarray):
        eps = np.array([0.002, 0.0], dtype=np.float32)
        self._append_ink_segment(ink_prefix, uv - eps, uv + eps)

    def _append_ink_segment(self, ink_prefix: str, a: np.ndarray, b: np.ndarray):
        self._set_ink_segment(ink_prefix, self._draw_segment_count, a, b)
        self._draw_segment_count += 1

    def _ink_geom_capacity(self, ink_prefix: str) -> int:
        count = 0
        while self._ink_geom_id(ink_prefix, count) is not None:
            count += 1
        return count

    def _dense_uv_segment(self, a: np.ndarray, b: np.ndarray, step: float = 0.015) -> np.ndarray:
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        dist = float(np.linalg.norm(b - a))
        n = max(2, int(np.ceil(dist / step)) + 1)
        return np.asarray([(1.0 - t) * a + t * b for t in np.linspace(0.0, 1.0, n)], dtype=np.float32)

    def _set_ink_dot(self, ink_prefix: str, idx: int, uv: np.ndarray) -> bool:
        geom_id = self._ink_geom_id(ink_prefix, idx)
        if geom_id is None:
            return False
        local = np.array([(float(uv[0]) - 0.5) * 0.38, (float(uv[1]) - 0.5) * 0.28], dtype=np.float64)
        self.sim.model.geom_pos[geom_id] = np.array([local[0], local[1], 0.032])
        radius = 0.0048 if ink_prefix.startswith("ref_") else 0.0065
        self.sim.model.geom_size[geom_id] = np.array([radius, radius, 0.0012])
        self.sim.model.geom_quat[geom_id] = np.array([1.0, 0.0, 0.0, 0.0])
        self.sim.model.geom_rgba[geom_id] = np.array([0.01, 0.01, 0.01, 1.0])
        return True

    def _set_ink_segment(self, ink_prefix: str, idx: int, a: np.ndarray, b: np.ndarray) -> bool:
        geom_id = self._ink_geom_id(ink_prefix, idx)
        if geom_id is None:
            return False
        local_a = np.array([(float(a[0]) - 0.5) * 0.38, (float(a[1]) - 0.5) * 0.28], dtype=np.float64)
        local_b = np.array([(float(b[0]) - 0.5) * 0.38, (float(b[1]) - 0.5) * 0.28], dtype=np.float64)
        d = local_b - local_a
        length = float(np.linalg.norm(d))
        if length < 1e-4:
            length = 1e-4
        mid = 0.5 * (local_a + local_b)
        theta = float(np.arctan2(d[1], d[0]))
        self.sim.model.geom_pos[geom_id] = np.array([mid[0], mid[1], 0.032])
        half_width = 0.0048 if ink_prefix.startswith("ref_") else 0.010
        self.sim.model.geom_size[geom_id] = np.array([max(0.0015, length / 2.0), half_width, 0.0012])
        self.sim.model.geom_quat[geom_id] = np.array([np.cos(theta / 2.0), 0.0, 0.0, np.sin(theta / 2.0)])
        self.sim.model.geom_rgba[geom_id] = np.array([0.01, 0.01, 0.01, 1.0])
        return True

    def _ink_geom_id(self, ink_prefix: str, idx: int):
        if ink_prefix.startswith("ref_ink"):
            prefixes = ("drawing_board_reference_1",)
        elif ink_prefix.startswith("draw_ink"):
            prefixes = ("drawing_board_blank_1",)
        else:
            prefixes = ("drawing_board_reference_1", "drawing_board_blank_1")
        for object_prefix in prefixes:
            name = f"{object_prefix}_{ink_prefix}_{idx:03d}"
            try:
                return self.sim.model.geom_name2id(name)
            except Exception:
                continue
        return None

    def get_drawing_waypoints(self) -> np.ndarray:
        strokes = self.get_drawing_stroke_waypoints()
        if not strokes:
            return np.empty((0, 3), dtype=np.float32)
        return np.concatenate(strokes, axis=0)

    def get_drawing_stroke_waypoints(self) -> list[np.ndarray]:
        if self.reference_curve is None:
            return []
        board_pos = self.sim.data.body_xpos[self.obj_body_id["drawing_board_blank_1"]]
        surface_z = board_pos[2] + 0.018
        out = []
        for stroke in self.reference_strokes:
            pts = []
            for u, v in np.asarray(stroke, dtype=np.float32):
                x = board_pos[0] + (float(u) - 0.5) * 0.38
                y = board_pos[1] + (float(v) - 0.5) * 0.28
                z = surface_z
                pts.append([x, y, z])
            if pts:
                out.append(np.asarray(pts, dtype=np.float32))
        return out

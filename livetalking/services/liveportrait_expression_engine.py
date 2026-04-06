from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from logger import logger


class LivePortraitExpressionEngine:
    """
    生成 LivePortrait “表情底图 crop”（与 avatar 的 face_imgs 对齐）。

    设计目标：
    1) 只负责生成表情 crop。
    2) 嘴巴/口型仍由现有 wav2lip/musetalk/lightreal 在 paste_back 阶段覆盖（保证口型=语音一致）。
    """

    def __init__(
        self,
        face_list_cycle: List[np.ndarray],
        expression_ids: Optional[List[str]] = None,
    ):
        self.face_list_cycle = face_list_cycle
        self.expression_ids = expression_ids or ["neutral", "smile", "concerned", "encouraging"]

        self._expr_driving_idx: Dict[str, int] = self._choose_driving_indices()
        self._expr_driving_xd_new: Dict[str, torch.Tensor] = {}
        self._src_feature_cache: Dict[int, torch.Tensor] = {}
        self._src_xs_cache: Dict[int, torch.Tensor] = {}
        self._expression_face_cache: Dict[Tuple[str, int], np.ndarray] = {}

        self._init_liveportrait_wrapper()
        self._precompute_driving_kp()

    def _init_liveportrait_wrapper(self) -> None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        liveportrait_root = os.path.join(project_root, "LivePortrait")
        if liveportrait_root not in sys.path:
            sys.path.append(liveportrait_root)

        # LivePortrait 代码内部使用 `from src.xxx import ...` 相对导入约定。
        from src.config.inference_config import InferenceConfig  # type: ignore
        from src.live_portrait_wrapper import LivePortraitWrapper  # type: ignore

        cfg = InferenceConfig()
        cfg.flag_use_half_precision = True

        self.wrapper = LivePortraitWrapper(inference_cfg=cfg)

    def _bgr_to_rgb(self, img_bgr: np.ndarray) -> np.ndarray:
        return img_bgr[..., ::-1].copy()

    def _prepare_face_tensor(self, face_bgr: np.ndarray) -> torch.Tensor:
        img_rgb = self._bgr_to_rgb(face_bgr)
        return self.wrapper.prepare_source(img_rgb)

    def _compute_kp_trans_for_face(self, idx: int) -> torch.Tensor:
        face_bgr = self.face_list_cycle[idx]
        x_prepared = self._prepare_face_tensor(face_bgr)
        kp_info = self.wrapper.get_kp_info(x_prepared)
        x_trans = self.wrapper.transform_keypoint(kp_info)
        return x_trans

    def _precompute_driving_kp(self) -> None:
        for eid in self.expression_ids:
            driving_idx = self._expr_driving_idx.get(eid, 0)
            try:
                xd_new = self._compute_kp_trans_for_face(driving_idx)
                self._expr_driving_xd_new[eid] = xd_new
            except Exception as e:
                logger.warning(f"[LivePortraitExpressionEngine] precompute driving kp failed eid={eid}: {e}")

    def _choose_driving_indices(self) -> Dict[str, int]:
        """
        从 avatar 的 face_imgs 中用启发式挑 driving 帧。

        说明：我们不额外下载 driving 样例，改用你现有 avatar 数据的帧差异来近似不同表情。
        """
        n = len(self.face_list_cycle)
        if n <= 0:
            return {eid: 0 for eid in self.expression_ids}  # type: ignore

        mouth_scores = np.zeros(n, dtype=np.float32)
        eye_scores = np.zeros(n, dtype=np.float32)

        # 256x256 crop 内粗略区域（经验值）
        eye_roi = (60, 120, 70, 186)  # y0,y1,x0,x1
        mouth_roi = (140, 230, 60, 196)

        for i, img_bgr in enumerate(self.face_list_cycle):
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

            y0, y1, x0, x1 = eye_roi
            eye_mean = float(gray[y0:y1, x0:x1].mean())

            y0, y1, x0, x1 = mouth_roi
            mouth_mean = float(gray[y0:y1, x0:x1].mean())

            mouth_open_score = 1.0 - mouth_mean
            eye_open_score = eye_mean

            mouth_scores[i] = mouth_open_score
            eye_scores[i] = eye_open_score

        # 由于最终会用 lip-sync 覆盖嘴部区域，所以这里尽量让不同 expression 在“眼睛/眉眼/上半脸”更明显。
        neutral_obj = (-0.35 * mouth_scores) + (0.95 * eye_scores)
        smile_obj = (0.18 * mouth_scores) + (1.15 * eye_scores)
        concerned_obj = (-0.15 * mouth_scores) + (1.30 * eye_scores)
        encouraging_obj = (0.22 * mouth_scores) + (0.75 * eye_scores)

        def _pick_distinct(obj: np.ndarray, banned: set[int]) -> int:
            order = np.argsort(-obj)  # desc
            for i in order:
                ii = int(i)
                if ii not in banned:
                    return ii
            return int(order[0])

        chosen: Dict[str, int] = {}
        banned: set[int] = set()
        for name, obj in [
            ("neutral", neutral_obj),
            ("smile", smile_obj),
            ("concerned", concerned_obj),
            ("encouraging", encouraging_obj),
        ]:
            v = _pick_distinct(obj, banned)
            chosen[name] = v
            banned.add(int(v))

        logger.info(f"[LivePortraitExpressionEngine] driving indices selected: {chosen}")
        return chosen

    def get_expression_face(self, idx: int, expression_id: str) -> np.ndarray:
        """
        返回：BGR uint8，shape=(H,W,3)，通常为 (256,256,3)。
        """
        if expression_id not in self.expression_ids:
            expression_id = "neutral"

        key = (expression_id, idx)
        cached = self._expression_face_cache.get(key)
        if cached is not None:
            return cached

        try:
            xd_new = self._expr_driving_xd_new.get(expression_id)
            if xd_new is None:
                raise RuntimeError(f"Missing driving kp for expression_id={expression_id}")

            # source 的特征与 kp 变换按需缓存
            if idx not in self._src_feature_cache or idx not in self._src_xs_cache:
                face_bgr = self.face_list_cycle[idx]
                x_prepared = self._prepare_face_tensor(face_bgr)
                feature_3d = self.wrapper.extract_feature_3d(x_prepared)
                kp_info = self.wrapper.get_kp_info(x_prepared)
                x_s = self.wrapper.transform_keypoint(kp_info)
                self._src_feature_cache[idx] = feature_3d
                self._src_xs_cache[idx] = x_s

            f_s = self._src_feature_cache[idx]
            x_s = self._src_xs_cache[idx]

            out = self.wrapper.warp_decode(f_s, x_s, xd_new)
            out_rgb = self.wrapper.parse_output(out["out"])[0]
            out_bgr = out_rgb[..., ::-1].copy()

            self._expression_face_cache[key] = out_bgr
            return out_bgr
        except Exception as e:
            logger.warning(
                f"[LivePortraitExpressionEngine] get_expression_face failed eid={expression_id} idx={idx}: {e}"
            )
            return self.face_list_cycle[idx]


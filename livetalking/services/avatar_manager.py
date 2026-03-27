from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

from logger import logger


@dataclass
class AvatarResources:
    model: Any
    avatar: Any
    loaded_at: float


class AvatarManager:
    """
    负责：
    - 根据 avatar 目录结构推断模型类型
    - 预加载并缓存 (model, avatar) 资源
    - 为会话构建 BaseReal 实例（并发安全：不修改全局 opt）
    """

    def __init__(self, opt, get_config_value):
        self._opt = opt
        self._get_config_value = get_config_value

        self.preload_enabled: bool = False
        self.preload_all_on_start: bool = False

        # key: "{model_type}_{avatar_id}"
        self.avatar_cache: Dict[str, AvatarResources] = {}

    def configure(self, preload_enabled: bool, preload_all_on_start: bool) -> None:
        self.preload_enabled = bool(preload_enabled)
        self.preload_all_on_start = bool(preload_all_on_start)

    def infer_model_type(self, avatar_id: str, fallback_model: str) -> str:
        if not avatar_id:
            return fallback_model

        avatar_path = f"./data/avatars/{avatar_id}"

        # ultralight
        if os.path.exists(os.path.join(avatar_path, "ultralight.pth")):
            return "ultralight"

        # musetalk
        if os.path.exists(os.path.join(avatar_path, "latents.pt")) and os.path.exists(
            os.path.join(avatar_path, "mask_coords.pkl")
        ):
            return "musetalk"

        # wav2lip
        if os.path.exists(os.path.join(avatar_path, "coords.pkl")) and os.path.exists(
            os.path.join(avatar_path, "face_imgs")
        ):
            return "wav2lip"

        # 兜底：允许按名字推断
        if self._get_config_value("avatar.model_infer_from_id", True):
            aid = avatar_id.lower()
            if "wav2lip" in aid:
                return "wav2lip"
            if "ultralight" in aid:
                return "ultralight"

        return fallback_model

    def _cache_key(self, model_type: str, avatar_id: str) -> str:
        return f"{model_type}_{avatar_id}"

    def _resolve_custom_actions(self, target_avatar_id: str) -> List[Dict[str, Any]]:
        """
        按头像过滤 custom action 配置，保证不同数字人动作编排互不干扰。

        兼容两种写法：
        1) 旧格式（list，条目无 avatar_id）：视为“全局动作”，对所有头像生效
        2) 新格式（list，条目可带 avatar_id）：只对匹配头像生效
        """
        raw = getattr(self._opt, "customopt", [])
        if not raw:
            return []

        # 兼容把 customopt 写成 dict 的情况（可选增强）
        if isinstance(raw, dict):
            # 约定：{"default":[...], "wav2lip256_avatar1":[...], ...}
            default_items = raw.get("default", [])
            avatar_items = raw.get(target_avatar_id, [])
            merged = []
            if isinstance(default_items, list):
                merged.extend(default_items)
            if isinstance(avatar_items, list):
                merged.extend(avatar_items)
            return merged

        if not isinstance(raw, list):
            return []

        resolved: List[Dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            aid = item.get("avatar_id", None)
            if aid is None or str(aid).strip() == "":
                # 旧格式：无 avatar_id，视为全局动作
                resolved.append(item)
            elif str(aid) == str(target_avatar_id):
                resolved.append(item)
        return resolved

    def preload_avatar_resources(self, avatar_id: str) -> Optional[AvatarResources]:
        model_type = self.infer_model_type(avatar_id, getattr(self._opt, "model", "musetalk"))
        cache_key = self._cache_key(model_type, avatar_id)

        if cache_key in self.avatar_cache:
            logger.info(f"资源已在缓存中: {cache_key}")
            return self.avatar_cache[cache_key]

        logger.info(f"开始预加载资源: {cache_key}")

        if model_type == "wav2lip":
            from lipreal import load_model, load_avatar

            current_model = load_model("./models/wav2lip.pth")
            current_avatar = load_avatar(avatar_id)
        elif model_type == "musetalk":
            from musereal import load_model, load_avatar

            current_model = load_model()
            current_avatar = load_avatar(avatar_id)
        elif model_type == "ultralight":
            from lightreal import load_model, load_avatar

            current_model = load_model(self._opt)
            current_avatar = load_avatar(avatar_id)
        else:
            logger.error(f"不支持的模型类型: {model_type}")
            return None

        resources = AvatarResources(model=current_model, avatar=current_avatar, loaded_at=time.time())
        self.avatar_cache[cache_key] = resources
        logger.info(f"资源预加载完成: {cache_key}")
        return resources

    def build_nerfreal(self, sessionid: int, avatar_id: Optional[str] = None):
        # 并发安全：每个会话复制一份 opt
        session_opt = copy.copy(self._opt)
        session_opt.sessionid = sessionid

        target_avatar_id = avatar_id if avatar_id is not None else getattr(self._opt, "avatar_id", None)
        session_opt.customopt = self._resolve_custom_actions(target_avatar_id)
        logger.info(
            f"session={sessionid} avatar={target_avatar_id} custom_actions={len(getattr(session_opt, 'customopt', []))}"
        )
        model_type = self.infer_model_type(target_avatar_id, getattr(self._opt, "model", "musetalk"))
        cache_key = self._cache_key(model_type, target_avatar_id)

        resources = None
        if self.preload_enabled and cache_key in self.avatar_cache:
            logger.info(f"从缓存加载资源: {cache_key}")
            resources = self.avatar_cache[cache_key]

        if model_type == "wav2lip":
            from lipreal import LipReal, load_model, load_avatar

            current_model = resources.model if resources else load_model("./models/wav2lip.pth")
            current_avatar = resources.avatar if resources else load_avatar(target_avatar_id)
            return LipReal(session_opt, current_model, current_avatar)

        if model_type == "musetalk":
            from musereal import MuseReal, load_model, load_avatar

            current_model = resources.model if resources else load_model()
            current_avatar = resources.avatar if resources else load_avatar(target_avatar_id)
            return MuseReal(session_opt, current_model, current_avatar)

        if model_type == "ultralight":
            from lightreal import LightReal, load_model, load_avatar

            current_model = resources.model if resources else load_model(self._opt)
            current_avatar = resources.avatar if resources else load_avatar(target_avatar_id)
            return LightReal(session_opt, current_model, current_avatar)

        raise ValueError(f"Unsupported model_type={model_type}")

    def preload_all_avatars_on_start(self) -> int:
        avatars_dir = "data/avatars"
        loaded = 0

        if not os.path.exists(avatars_dir):
            logger.warning(f"avatars 目录不存在: {avatars_dir}")
            return 0

        for item in os.listdir(avatars_dir):
            item_path = os.path.join(avatars_dir, item)
            if not os.path.isdir(item_path) or item == ".gitkeep":
                continue

            # 至少要能被识别为有效头像目录
            if os.path.exists(os.path.join(item_path, "coords.pkl")) and os.path.exists(os.path.join(item_path, "face_imgs")):
                if self.preload_avatar_resources(item) is not None:
                    loaded += 1
            else:
                logger.warning(f"跳过无效头像目录: {item}")

        return loaded


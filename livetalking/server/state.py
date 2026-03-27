from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Set, Tuple

import asyncio

from basereal import BaseReal
from livetalking.services.avatar_manager import AvatarManager
from knowledge_base import StudyAbroadKnowledgeBase


@dataclass
class AppState:
    # core config / services
    opt: Any
    avatar_manager: AvatarManager
    get_config_value: Callable[[str, Any], Any]
    rand_sessionid: Callable[[], int]

    # session state
    nerfreals: Dict[int, BaseReal]
    identities: Dict[int, str]
    avatar_identities: Dict[str, str]
    save_avatar_identities: Callable[[], bool]

    # preload queue
    preload_queue: List[Tuple[int, str]]
    preload_in_progress: bool

    # webrtc connections
    pcs: Set[Any]  # aiortc.RTCPeerConnection

    # runtime guards / stability
    chat_semaphore: asyncio.Semaphore

    # generation id per session (used to cancel in-flight jobs)
    chat_gen_ids: Dict[int, str]

    # knowledge base (SQLite) for keyword RAG
    kb: StudyAbroadKnowledgeBase


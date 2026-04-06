from __future__ import annotations

import asyncio
import json
import os
import torch
import time
import uuid
import sqlite3
import random
import string
import io

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceServer, RTCConfiguration
from aiortc.rtcrtpsender import RTCRtpSender

from logger import logger
from webrtc import HumanPlayer

from livetalking.server.state import AppState
from livetalking.services.chat_service import llm_response_with_identity
from typing import Optional
from livetalking.server.auth import (
    create_token,
    decode_token,
    get_bearer_token_from_auth_header,
)


def _sync_avatar_profiles_from_disk(state: AppState) -> None:
    """Ensure SQLite admin profiles exist for every valid folder under data/avatars (e.g. new assets without server restart)."""
    avatars_dir = "data/avatars"
    if not os.path.exists(avatars_dir):
        return
    try:
        for item in os.listdir(avatars_dir):
            item_path = os.path.join(avatars_dir, item)
            if not os.path.isdir(item_path) or item == ".gitkeep":
                continue
            face_imgs_path = os.path.join(item_path, "face_imgs")
            coords_path = os.path.join(item_path, "coords.pkl")
            if os.path.exists(face_imgs_path) and os.path.exists(coords_path):
                inferred = state.avatar_manager.infer_model_type(item, getattr(state.opt, "model", "musetalk"))
                state.avatar_admin_store.ensure_avatar(item, item, inferred)
    except Exception as e:
        logger.warning(f"sync avatar profiles from disk: {e}")


def _try_gpu_util_percent() -> float | None:
    try:
        import subprocess

        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if p.returncode == 0 and p.stdout.strip():
            return float(p.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return None


def setup_routes(app: web.Application, state: AppState) -> None:
    def _require_admin(request: web.Request) -> dict | None:
        auth_header = request.headers.get("Authorization")
        token = get_bearer_token_from_auth_header(auth_header)
        payload = decode_token(token, state.auth_token_secret)
        if not payload or payload.get("role") != "admin":
            return None
        return payload

    def _require_login(request: web.Request) -> dict | None:
        auth_header = request.headers.get("Authorization")
        token = get_bearer_token_from_auth_header(auth_header)
        payload = decode_token(token, state.auth_token_secret)
        if not payload:
            return None
        if payload.get("role") not in ("admin", "student"):
            return None
        return payload

    async def offer(request: web.Request) -> web.Response:
        params = await request.json()
        offer_sdp = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        sessionid = state.rand_sessionid()

        state.nerfreals[sessionid] = None
        logger.info("sessionid=%d, session num=%d", sessionid, len(state.nerfreals))

        avatar_id = params.get("avatar_id", state.opt.avatar_id)
        logger.info(f"Creating nerfreal instance with avatar_id: {avatar_id}")
        nerfreal = await asyncio.get_event_loop().run_in_executor(None, state.avatar_manager.build_nerfreal, sessionid, avatar_id)
        state.nerfreals[sessionid] = nerfreal

        ice_server = RTCIceServer(urls="stun:stun.miwifi.com:3478")
        pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[ice_server]))
        state.pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info("Connection state is %s" % pc.connectionState)
            if pc.connectionState == "failed":
                await pc.close()
                state.pcs.discard(pc)
                if sessionid in state.nerfreals:
                    del state.nerfreals[sessionid]
            if pc.connectionState == "closed":
                state.pcs.discard(pc)
                if sessionid in state.nerfreals:
                    del state.nerfreals[sessionid]

        player = HumanPlayer(state.nerfreals[sessionid])
        pc.addTrack(player.audio)
        pc.addTrack(player.video)

        capabilities = RTCRtpSender.getCapabilities("video")
        preferences = list(filter(lambda x: x.name == "H264", capabilities.codecs))
        preferences += list(filter(lambda x: x.name == "VP8", capabilities.codecs))
        preferences += list(filter(lambda x: x.name == "rtx", capabilities.codecs))
        transceiver = pc.getTransceivers()[1]
        transceiver.setCodecPreferences(preferences)

        await pc.setRemoteDescription(offer_sdp)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.Response(
            content_type="application/json",
            text=json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "sessionid": sessionid}),
        )

    async def human(request: web.Request) -> web.Response:
        try:
            params = await request.json()
            sessionid = params.get("sessionid", 0)
            trace_id = uuid.uuid4().hex[:12]
            t0 = time.perf_counter()
            rag_evidence_for_resp = None
            assistant_text = ""
            avatar_id_for_req = str(params.get("avatar_id", state.opt.avatar_id or "") or "")
            user_id_for_req = str(params.get("user_id", request.remote or "") or "")
            chat_session_db_id = None
            llm_cost_ms = None
            requested_chat_session_id: Optional[int] = None
            try:
                requested_chat_session_id = int(params.get("chat_session_id", 0) or 0)
            except Exception:
                requested_chat_session_id = None

            if params.get("interrupt"):
                state.nerfreals[sessionid].flush_talk()

            if params["type"] == "echo":
                state.nerfreals[sessionid].put_msg_txt(params["text"])
            elif params["type"] == "chat":
                identity = state.identities.get(sessionid)
                llm_timeout_s = float(state.get_config_value("server.llm_timeout_seconds", 30))
                fallback_text = str(
                    state.get_config_value(
                        "server.llm_fallback_text",
                        "我正在整理答案，稍后再为你详细说明。",
                    )
                )

                # RAG evidence retrieval (keyword / hybrid)
                rag_enabled = bool(state.get_config_value("rag.enabled", True))
                rag_mode = str(state.get_config_value("rag.retrieval_mode", "keyword")).lower()
                rag_top_k = int(state.get_config_value("rag.top_k", 5))
                rag_min_credibility = float(state.get_config_value("rag.min_credibility", 0.7))
                rag_category = state.get_config_value("rag.category", None)
                if rag_category == "":
                    rag_category = None
                rag_alpha = float(state.get_config_value("rag.hybrid_alpha", 0.5))
                rag_max_chars = int(state.get_config_value("rag.max_content_chars", 600))

                rag_evidence = None
                if rag_enabled:
                    try:
                        if rag_mode == "hybrid":
                            rag_results = state.kb.search_knowledge_hybrid(
                                params["text"],
                                category=rag_category,
                                min_credibility=rag_min_credibility,
                                top_k=rag_top_k,
                                alpha=rag_alpha,
                            )
                        else:
                            rag_results = state.kb.search_knowledge(
                                params["text"],
                                category=rag_category,
                                min_credibility=rag_min_credibility,
                            )
                            rag_results = rag_results[:rag_top_k] if rag_results else []

                        if rag_results:
                            # Trim evidence content for prompt size control
                            rag_evidence = []
                            for ev in rag_results:
                                content = ev.get("content", "") or ""
                                ev_out = dict(ev)
                                ev_out["content_excerpt"] = content[:rag_max_chars].strip()
                                rag_evidence.append(ev_out)
                        else:
                            rag_evidence = None
                    except Exception as e:
                        logger.warning(f"[trace={trace_id}] rag retrieval failed: {e}")
                        rag_evidence = None
                rag_evidence_for_resp = rag_evidence

                # create chat session record (best-effort)
                try:
                    rag_evidence_json = json.dumps(rag_evidence or [], ensure_ascii=False)
                    if requested_chat_session_id and requested_chat_session_id > 0:
                        # verify ownership for student
                        try:
                            payload = _require_login(request)
                        except Exception:
                            payload = None
                        if payload and payload.get("role") != "admin":
                            detail = state.chat_history.get_session_detail(int(requested_chat_session_id))
                            if str(detail["session"].get("user_id", "")) != str(payload.get("u", "")):
                                return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))
                        chat_session_db_id = int(requested_chat_session_id)
                    else:
                        chat_session_db_id = state.chat_history.create_chat_session(
                            trace_id=trace_id,
                            sessionid=int(sessionid),
                            user_id=user_id_for_req,
                            avatar_id=avatar_id_for_req,
                            rag_enabled=bool(rag_enabled),
                            rag_mode=str(rag_mode),
                            rag_hit_count=len(rag_evidence or []) if rag_enabled else 0,
                            rag_evidence_json=rag_evidence_json,
                        )
                    state.chat_history.add_message(chat_session_db_id, "user", params["text"])
                except Exception as e:
                    logger.warning(f"[trace={trace_id}] chat history create failed: {e}")

                # Concurrency guard: avoid too many in-flight chats causing latency spikes / OOM.
                try:
                    await asyncio.wait_for(state.chat_semaphore.acquire(), timeout=0.001)
                except TimeoutError:
                    return web.Response(
                        content_type="application/json",
                        text=json.dumps({"code": -2, "msg": "busy, try again"}),
                    )

                user_text = params["text"]
                nerfreal = state.nerfreals[sessionid]
                gen_id = uuid.uuid4().hex
                state.chat_gen_ids[sessionid] = gen_id
                try:
                    nerfreal.reset_runtime_metrics()
                except Exception:
                    pass

                def is_current():
                    return state.chat_gen_ids.get(sessionid) == gen_id

                def _run_llm_job():
                    try:
                        t_job0 = time.perf_counter()
                        logger.info(f"[trace={trace_id}] chat start session={sessionid} len={len(user_text)}")
                        ui_lang = str(
                            params.get("lang")
                            or request.headers.get("X-UI-Lang")
                            or request.headers.get("Accept-Language", "zh-CN")
                        )
                        resp_text = llm_response_with_identity(
                            user_text,
                            nerfreal,
                            identity,
                            is_current,
                            rag_evidence=rag_evidence,
                            ui_lang=ui_lang,
                        )
                        logger.info(f"[trace={trace_id}] chat done dt={(time.perf_counter() - t_job0):.3f}s")
                        return resp_text or "", int((time.perf_counter() - t_job0) * 1000)
                    except Exception as e:
                        logger.exception(f"[trace={trace_id}] chat error: {e}")
                        return "", None

                # Run and wait for LLM completion up to timeout; if exceeded, do degradation.
                try:
                    future = asyncio.get_event_loop().run_in_executor(None, _run_llm_job)
                    assistant_text, llm_cost_ms = await asyncio.wait_for(future, timeout=llm_timeout_s)
                    # persist assistant message
                    try:
                        if chat_session_db_id is not None and assistant_text:
                            state.chat_history.add_message(chat_session_db_id, "assistant", assistant_text)
                    except Exception as e:
                        logger.warning(f"[trace={trace_id}] chat history add assistant failed: {e}")
                except asyncio.TimeoutError:
                    logger.warning(f"[trace={trace_id}] llm timeout after {llm_timeout_s}s, degrade")
                    # invalidate in-flight generation
                    state.chat_gen_ids[sessionid] = uuid.uuid4().hex
                    try:
                        nerfreal.flush_talk()
                        nerfreal.put_msg_txt(fallback_text)
                    except Exception:
                        pass
                    dt = time.perf_counter() - t0
                    try:
                        metrics = {}
                        try:
                            metrics = nerfreal.snapshot_runtime_metrics()
                        except Exception:
                            metrics = {}
                        if chat_session_db_id is not None:
                            state.chat_history.finish_chat_session(
                                chat_session_db_id,
                                int(dt * 1000),
                                llm_timeout=True,
                                llm_ms=llm_cost_ms,
                                tts_ms=metrics.get("tts_ms"),
                                action_ms=metrics.get("action_ms"),
                            )
                            state.chat_history.add_message(chat_session_db_id, "assistant", fallback_text)
                    except Exception:
                        pass
                    return web.Response(
                        content_type="application/json",
                        text=json.dumps(
                            {
                                "code": -3,
                                "msg": "llm timeout, fallback spoken",
                                "trace_id": trace_id,
                                "chat_session_id": chat_session_db_id,
                                "assistant_text": fallback_text,
                                "dt_ms": int(dt * 1000),
                            }
                        ),
                    )
                finally:
                    # release semaphore once per request
                    try:
                        state.chat_semaphore.release()
                    except Exception:
                        pass

            dt = time.perf_counter() - t0
            try:
                metrics = {}
                try:
                    metrics = nerfreal.snapshot_runtime_metrics()
                except Exception:
                    metrics = {}
                if chat_session_db_id is not None:
                    state.chat_history.finish_chat_session(
                        chat_session_db_id,
                        int(dt * 1000),
                        llm_timeout=False,
                        llm_ms=llm_cost_ms,
                        tts_ms=metrics.get("tts_ms"),
                        action_ms=metrics.get("action_ms"),
                    )
            except Exception:
                pass
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {
                        "code": 0,
                        "msg": "ok",
                        "trace_id": trace_id,
                        "chat_session_id": chat_session_db_id,
                        "dt_ms": int(dt * 1000),
                        "rag_evidence": rag_evidence_for_resp,
                        "assistant_text": assistant_text,
                    }
                ),
            )
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def my_sessions_create(request: web.Request) -> web.Response:
        try:
            payload = _require_login(request)
            if not payload:
                return web.Response(content_type="application/json", text=json.dumps({"code": -20, "msg": "unauthorized"}))
            params = await request.json()
            sessionid = int(params.get("sessionid", 0) or 0)
            avatar_id = str(params.get("avatar_id", "") or "").strip()
            if sessionid <= 0:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "sessionid required"}))
            if not avatar_id:
                avatar_id = str(getattr(state.opt, "avatar_id", "") or "")
            trace_id = uuid.uuid4().hex[:12]
            sid = state.chat_history.create_empty_session(trace_id, sessionid, str(payload.get("u", "")), avatar_id)
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok", "data": {"id": sid}}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def interrupt_talk(request: web.Request) -> web.Response:
        try:
            params = await request.json()
            sessionid = params.get("sessionid", 0)
            state.nerfreals[sessionid].flush_talk()
            # invalidate in-flight job so it won't continue to put new speech
            state.chat_gen_ids[sessionid] = uuid.uuid4().hex
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok"}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def humanaudio(request: web.Request) -> web.Response:
        try:
            form = await request.post()
            sessionid = int(form.get("sessionid", 0))
            fileobj = form["file"]
            filebytes = fileobj.file.read()
            state.nerfreals[sessionid].put_audio_file(filebytes)
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok"}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def set_audiotype(request: web.Request) -> web.Response:
        try:
            params = await request.json()
            sessionid = params.get("sessionid", 0)
            state.nerfreals[sessionid].set_custom_state(params["audiotype"], params["reinit"])
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok"}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def set_tts_speed(request: web.Request) -> web.Response:
        try:
            params = await request.json()
            sessionid = int(params.get("sessionid", 0))
            speed = float(params.get("speed", 1.0))
            speed = max(0.5, min(2.0, speed))
            pct = int(round((speed - 1.0) * 100))
            rate_str = f"{pct:+d}%"

            nerfreal = state.nerfreals.get(sessionid)
            if nerfreal is None:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "invalid sessionid"}))

            # Set session-level runtime tts rate for upcoming utterances
            try:
                setattr(nerfreal.opt, "TTS_RATE", rate_str)
            except Exception:
                pass
            try:
                setattr(nerfreal.tts.opt, "TTS_RATE", rate_str)
            except Exception:
                pass

            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok", "data": {"speed": speed, "rate": rate_str}}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def set_tts_voice(request: web.Request) -> web.Response:
        try:
            params = await request.json()
            sessionid = int(params.get("sessionid", 0))
            voice = str(params.get("voice", "") or "").strip()
            if sessionid <= 0:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "invalid sessionid"}))
            if not voice:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "voice required"}))

            allowed = state.avatar_admin_store.list_allowed_voices(enabled_only=True)
            allowed_set = {str(v.get("voice", "")).strip() for v in allowed}
            if voice not in allowed_set:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "voice not allowed"}))

            nerfreal = state.nerfreals.get(sessionid)
            if nerfreal is None:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "invalid sessionid"}))

            try:
                setattr(nerfreal.opt, "REF_FILE", voice)
                setattr(nerfreal.opt, "_tts_voice_user_locked", True)
            except Exception:
                pass
            try:
                setattr(nerfreal.tts.opt, "REF_FILE", voice)
                setattr(nerfreal.tts.opt, "_tts_voice_user_locked", True)
            except Exception:
                pass
            # 同步到 fallback 引擎实例，避免主引擎失败切换后回到旧音色
            try:
                for fb in getattr(nerfreal, "fallback_tts_list", []) or []:
                    if hasattr(fb, "opt"):
                        setattr(fb.opt, "REF_FILE", voice)
                        setattr(fb.opt, "_tts_voice_user_locked", True)
            except Exception:
                pass
            logger.info(f"[TTS] session={sessionid} set voice={voice}")
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok", "data": {"voice": voice}}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def record(request: web.Request) -> web.Response:
        try:
            params = await request.json()
            sessionid = params.get("sessionid", 0)
            if params["type"] == "start_record":
                state.nerfreals[sessionid].start_recording()
            elif params["type"] == "end_record":
                state.nerfreals[sessionid].stop_recording()
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok"}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def is_speaking(request: web.Request) -> web.Response:
        params = await request.json()
        sessionid = params.get("sessionid", 0)
        return web.Response(content_type="application/json", text=json.dumps({"code": 0, "data": state.nerfreals[sessionid].is_speaking()}))

    async def get_avatars(request: web.Request) -> web.Response:
        try:
            # 与管理后台保持同源数据：先同步磁盘目录到管理库，再用管理库配置（名称/启用状态）返回前端
            _sync_avatar_profiles_from_disk(state)
            profiles = state.avatar_admin_store.list_profiles()
            profile_by_id = {str(p.get("avatar_id", "")): p for p in profiles if isinstance(p, dict)}

            avatars_dir = "data/avatars"
            avatar_list = []
            if os.path.exists(avatars_dir):
                logger.info(f"Scanning avatars directory: {avatars_dir}")
                for item in os.listdir(avatars_dir):
                    item_path = os.path.join(avatars_dir, item)
                    logger.info(f"Checking item: {item}, is_dir: {os.path.isdir(item_path)}")
                    if os.path.isdir(item_path) and item != ".gitkeep":
                        face_imgs_path = os.path.join(item_path, "face_imgs")
                        coords_path = os.path.join(item_path, "coords.pkl")
                        logger.info(
                            f"Checking files for {item}: face_imgs exists: {os.path.exists(face_imgs_path)}, coords exists: {os.path.exists(coords_path)}"
                        )
                        if os.path.exists(face_imgs_path) and os.path.exists(coords_path):
                            profile = profile_by_id.get(item, {})
                            # 管理后台可禁用某个数字人；用户前端列表应同步隐藏
                            if profile and profile.get("enabled") is False:
                                continue
                            avatar_type = (
                                str(profile.get("model_type", "")).strip()
                                or state.avatar_manager.infer_model_type(item, getattr(state.opt, "model", "musetalk"))
                            )
                            avatar_name = str(profile.get("name", "")).strip() or item
                            identity_type = str(profile.get("identity_type", "")).strip()
                            # 显示名兜底：若名称仍是目录ID，优先显示业务身份标签（如“英国留学顾问”）
                            display_name = avatar_name
                            if (not display_name or display_name == item) and identity_type:
                                display_name = identity_type
                            avatar_list.append(
                                {
                                    "id": item,
                                    "name": avatar_name,
                                    "display_name": display_name,
                                    "type": avatar_type,
                                    "identity_type": identity_type,
                                    "identity_desc": str(profile.get("identity_desc", "")).strip(),
                                    "enabled": bool(profile.get("enabled", True)),
                                }
                            )
                            logger.info(f"Added avatar: {item}")
            logger.info(f"Found {len(avatar_list)} available avatars: {[a['id'] for a in avatar_list]}")
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok", "data": avatar_list}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def switch_avatar(request: web.Request) -> web.Response:
        try:
            params = await request.json()
            sessionid = params.get("sessionid", 0)
            avatar_id = params.get("avatar_id", "wav2lip256_avatar1")

            logger.info(f"Switching avatar to {avatar_id} for session {sessionid}")
            old_nerfreal = state.nerfreals.get(sessionid)
            if old_nerfreal:
                old_nerfreal.flush_talk()
                if hasattr(old_nerfreal, "asr") and old_nerfreal.asr and hasattr(old_nerfreal.asr, "flush_talk"):
                    old_nerfreal.asr.flush_talk()
                if hasattr(old_nerfreal, "model"):
                    old_nerfreal.model = None
                if hasattr(old_nerfreal, "avatar"):
                    old_nerfreal.avatar = None
                del old_nerfreal
                import gc

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            new_nerfreal = await asyncio.get_event_loop().run_in_executor(None, state.avatar_manager.build_nerfreal, sessionid, avatar_id)
            state.nerfreals[sessionid] = new_nerfreal

            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok"}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def set_identity(request: web.Request) -> web.Response:
        try:
            params = await request.json()
            sessionid = params.get("sessionid", 0)
            identity = params.get("identity", "")
            state.identities[sessionid] = identity
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok"}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def get_avatar_identity(request: web.Request) -> web.Response:
        try:
            avatar_id = request.query.get("avatar_id", "")
            if not avatar_id:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "avatar_id is required"}))
            identity = state.avatar_identities.get(avatar_id, "")
            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": 0, "msg": "ok", "data": {"avatar_id": avatar_id, "identity": identity}}),
            )
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def auth_me(request: web.Request) -> web.Response:
        try:
            auth_header = request.headers.get("Authorization")
            token = get_bearer_token_from_auth_header(auth_header)
            payload = decode_token(token, state.auth_token_secret)
            if not payload:
                return web.Response(content_type="application/json", text=json.dumps({"code": -20, "msg": "unauthorized"}))
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {
                        "code": 0,
                        "msg": "ok",
                        "data": {"username": payload.get("u", ""), "role": payload.get("role", "")},
                    }
                ),
            )
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def auth_change_password(request: web.Request) -> web.Response:
        try:
            payload = _require_login(request)
            if not payload:
                return web.Response(content_type="application/json", text=json.dumps({"code": -20, "msg": "unauthorized"}))
            params = await request.json()
            old_password = str(params.get("old_password", ""))
            new_password = str(params.get("new_password", ""))
            if not old_password or not new_password:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "old/new password required"}))
            if len(new_password) < 6:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "new password too short"}))
            ok = state.auth_store.change_password(str(payload.get("u", "")), old_password, new_password)
            if not ok:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "old password incorrect"}))
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok"}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def admin_sessions_list(request: web.Request) -> web.Response:
        try:
            if not _require_admin(request):
                return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))

            q = request.query
            start_ms = q.get("start_ms")
            end_ms = q.get("end_ms")
            user_id = q.get("user_id", "")
            keyword = q.get("keyword", "")
            page = int(q.get("page", "1"))
            page_size = int(q.get("page_size", "20"))

            res = state.chat_history.list_sessions(
                start_ms=int(start_ms) if start_ms else None,
                end_ms=int(end_ms) if end_ms else None,
                user_id="",
                user_id_like=str(user_id or ""),
                keyword=str(keyword or ""),
                page=page,
                page_size=page_size,
            )
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok", "data": res}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def admin_sessions_detail(request: web.Request) -> web.Response:
        try:
            if not _require_admin(request):
                return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))
            sid = int(request.match_info.get("id", "0"))
            detail = state.chat_history.get_session_detail(sid)
            try:
                rej = detail["session"].get("rag_evidence_json", "")
                detail["session"]["rag_evidence"] = json.loads(rej) if rej else []
            except Exception:
                detail["session"]["rag_evidence"] = []
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok", "data": detail}))
        except KeyError:
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "not found"}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def my_sessions_list(request: web.Request) -> web.Response:
        try:
            payload = _require_login(request)
            if not payload:
                return web.Response(content_type="application/json", text=json.dumps({"code": -20, "msg": "unauthorized"}))

            q = request.query
            start_ms = q.get("start_ms")
            end_ms = q.get("end_ms")
            keyword = q.get("keyword", "")
            page = int(q.get("page", "1"))
            page_size = int(q.get("page_size", "20"))
            user_id = str(payload.get("u", ""))

            res = state.chat_history.list_sessions(
                start_ms=int(start_ms) if start_ms else None,
                end_ms=int(end_ms) if end_ms else None,
                user_id=user_id,
                user_id_like="",
                keyword=str(keyword or ""),
                page=page,
                page_size=page_size,
            )
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok", "data": res}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def my_sessions_detail(request: web.Request) -> web.Response:
        try:
            payload = _require_login(request)
            if not payload:
                return web.Response(content_type="application/json", text=json.dumps({"code": -20, "msg": "unauthorized"}))

            sid = int(request.match_info.get("id", "0"))
            detail = state.chat_history.get_session_detail(sid)
            try:
                rej = detail["session"].get("rag_evidence_json", "")
                detail["session"]["rag_evidence"] = json.loads(rej) if rej else []
            except Exception:
                detail["session"]["rag_evidence"] = []
            if payload.get("role") != "admin":
                user_id = str(payload.get("u", ""))
                if str(detail["session"].get("user_id", "")) != user_id:
                    return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok", "data": detail}))
        except KeyError:
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "not found"}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def my_sessions_rate(request: web.Request) -> web.Response:
        try:
            payload = _require_login(request)
            if not payload:
                return web.Response(content_type="application/json", text=json.dumps({"code": -20, "msg": "unauthorized"}))
            params = await request.json()
            sid = int(params.get("id", 0))
            score = int(params.get("score", 0))
            if sid <= 0 or score <= 0:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "id/score required"}))
            detail = state.chat_history.get_session_detail(sid)
            if payload.get("role") != "admin":
                user_id = str(payload.get("u", ""))
                if str(detail["session"].get("user_id", "")) != user_id:
                    return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))
            state.chat_history.set_satisfaction(sid, score)
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok"}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def my_sessions_delete(request: web.Request) -> web.Response:
        try:
            payload = _require_login(request)
            if not payload:
                return web.Response(content_type="application/json", text=json.dumps({"code": -20, "msg": "unauthorized"}))
            params = await request.json()
            sid = int(params.get("id", 0))
            if sid <= 0:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "id required"}))
            detail = state.chat_history.get_session_detail(sid)
            if payload.get("role") != "admin":
                user_id = str(payload.get("u", ""))
                if str(detail["session"].get("user_id", "")) != user_id:
                    return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))
            state.chat_history.delete_session(sid)
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok"}))
        except KeyError:
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "not found"}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def my_sessions_rename(request: web.Request) -> web.Response:
        try:
            payload = _require_login(request)
            if not payload:
                return web.Response(content_type="application/json", text=json.dumps({"code": -20, "msg": "unauthorized"}))
            params = await request.json()
            sid = int(params.get("id", 0))
            title = str(params.get("title", "")).strip()
            if sid <= 0 or not title:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "id/title required"}))
            detail = state.chat_history.get_session_detail(sid)
            if payload.get("role") != "admin":
                user_id = str(payload.get("u", ""))
                if str(detail["session"].get("user_id", "")) != user_id:
                    return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))
            ok = state.chat_history.rename_session(sid, title)
            if not ok:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "rename failed"}))
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok"}))
        except KeyError:
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "not found"}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def my_sessions_end(request: web.Request) -> web.Response:
        try:
            payload = _require_login(request)
            if not payload:
                return web.Response(content_type="application/json", text=json.dumps({"code": -20, "msg": "unauthorized"}))
            params = await request.json()
            sid = int(params.get("id", 0))
            if sid <= 0:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "id required"}))
            detail = state.chat_history.get_session_detail(sid)
            if payload.get("role") != "admin":
                user_id = str(payload.get("u", ""))
                if str(detail["session"].get("user_id", "")) != user_id:
                    return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))
            ok = state.chat_history.end_session(sid)
            if not ok:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "not found"}))
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok"}))
        except KeyError:
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "not found"}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def admin_sessions_delete(request: web.Request) -> web.Response:
        try:
            if not _require_admin(request):
                return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))
            params = await request.json()
            sid = int(params.get("id", 0))
            if sid <= 0:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "id required"}))
            state.chat_history.delete_session(sid)
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok"}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def admin_analytics_overview(request: web.Request) -> web.Response:
        try:
            if not _require_admin(request):
                return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))
            ov = state.chat_history.analytics_overview()
            ov["webrtc_connections"] = len(state.pcs)
            ov["active_real_slots"] = len(state.nerfreals)
            try:
                ov["registered_avatars"] = len(state.avatar_admin_store.list_profiles())
            except Exception:
                ov["registered_avatars"] = None
            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": 0, "msg": "ok", "data": {"overview": ov}}),
            )
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def admin_analytics_trend(request: web.Request) -> web.Response:
        try:
            if not _require_admin(request):
                return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))
            conn = sqlite3.connect(state.chat_history.db_path)
            cur = conn.cursor()
            rows = cur.execute(
                """
                SELECT strftime('%Y-%m-%d', created_at_ms/1000, 'unixepoch') AS d, COUNT(*)
                FROM chat_sessions
                GROUP BY d
                ORDER BY d ASC
                LIMIT 90
                """
            ).fetchall()
            conn.close()
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok", "data": [{"day": r[0], "count": r[1]} for r in rows]}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def admin_analytics_response_time(request: web.Request) -> web.Response:
        try:
            if not _require_admin(request):
                return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))
            conn = sqlite3.connect(state.chat_history.db_path)
            cur = conn.cursor()
            rows = cur.execute(
                """
                SELECT strftime('%Y-%m-%d', created_at_ms/1000, 'unixepoch') AS d,
                       AVG(dt_ms)
                FROM chat_sessions
                WHERE dt_ms IS NOT NULL
                GROUP BY d
                ORDER BY d ASC
                LIMIT 90
                """
            ).fetchall()
            conn.close()
            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": 0, "msg": "ok", "data": [{"day": r[0], "avg_dt_ms": int(r[1]) if r[1] is not None else None} for r in rows]}),
            )
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def admin_analytics_categories(request: web.Request) -> web.Response:
        try:
            if not _require_admin(request):
                return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))
            conn = sqlite3.connect(state.chat_history.db_path)
            cur = conn.cursor()
            rows = cur.execute(
                """
                SELECT LOWER(content) FROM chat_messages
                WHERE role='user'
                ORDER BY id DESC
                LIMIT 5000
                """
            ).fetchall()
            conn.close()

            counters = {
                "申请": 0,
                "签证": 0,
                "语言": 0,
                "国家/地区": 0,
                "费用": 0,
                "其他": 0,
            }
            for r in rows:
                q = str(r[0] or "")
                hit = False
                if any(k in q for k in ["申请", "文书", "offer", "选校", "专业"]):
                    counters["申请"] += 1
                    hit = True
                if any(k in q for k in ["签证", "visa", "拒签"]):
                    counters["签证"] += 1
                    hit = True
                if any(k in q for k in ["雅思", "托福", "gre", "gmat", "语言"]):
                    counters["语言"] += 1
                    hit = True
                if any(k in q for k in ["美国", "英国", "加拿大", "澳洲", "新加坡", "香港", "国家"]):
                    counters["国家/地区"] += 1
                    hit = True
                if any(k in q for k in ["费用", "学费", "生活费", "奖学金", "预算"]):
                    counters["费用"] += 1
                    hit = True
                if not hit:
                    counters["其他"] += 1

            data = [{"name": k, "value": int(v)} for k, v in counters.items() if v > 0]
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok", "data": data}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def admin_analytics_tts_success(request: web.Request) -> web.Response:
        try:
            if not _require_admin(request):
                return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))
            # Approximation: non-timeout chats are treated as success.
            conn = sqlite3.connect(state.chat_history.db_path)
            cur = conn.cursor()
            total = cur.execute("SELECT COUNT(*) FROM chat_sessions").fetchone()[0] or 0
            timeout_cnt = cur.execute("SELECT COUNT(*) FROM chat_sessions WHERE llm_timeout=1").fetchone()[0] or 0
            conn.close()
            success = max(0, total - timeout_cnt)
            rate = (success / total * 100.0) if total > 0 else 0.0
            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": 0, "msg": "ok", "data": {"total": int(total), "success": int(success), "failed": int(timeout_cnt), "success_rate": round(rate, 2)}}),
            )
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def admin_analytics_report(request: web.Request) -> web.Response:
        try:
            if not _require_admin(request):
                return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))
            conn = sqlite3.connect(state.chat_history.db_path)
            cur = conn.cursor()
            total = cur.execute("SELECT COUNT(*) FROM chat_sessions").fetchone()[0] or 0
            avg_dt = cur.execute("SELECT AVG(dt_ms) FROM chat_sessions WHERE dt_ms IS NOT NULL").fetchone()[0]
            by_day = cur.execute(
                "SELECT strftime('%Y-%m-%d', created_at_ms/1000, 'unixepoch') d, COUNT(*) c FROM chat_sessions GROUP BY d ORDER BY d ASC LIMIT 180"
            ).fetchall()
            timeout_cnt = cur.execute("SELECT COUNT(*) FROM chat_sessions WHERE llm_timeout=1").fetchone()[0] or 0
            by_avatar = cur.execute(
                """
                SELECT COALESCE(NULLIF(TRIM(avatar_id), ''), '(未指定)') AS aid, COUNT(*) AS c
                FROM chat_sessions
                GROUP BY COALESCE(NULLIF(TRIM(avatar_id), ''), '(未指定)')
                ORDER BY c DESC
                LIMIT 20
                """
            ).fetchall()
            conn.close()
            ov = state.chat_history.analytics_overview()
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {
                        "code": 0,
                        "msg": "ok",
                        "data": {
                            "generated_at_ms": int(time.time() * 1000),
                            "overview": ov,
                            "summary": {"total_sessions": int(total), "avg_dt_ms": int(avg_dt) if avg_dt else None, "llm_timeout_sessions": int(timeout_cnt)},
                            "trend": [{"day": r[0], "count": r[1]} for r in by_day],
                            "sessions_by_avatar": [{"avatar_id": r[0], "count": int(r[1])} for r in by_avatar],
                            "runtime": {
                                "webrtc_connections": len(state.pcs),
                                "active_real_slots": len(state.nerfreals),
                                "gpu_usage_percent": _try_gpu_util_percent(),
                            },
                        },
                    }
                ),
            )
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def admin_analytics_satisfaction(request: web.Request) -> web.Response:
        try:
            if not _require_admin(request):
                return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))
            conn = sqlite3.connect(state.chat_history.db_path)
            cur = conn.cursor()
            rows = cur.execute(
                """
                SELECT strftime('%Y-%m-%d', created_at_ms/1000, 'unixepoch') d, AVG(satisfaction)
                FROM chat_sessions
                WHERE satisfaction IS NOT NULL
                GROUP BY d
                ORDER BY d ASC
                LIMIT 180
                """
            ).fetchall()
            low = cur.execute(
                """
                SELECT id, user_id, created_at_ms, question_summary
                FROM (
                  SELECT s.id id, s.user_id user_id, s.created_at_ms created_at_ms,
                         (SELECT content FROM chat_messages m WHERE m.chat_session_id=s.id AND m.role='user' ORDER BY m.id ASC LIMIT 1) question_summary,
                         s.satisfaction satisfaction
                  FROM chat_sessions s
                )
                WHERE satisfaction IS NOT NULL AND satisfaction <= 2
                ORDER BY created_at_ms DESC
                LIMIT 30
                """
            ).fetchall()
            conn.close()
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {
                        "code": 0,
                        "msg": "ok",
                        "data": {
                            "trend": [{"day": r[0], "avg_satisfaction": round(float(r[1]), 2) if r[1] is not None else None} for r in rows],
                            "low_score_sessions": [
                                {"id": r[0], "user_id": r[1], "created_at_ms": r[2], "question_summary": (r[3] or "")[:80]}
                                for r in low
                            ],
                        },
                    }
                ),
            )
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def admin_analytics_wordcloud(request: web.Request) -> web.Response:
        try:
            if not _require_admin(request):
                return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))
            conn = sqlite3.connect(state.chat_history.db_path)
            cur = conn.cursor()
            rows = cur.execute(
                """
                SELECT content FROM chat_messages
                WHERE role='user'
                ORDER BY id DESC
                LIMIT 2000
                """
            ).fetchall()
            conn.close()
            text = " ".join((r[0] or "") for r in rows)
            # simple word split for mixed zh/en (not ideal, but dependency-free)
            import re
            tokens = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{3,}", text)
            stop = {"留学", "咨询", "数字人", "问题", "什么", "怎么", "可以", "需要", "是不是", "这个", "那个"}
            freq = {}
            for t in tokens:
                if t in stop:
                    continue
                freq[t] = freq.get(t, 0) + 1
            top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:80]
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok", "data": [{"word": k, "count": v} for k, v in top]}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def admin_analytics_performance(request: web.Request) -> web.Response:
        try:
            if not _require_admin(request):
                return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))
            gpu = _try_gpu_util_percent()
            data = {
                "concurrent_connections": len(state.pcs),
                "active_sessions": len(state.nerfreals),
                "gpu_usage_percent": gpu,
                "infer_fps": None,
                "final_fps": None,
                "system_latency_ms": None,
            }
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok", "data": data}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def admin_analytics_by_avatar(request: web.Request) -> web.Response:
        try:
            if not _require_admin(request):
                return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))
            conn = sqlite3.connect(state.chat_history.db_path)
            cur = conn.cursor()
            rows = cur.execute(
                """
                SELECT COALESCE(NULLIF(TRIM(avatar_id), ''), '(未指定)') AS aid, COUNT(*) AS c
                FROM chat_sessions
                GROUP BY COALESCE(NULLIF(TRIM(avatar_id), ''), '(未指定)')
                ORDER BY c DESC
                LIMIT 40
                """
            ).fetchall()
            conn.close()
            data = [{"avatar_id": r[0], "count": int(r[1])} for r in rows]
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok", "data": data}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    # -------------------------
    # Avatar admin APIs
    # -------------------------
    async def admin_avatar_list(request: web.Request) -> web.Response:
        try:
            if not _require_admin(request):
                return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))
            _sync_avatar_profiles_from_disk(state)
            data = state.avatar_admin_store.list_profiles()
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok", "data": data}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def admin_avatar_save(request: web.Request) -> web.Response:
        try:
            if not _require_admin(request):
                return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))
            p = await request.json()
            avatar_id = str(p.get("avatar_id", "")).strip()
            name = str(p.get("name", "")).strip() or avatar_id
            model_type = str(p.get("model_type", "")).strip() or "musetalk"
            identity_type = str(p.get("identity_type", "")).strip()
            identity_desc = str(p.get("identity_desc", "")).strip()
            enabled = bool(p.get("enabled", True))
            if not avatar_id:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "avatar_id required"}))
            state.avatar_admin_store.upsert_profile(avatar_id, name, model_type, identity_type, identity_desc, enabled)
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok"}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def admin_avatar_delete(request: web.Request) -> web.Response:
        try:
            if not _require_admin(request):
                return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))
            p = await request.json()
            avatar_id = str(p.get("avatar_id", "")).strip()
            if not avatar_id:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "avatar_id required"}))
            state.avatar_admin_store.delete_profile(avatar_id)
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok"}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def admin_avatar_copy(request: web.Request) -> web.Response:
        try:
            if not _require_admin(request):
                return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))
            p = await request.json()
            source_avatar_id = str(p.get("source_avatar_id", "")).strip()
            target_avatar_id = str(p.get("target_avatar_id", "")).strip()
            target_name = str(p.get("target_name", "")).strip() or target_avatar_id
            if not source_avatar_id or not target_avatar_id:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "source/target required"}))
            state.avatar_admin_store.copy_profile(source_avatar_id, target_avatar_id, target_name)
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok"}))
        except KeyError:
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "source not found"}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def admin_avatar_actions_get(request: web.Request) -> web.Response:
        try:
            if not _require_admin(request):
                return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))
            avatar_id = str(request.query.get("avatar_id", "")).strip()
            if not avatar_id:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "avatar_id required"}))
            data = state.avatar_admin_store.get_actions(avatar_id)
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok", "data": data}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def admin_avatar_actions_save(request: web.Request) -> web.Response:
        try:
            if not _require_admin(request):
                return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))
            p = await request.json()
            avatar_id = str(p.get("avatar_id", "")).strip()
            actions = p.get("actions", []) or []
            if not avatar_id or not isinstance(actions, list):
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "invalid params"}))
            state.avatar_admin_store.save_actions(avatar_id, actions)
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok"}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def admin_avatar_tts_get(request: web.Request) -> web.Response:
        try:
            if not _require_admin(request):
                return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))
            avatar_id = str(request.query.get("avatar_id", "")).strip()
            if not avatar_id:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "avatar_id required"}))
            data = state.avatar_admin_store.get_tts(avatar_id)
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok", "data": data}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def avatar_tts_get(request: web.Request) -> web.Response:
        try:
            avatar_id = str(request.query.get("avatar_id", "")).strip()
            if not avatar_id:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "avatar_id required"}))
            data = state.avatar_admin_store.get_tts(avatar_id)
            allowed = {
                str(it.get("voice") or "").strip()
                for it in state.avatar_admin_store.list_allowed_voices(enabled_only=True)
                if it.get("voice")
            }
            vbl_raw = data.get("voices_by_lang") if isinstance(data.get("voices_by_lang"), dict) else {}
            vbl = {str(k): str(v).strip() for k, v in vbl_raw.items() if str(v or "").strip() in allowed}
            vo = str(data.get("voice", "") or "").strip()
            if vo not in allowed:
                vo = ""
            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": 0, "msg": "ok", "data": {"voice": vo, "voices_by_lang": vbl}}),
            )
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def admin_avatar_tts_save(request: web.Request) -> web.Response:
        try:
            if not _require_admin(request):
                return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))
            p = await request.json()
            avatar_id = str(p.get("avatar_id", "")).strip()
            speed = p.get("speed", None)
            tone = p.get("tone", None)
            voice = str(p.get("voice", "") or "")
            keyword_pron = str(p.get("keyword_pron", "") or "")
            if not avatar_id:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "avatar_id required"}))
            speed_v = float(speed) if speed not in (None, "") else None
            tone_v = float(tone) if tone not in (None, "") else None
            state.avatar_admin_store.save_tts(avatar_id, speed_v, tone_v, voice, keyword_pron)
            vbl = p.get("voices_by_lang")
            if isinstance(vbl, dict):
                state.avatar_admin_store.save_tts_locales(avatar_id, vbl)
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok"}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def admin_avatar_tts_apply_all(request: web.Request) -> web.Response:
        try:
            if not _require_admin(request):
                return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))
            p = await request.json()
            source_avatar_id = str(p.get("avatar_id", "")).strip()
            if not source_avatar_id:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "avatar_id required"}))
            src = state.avatar_admin_store.get_tts(source_avatar_id)
            vbl = src.get("voices_by_lang") if isinstance(src.get("voices_by_lang"), dict) else {}
            for profile in state.avatar_admin_store.list_profiles():
                aid = profile.get("avatar_id")
                if not aid:
                    continue
                state.avatar_admin_store.save_tts(aid, src.get("speed"), src.get("tone"), src.get("voice", ""), src.get("keyword_pron", ""))
                state.avatar_admin_store.save_tts_locales(aid, dict(vbl))
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok"}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def admin_tts_allowed_voices_get(request: web.Request) -> web.Response:
        try:
            if not _require_admin(request):
                return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))
            data = state.avatar_admin_store.list_allowed_voices(enabled_only=False)
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok", "data": data}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def admin_tts_allowed_voices_save(request: web.Request) -> web.Response:
        try:
            if not _require_admin(request):
                return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))
            p = await request.json()
            items = p.get("items", [])
            if not isinstance(items, list):
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "items must be list"}))
            state.avatar_admin_store.save_allowed_voices(items)
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok"}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def tts_allowed_voices_get(request: web.Request) -> web.Response:
        try:
            data = state.avatar_admin_store.list_allowed_voices(enabled_only=True)
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok", "data": data}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    # -------------------------
    # Knowledge base admin APIs
    # -------------------------
    async def admin_kb_list(request: web.Request) -> web.Response:
        try:
            payload = _require_admin(request)
            if not payload:
                return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))

            q = request.query
            query = str(q.get("query", "") or "").strip()
            category = str(q.get("category", "") or "").strip()
            page = max(1, int(q.get("page", "1")))
            page_size = max(1, min(200, int(q.get("page_size", "20"))))
            offset = (page - 1) * page_size

            conn = sqlite3.connect(state.kb.db_path)
            cur = conn.cursor()
            conds = []
            params = []
            if query:
                conds.append("(title LIKE ? OR content LIKE ? OR tags LIKE ?)")
                params.extend([f"%{query}%", f"%{query}%", f"%{query}%"])
            if category:
                conds.append("category = ?")
                params.append(category)
            where_sql = ("WHERE " + " AND ".join(conds)) if conds else ""

            total = cur.execute(f"SELECT COUNT(*) FROM knowledge_entries {where_sql}", params).fetchone()[0]
            rows = cur.execute(
                f"""
                SELECT id, category, title, source, credibility_score, last_updated, tags
                FROM knowledge_entries
                {where_sql}
                ORDER BY last_updated DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                params + [page_size, offset],
            ).fetchall()
            conn.close()

            items = []
            for r in rows:
                items.append(
                    {
                        "id": r[0],
                        "category": r[1],
                        "title": r[2],
                        "source": r[3],
                        "credibility_score": r[4],
                        "last_updated": r[5],
                        "tags": json.loads(r[6]) if r[6] else [],
                    }
                )

            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": 0, "msg": "ok", "data": {"page": page, "page_size": page_size, "total": int(total), "items": items}}),
            )
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def admin_kb_detail(request: web.Request) -> web.Response:
        try:
            payload = _require_admin(request)
            if not payload:
                return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))

            entry_id = int(request.match_info.get("id", "0"))
            conn = sqlite3.connect(state.kb.db_path)
            cur = conn.cursor()
            row = cur.execute(
                "SELECT id, category, title, content, source, credibility_score, last_updated, expiration_date, tags FROM knowledge_entries WHERE id=?",
                (entry_id,),
            ).fetchone()
            conn.close()
            if not row:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "not found"}))
            data = {
                "id": row[0],
                "category": row[1],
                "title": row[2],
                "content": row[3],
                "source": row[4],
                "credibility_score": row[5],
                "last_updated": row[6],
                "expiration_date": row[7],
                "tags": json.loads(row[8]) if row[8] else [],
            }
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok", "data": data}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def admin_kb_add(request: web.Request) -> web.Response:
        try:
            payload = _require_admin(request)
            if not payload:
                return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))

            params = await request.json()
            category = str(params.get("category", "")).strip() or "未分类"
            title = str(params.get("title", "")).strip()
            content = str(params.get("content", "")).strip()
            source = str(params.get("source", "")).strip() or "用户贡献"
            tags = params.get("tags", []) or []
            expiration_days = int(params.get("expiration_days", 365))
            if not title or not content:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "title/content required"}))

            entry_id = state.kb.add_knowledge_entry(
                category=category,
                title=title,
                content=content,
                source=source,
                tags=tags if isinstance(tags, list) else [],
                expiration_days=expiration_days,
            )

            # audit (reuse update_logs with new_content only)
            try:
                conn = sqlite3.connect(state.kb.db_path)
                conn.execute(
                    "INSERT INTO update_logs(entry_id, update_type, old_content, new_content) VALUES (?, ?, ?, ?)",
                    (int(entry_id), f"admin_add by {payload.get('u','')}", "", content),
                )
                conn.commit()
                conn.close()
            except Exception:
                pass

            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok", "data": {"id": entry_id}}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def admin_kb_update(request: web.Request) -> web.Response:
        try:
            payload = _require_admin(request)
            if not payload:
                return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))

            params = await request.json()
            entry_id = int(params.get("id", 0))
            content = str(params.get("content", "")).strip()
            reason = str(params.get("reason", "admin_update")).strip()
            if entry_id <= 0 or not content:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "id/content required"}))

            state.kb.update_knowledge_entry(entry_id, content, update_reason=f"{reason} by {payload.get('u','')}")
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok"}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def admin_kb_delete(request: web.Request) -> web.Response:
        try:
            payload = _require_admin(request)
            if not payload:
                return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))

            params = await request.json()
            entry_id = int(params.get("id", 0))
            if entry_id <= 0:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "id required"}))

            conn = sqlite3.connect(state.kb.db_path)
            cur = conn.cursor()
            row = cur.execute("SELECT content FROM knowledge_entries WHERE id=?", (entry_id,)).fetchone()
            old_content = row[0] if row else ""
            cur.execute("DELETE FROM knowledge_entries WHERE id=?", (entry_id,))
            conn.commit()
            conn.close()

            try:
                conn = sqlite3.connect(state.kb.db_path)
                conn.execute(
                    "INSERT INTO update_logs(entry_id, update_type, old_content, new_content) VALUES (?, ?, ?, ?)",
                    (int(entry_id), f"admin_delete by {payload.get('u','')}", old_content, ""),
                )
                conn.commit()
                conn.close()
            except Exception:
                pass

            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok"}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def admin_kb_export(request: web.Request) -> web.Response:
        try:
            payload = _require_admin(request)
            if not payload:
                return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))
            conn = sqlite3.connect(state.kb.db_path)
            cur = conn.cursor()
            rows = cur.execute(
                "SELECT id, category, title, content, source, credibility_score, last_updated, expiration_date, tags FROM knowledge_entries ORDER BY id ASC"
            ).fetchall()
            conn.close()
            out = []
            for r in rows:
                out.append(
                    {
                        "id": r[0],
                        "category": r[1],
                        "title": r[2],
                        "content": r[3],
                        "source": r[4],
                        "credibility_score": r[5],
                        "last_updated": r[6],
                        "expiration_date": r[7],
                        "tags": json.loads(r[8]) if r[8] else [],
                    }
                )
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok", "data": out}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def admin_kb_import(request: web.Request) -> web.Response:
        try:
            payload = _require_admin(request)
            if not payload:
                return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))
            params = await request.json()
            items = params.get("items", [])
            if not isinstance(items, list):
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "items must be list"}))
            imported = 0
            for it in items:
                if not isinstance(it, dict):
                    continue
                title = str(it.get("title", "")).strip()
                content = str(it.get("content", "")).strip()
                if not title or not content:
                    continue
                state.kb.add_knowledge_entry(
                    category=str(it.get("category", "")).strip() or "未分类",
                    title=title,
                    content=content,
                    source=str(it.get("source", "")).strip() or "用户贡献",
                    tags=it.get("tags", []) if isinstance(it.get("tags", []), list) else [],
                    expiration_days=int(it.get("expiration_days", 365)),
                )
                imported += 1
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok", "data": {"imported": imported}}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def set_avatar_identity(request: web.Request) -> web.Response:
        try:
            # Admin-only: protect avatar identity persistence
            auth_header = request.headers.get("Authorization")
            token = get_bearer_token_from_auth_header(auth_header)
            payload = decode_token(token, state.auth_token_secret)
            if not payload:
                return web.Response(
                    content_type="application/json",
                    text=json.dumps({"code": -20, "msg": "unauthorized"}),
                )
            role = payload.get("role")
            if role != "admin":
                return web.Response(
                    content_type="application/json",
                    text=json.dumps({"code": -21, "msg": "forbidden"}),
                )

            params = await request.json()
            avatar_id = str(params.get("avatar_id", "")).strip()
            identity = str(params.get("identity", ""))
            sessionid = params.get("sessionid", 0)

            if not avatar_id:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "avatar_id is required"}))

            state.avatar_identities[avatar_id] = identity
            ok = state.save_avatar_identities()

            # 同步到当前会话（如果提供）
            try:
                if sessionid:
                    state.identities[int(sessionid)] = identity
            except Exception:
                pass

            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": 0 if ok else -1, "msg": "ok" if ok else "save failed"}),
            )
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def clear_avatar_identity(request: web.Request) -> web.Response:
        try:
            # Admin-only: protect avatar identity persistence
            auth_header = request.headers.get("Authorization")
            token = get_bearer_token_from_auth_header(auth_header)
            payload = decode_token(token, state.auth_token_secret)
            if not payload:
                return web.Response(
                    content_type="application/json",
                    text=json.dumps({"code": -20, "msg": "unauthorized"}),
                )
            role = payload.get("role")
            if role != "admin":
                return web.Response(
                    content_type="application/json",
                    text=json.dumps({"code": -21, "msg": "forbidden"}),
                )

            params = await request.json()
            avatar_id = str(params.get("avatar_id", "")).strip()
            sessionid = params.get("sessionid", 0)
            if not avatar_id:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "avatar_id is required"}))

            if avatar_id in state.avatar_identities:
                del state.avatar_identities[avatar_id]
            ok = state.save_avatar_identities()

            try:
                if sessionid and int(sessionid) in state.identities:
                    del state.identities[int(sessionid)]
            except Exception:
                pass

            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": 0 if ok else -1, "msg": "ok" if ok else "save failed"}),
            )
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def clear_identity(request: web.Request) -> web.Response:
        try:
            params = await request.json()
            sessionid = params.get("sessionid", 0)
            if sessionid in state.identities:
                del state.identities[sessionid]
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok"}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def preload_avatar(request: web.Request) -> web.Response:
        try:
            params = await request.json()
            sessionid = params.get("sessionid", 0)
            avatar_id = params.get("avatar_id", "")
            logger.info(f"Preloading avatar {avatar_id} for session {sessionid}")

            if not avatar_id:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "avatar_id is required"}))

            if not state.avatar_manager.preload_enabled:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "preload disabled by server config"}))

            state.preload_queue.append((sessionid, avatar_id))
            asyncio.create_task(process_preload_queue())
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "preload started"}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def process_preload_queue():
        if state.preload_in_progress or not state.preload_queue:
            return
        state.preload_in_progress = True
        try:
            preload_queue_size = state.get_config_value("avatar.preload_queue_size", 10)
            processing_count = 0
            while state.preload_queue and processing_count < preload_queue_size:
                _, avatar_id = state.preload_queue.pop(0)
                await asyncio.get_event_loop().run_in_executor(None, state.avatar_manager.preload_avatar_resources, avatar_id)
                processing_count += 1
        except Exception as e:
            logger.exception(f"预加载队列处理错误: {e}")
        finally:
            state.preload_in_progress = False

    async def set_preload_status(request: web.Request) -> web.Response:
        try:
            params = await request.json()
            enabled = params.get("enabled", False)
            state.avatar_manager.preload_enabled = bool(enabled)
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": f"preload {'enabled' if enabled else 'disabled'}"}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def get_preload_status(request: web.Request) -> web.Response:
        try:
            cache_size = len(state.avatar_manager.avatar_cache)
            queue_size = len(state.preload_queue)
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {
                        "code": 0,
                        "data": {
                            "enabled": bool(state.avatar_manager.preload_enabled),
                            "cache_size": cache_size,
                            "queue_size": queue_size,
                            "cached_avatars": list(state.avatar_manager.avatar_cache.keys()),
                        },
                    }
                ),
            )
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def clear_cache(request: web.Request) -> web.Response:
        try:
            cache_size = len(state.avatar_manager.avatar_cache)
            state.avatar_manager.avatar_cache.clear()
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": f"cache cleared, removed {cache_size} items"}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def admin_login(request: web.Request) -> web.Response:
        try:
            params = await request.json()
            username = str(params.get("username", "")).strip()
            password = str(params.get("password", ""))
            if not username or not password:
                return web.Response(content_type="application/json", text=json.dumps({"code": -22, "msg": "username/password required"}))

            if not state.auth_store.verify_login(username, password, expected_role="admin"):
                return web.Response(content_type="application/json", text=json.dumps({"code": -23, "msg": "账号 / 密码错误"}))

            token = create_token(
                {"u": username, "role": "admin"},
                token_secret=state.auth_token_secret,
                ttl_seconds=state.auth_token_ttl_seconds,
            )
            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": 0, "msg": "ok", "data": {"token": token, "role": "admin", "username": username}}),
            )
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def student_login(request: web.Request) -> web.Response:
        try:
            params = await request.json()
            username = str(params.get("username", "")).strip()
            password = str(params.get("password", ""))
            if not username or not password:
                return web.Response(content_type="application/json", text=json.dumps({"code": -22, "msg": "username/password required"}))

            if not state.auth_store.verify_login(username, password, expected_role="student"):
                return web.Response(content_type="application/json", text=json.dumps({"code": -23, "msg": "账号 / 密码错误"}))

            token = create_token(
                {"u": username, "role": "student"},
                token_secret=state.auth_token_secret,
                ttl_seconds=state.auth_token_ttl_seconds,
            )
            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": 0, "msg": "ok", "data": {"token": token, "role": "student", "username": username}}),
            )
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def student_register(request: web.Request) -> web.Response:
        try:
            params = await request.json()
            student_id = str(params.get("student_id", "")).strip()
            real_name = str(params.get("real_name", "")).strip()
            gender = str(params.get("gender", "")).strip()
            college = str(params.get("college", "")).strip()
            major = str(params.get("major", "")).strip()
            username = str(params.get("username", "")).strip()
            password = str(params.get("password", ""))
            if not student_id or not username or not password:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "学号/用户名/密码必填"}))
            if len(password) < 6:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "密码至少 6 位"}))
            ok = state.auth_store.student_register(
                student_id=student_id,
                real_name=real_name,
                gender=gender,
                college=college,
                major=major,
                username=username,
                password=password,
                enabled=True,
            )
            if not ok:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "注册失败：学号或用户名已存在"}))
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok"}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def admin_students_list(request: web.Request) -> web.Response:
        try:
            if not _require_admin(request):
                return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))
            q = request.query
            page = max(1, int(q.get("page", "1")))
            page_size = max(1, min(200, int(q.get("page_size", "20"))))
            student_id = str(q.get("student_id", "") or "")
            real_name = str(q.get("real_name", "") or "")
            college = str(q.get("college", "") or "")
            major = str(q.get("major", "") or "")
            status = str(q.get("status", "") or "")
            data = state.auth_store.list_students(
                student_id=student_id,
                real_name=real_name,
                college=college,
                major=major,
                status=status,
                page=page,
                page_size=page_size,
            )
            total = state.auth_store.count_students(
                student_id=student_id,
                real_name=real_name,
                college=college,
                major=major,
                status=status,
            )
            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": 0, "msg": "ok", "data": {"items": data, "total": total, "page": page, "page_size": page_size}}),
            )
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def admin_students_detail(request: web.Request) -> web.Response:
        try:
            if not _require_admin(request):
                return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))
            user_id = int(request.match_info.get("id", "0"))
            item = state.auth_store.get_student_by_id(user_id)
            if not item:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "not found"}))
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok", "data": item}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def admin_students_save(request: web.Request) -> web.Response:
        try:
            if not _require_admin(request):
                return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))
            p = await request.json()
            user_id = int(p.get("id", 0))
            student_id = str(p.get("student_id", "")).strip()
            real_name = str(p.get("real_name", "")).strip()
            gender = str(p.get("gender", "")).strip()
            college = str(p.get("college", "")).strip()
            major = str(p.get("major", "")).strip()
            username = str(p.get("username", "")).strip()
            enabled = bool(p.get("enabled", True))
            password = str(p.get("password", ""))

            if user_id > 0:
                ok = state.auth_store.update_student(user_id, student_id, real_name, gender, college, major, username, enabled)
                if not ok:
                    return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "保存失败：学号或用户名冲突"}))
            else:
                if not password:
                    alphabet = string.ascii_letters + string.digits
                    password = "".join(random.choice(alphabet) for _ in range(10))
                ok = state.auth_store.student_register(
                    student_id=student_id,
                    real_name=real_name,
                    gender=gender,
                    college=college,
                    major=major,
                    username=username,
                    password=password,
                    enabled=enabled,
                )
                if not ok:
                    return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "新增失败：学号或用户名冲突"}))
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok"}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def admin_students_delete(request: web.Request) -> web.Response:
        try:
            if not _require_admin(request):
                return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))
            p = await request.json()
            user_id = int(p.get("id", 0))
            if user_id <= 0:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "id required"}))
            state.auth_store.delete_student(user_id)
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok"}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def admin_students_enable(request: web.Request) -> web.Response:
        try:
            if not _require_admin(request):
                return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))
            p = await request.json()
            user_id = int(p.get("id", 0))
            enabled = bool(p.get("enabled", True))
            if user_id <= 0:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "id required"}))
            state.auth_store.set_user_enabled(user_id, enabled)
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok"}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def admin_students_reset_password(request: web.Request) -> web.Response:
        try:
            if not _require_admin(request):
                return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))
            p = await request.json()
            user_id = int(p.get("id", 0))
            if user_id <= 0:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "id required"}))
            alphabet = string.ascii_letters + string.digits
            new_password = "".join(random.choice(alphabet) for _ in range(10))
            ok = state.auth_store.reset_password(user_id, new_password)
            if not ok:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "not found"}))
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok", "data": {"new_password": new_password}}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def admin_students_batch_delete(request: web.Request) -> web.Response:
        try:
            if not _require_admin(request):
                return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))
            p = await request.json()
            ids = p.get("ids", []) or []
            if not isinstance(ids, list):
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "ids must be list"}))
            cnt = 0
            for sid in ids:
                try:
                    state.auth_store.delete_student(int(sid))
                    cnt += 1
                except Exception:
                    pass
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok", "data": {"deleted": cnt}}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def admin_students_batch_enable(request: web.Request) -> web.Response:
        try:
            if not _require_admin(request):
                return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))
            p = await request.json()
            ids = p.get("ids", []) or []
            enabled = bool(p.get("enabled", True))
            if not isinstance(ids, list):
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "ids must be list"}))
            cnt = 0
            for sid in ids:
                try:
                    state.auth_store.set_user_enabled(int(sid), enabled)
                    cnt += 1
                except Exception:
                    pass
            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": 0, "msg": "ok", "data": {"updated": cnt, "enabled": enabled}}),
            )
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def admin_students_batch_reset_password(request: web.Request) -> web.Response:
        try:
            if not _require_admin(request):
                return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))
            p = await request.json()
            ids = p.get("ids", []) or []
            if not isinstance(ids, list) or not ids:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "ids must be non-empty list"}))
            results = []
            for sid in ids:
                try:
                    user_id = int(sid)
                    item = state.auth_store.get_student_by_id(user_id)
                    if not item:
                        continue
                    alphabet = string.ascii_letters + string.digits
                    new_password = "".join(random.choice(alphabet) for _ in range(10))
                    ok = state.auth_store.reset_password(user_id, new_password)
                    if not ok:
                        continue
                    results.append(
                        {
                            "id": user_id,
                            "student_id": item.get("student_id", ""),
                            "real_name": item.get("real_name", ""),
                            "username": item.get("username", ""),
                            "new_password": new_password,
                        }
                    )
                except Exception:
                    pass
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok", "data": {"items": results, "count": len(results)}}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def admin_students_export(request: web.Request) -> web.Response:
        try:
            if not _require_admin(request):
                return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))
            data = state.auth_store.list_students()
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok", "data": data}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def admin_students_export_xlsx(request: web.Request) -> web.Response:
        try:
            if not _require_admin(request):
                return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))
            data = state.auth_store.list_students()
            try:
                import pandas as pd  # type: ignore
            except Exception:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "未安装 pandas/openpyxl，无法导出 xlsx"}))

            rows = []
            for it in data:
                rows.append(
                    {
                        "用户ID": it.get("id"),
                        "学号": it.get("student_id", ""),
                        "姓名": it.get("real_name", ""),
                        "性别": it.get("gender", ""),
                        "学院": it.get("college", ""),
                        "专业": it.get("major", ""),
                        "登录账号": it.get("username", ""),
                        "状态": it.get("status", ""),
                        "注册时间(ms)": it.get("created_at_ms"),
                        "最后登录(ms)": it.get("last_login_at_ms"),
                    }
                )
            df = pd.DataFrame(rows)
            bio = io.BytesIO()
            try:
                df.to_excel(bio, index=False)
            except Exception:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "导出 xlsx 失败，请确认安装 openpyxl"}))
            bio.seek(0)
            return web.Response(
                body=bio.read(),
                headers={
                    "Content-Type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    "Content-Disposition": "attachment; filename=students_export.xlsx",
                },
            )
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def admin_students_import(request: web.Request) -> web.Response:
        try:
            if not _require_admin(request):
                return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))
            p = await request.json()
            items = p.get("items", []) or []
            if not isinstance(items, list):
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "items must be list"}))
            imported = 0
            for it in items:
                if not isinstance(it, dict):
                    continue
                student_id = str(it.get("student_id", "")).strip()
                username = str(it.get("username", "")).strip()
                if not student_id or not username:
                    continue
                password = str(it.get("password", "")).strip()
                if not password:
                    alphabet = string.ascii_letters + string.digits
                    password = "".join(random.choice(alphabet) for _ in range(10))
                ok = state.auth_store.student_register(
                    student_id=student_id,
                    real_name=str(it.get("real_name", "")).strip(),
                    gender=str(it.get("gender", "")).strip(),
                    college=str(it.get("college", "")).strip(),
                    major=str(it.get("major", "")).strip(),
                    username=username,
                    password=password,
                    enabled=bool(it.get("enabled", True)),
                )
                if ok:
                    imported += 1
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok", "data": {"imported": imported}}))
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def admin_students_import_file(request: web.Request) -> web.Response:
        try:
            if not _require_admin(request):
                return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))
            form = await request.post()
            up = form.get("file")
            if up is None:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "file required"}))
            filename = str(getattr(up, "filename", "") or "").lower()
            raw = up.file.read()
            if not raw:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "empty file"}))

            required_cols = {"student_id", "real_name", "gender", "college", "major", "username", "password", "enabled"}
            rows: list[dict] = []
            if filename.endswith(".csv"):
                import csv

                text = raw.decode("utf-8-sig", errors="ignore")
                reader = csv.DictReader(io.StringIO(text))
                for r in reader:
                    rows.append(dict(r))
            elif filename.endswith(".xlsx"):
                try:
                    import pandas as pd  # type: ignore
                except Exception:
                    return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "未安装 pandas/openpyxl，暂不支持 xlsx"}))
                df = pd.read_excel(io.BytesIO(raw))
                rows = df.fillna("").to_dict(orient="records")
            else:
                return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": "仅支持 .csv / .xlsx"}))

            imported = 0
            skipped = 0
            for it in rows:
                if not isinstance(it, dict):
                    skipped += 1
                    continue
                keys_lower = {str(k).strip().lower(): v for k, v in it.items()}
                if not required_cols.issubset(set(keys_lower.keys())):
                    skipped += 1
                    continue
                student_id = str(keys_lower.get("student_id", "")).strip()
                username = str(keys_lower.get("username", "")).strip()
                if not student_id or not username:
                    skipped += 1
                    continue
                password = str(keys_lower.get("password", "")).strip()
                if not password:
                    alphabet = string.ascii_letters + string.digits
                    password = "".join(random.choice(alphabet) for _ in range(10))
                enabled_raw = str(keys_lower.get("enabled", "1")).strip().lower()
                enabled = enabled_raw in ("1", "true", "yes", "y", "正常", "启用")
                ok = state.auth_store.student_register(
                    student_id=student_id,
                    real_name=str(keys_lower.get("real_name", "")).strip(),
                    gender=str(keys_lower.get("gender", "")).strip(),
                    college=str(keys_lower.get("college", "")).strip(),
                    major=str(keys_lower.get("major", "")).strip(),
                    username=username,
                    password=password,
                    enabled=enabled,
                )
                if ok:
                    imported += 1
                else:
                    skipped += 1
            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": 0, "msg": "ok", "data": {"imported": imported, "skipped": skipped, "total": len(rows)}}),
            )
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    async def admin_students_template(request: web.Request) -> web.Response:
        try:
            if not _require_admin(request):
                return web.Response(content_type="application/json", text=json.dumps({"code": -21, "msg": "forbidden"}))
            csv_text = (
                "student_id,real_name,gender,college,major,username,password,enabled\n"
                "20260001,张三,男,信息工程学院,软件工程,stu_zhangsan,123456,1\n"
                "20260002,李四,女,外国语学院,英语,stu_lisi,123456,1\n"
            )
            return web.Response(
                body=csv_text.encode("utf-8-sig"),
                headers={
                    "Content-Type": "text/csv; charset=utf-8",
                    "Content-Disposition": "attachment; filename=students_template.csv",
                },
            )
        except Exception as e:
            logger.exception("exception:")
            return web.Response(content_type="application/json", text=json.dumps({"code": -1, "msg": str(e)}))

    # route registrations
    app.router.add_post("/admin/login", admin_login)
    app.router.add_post("/student/login", student_login)
    app.router.add_post("/student/register", student_register)
    app.router.add_get("/auth/me", auth_me)
    app.router.add_post("/auth/change_password", auth_change_password)
    app.router.add_get("/admin/students/list", admin_students_list)
    app.router.add_get("/admin/students/detail/{id}", admin_students_detail)
    app.router.add_post("/admin/students/save", admin_students_save)
    app.router.add_post("/admin/students/delete", admin_students_delete)
    app.router.add_post("/admin/students/enable", admin_students_enable)
    app.router.add_post("/admin/students/reset_password", admin_students_reset_password)
    app.router.add_post("/admin/students/batch_delete", admin_students_batch_delete)
    app.router.add_post("/admin/students/batch_enable", admin_students_batch_enable)
    app.router.add_post("/admin/students/batch_reset_password", admin_students_batch_reset_password)
    app.router.add_get("/admin/students/export", admin_students_export)
    app.router.add_get("/admin/students/export_xlsx", admin_students_export_xlsx)
    app.router.add_post("/admin/students/import", admin_students_import)
    app.router.add_post("/admin/students/import_file", admin_students_import_file)
    app.router.add_get("/admin/students/template", admin_students_template)
    app.router.add_get("/admin/sessions", admin_sessions_list)
    app.router.add_get("/admin/sessions/{id}", admin_sessions_detail)
    app.router.add_get("/my/sessions", my_sessions_list)
    app.router.add_get("/my/sessions/{id}", my_sessions_detail)
    app.router.add_post("/my/sessions/rate", my_sessions_rate)
    app.router.add_post("/my/sessions/delete", my_sessions_delete)
    app.router.add_post("/my/sessions/rename", my_sessions_rename)
    app.router.add_post("/my/sessions/end", my_sessions_end)
    app.router.add_post("/my/sessions/create", my_sessions_create)
    app.router.add_post("/admin/sessions/delete", admin_sessions_delete)
    app.router.add_get("/admin/analytics/overview", admin_analytics_overview)
    app.router.add_get("/admin/analytics/trend", admin_analytics_trend)
    app.router.add_get("/admin/analytics/response_time", admin_analytics_response_time)
    app.router.add_get("/admin/analytics/categories", admin_analytics_categories)
    app.router.add_get("/admin/analytics/tts_success", admin_analytics_tts_success)
    app.router.add_get("/admin/analytics/report", admin_analytics_report)
    app.router.add_get("/admin/analytics/satisfaction", admin_analytics_satisfaction)
    app.router.add_get("/admin/analytics/wordcloud", admin_analytics_wordcloud)
    app.router.add_get("/admin/analytics/performance", admin_analytics_performance)
    app.router.add_get("/admin/analytics/by_avatar", admin_analytics_by_avatar)
    app.router.add_get("/admin/avatar/list", admin_avatar_list)
    app.router.add_post("/admin/avatar/save", admin_avatar_save)
    app.router.add_post("/admin/avatar/delete", admin_avatar_delete)
    app.router.add_post("/admin/avatar/copy", admin_avatar_copy)
    app.router.add_get("/admin/avatar/actions", admin_avatar_actions_get)
    app.router.add_post("/admin/avatar/actions", admin_avatar_actions_save)
    app.router.add_get("/admin/avatar/tts", admin_avatar_tts_get)
    app.router.add_post("/admin/avatar/tts", admin_avatar_tts_save)
    app.router.add_post("/admin/avatar/tts/apply_all", admin_avatar_tts_apply_all)
    app.router.add_get("/avatar/tts", avatar_tts_get)
    app.router.add_get("/admin/tts/allowed_voices", admin_tts_allowed_voices_get)
    app.router.add_post("/admin/tts/allowed_voices", admin_tts_allowed_voices_save)
    app.router.add_get("/admin/kb/list", admin_kb_list)
    app.router.add_get("/admin/kb/detail/{id}", admin_kb_detail)
    app.router.add_post("/admin/kb/add", admin_kb_add)
    app.router.add_post("/admin/kb/update", admin_kb_update)
    app.router.add_post("/admin/kb/delete", admin_kb_delete)
    app.router.add_get("/admin/kb/export", admin_kb_export)
    app.router.add_post("/admin/kb/import", admin_kb_import)
    app.router.add_post("/offer", offer)
    app.router.add_post("/human", human)
    app.router.add_post("/humanaudio", humanaudio)
    app.router.add_post("/set_audiotype", set_audiotype)
    app.router.add_post("/set_tts_speed", set_tts_speed)
    app.router.add_post("/set_tts_voice", set_tts_voice)
    app.router.add_get("/tts/allowed_voices", tts_allowed_voices_get)
    app.router.add_post("/record", record)
    app.router.add_post("/interrupt_talk", interrupt_talk)
    app.router.add_post("/is_speaking", is_speaking)
    app.router.add_post("/switch_avatar", switch_avatar)
    app.router.add_post("/set_identity", set_identity)
    app.router.add_post("/clear_identity", clear_identity)
    app.router.add_get("/get_avatar_identity", get_avatar_identity)
    app.router.add_post("/set_avatar_identity", set_avatar_identity)
    app.router.add_post("/clear_avatar_identity", clear_avatar_identity)
    app.router.add_get("/get_avatars", get_avatars)
    app.router.add_post("/preload_avatar", preload_avatar)
    app.router.add_post("/set_preload_status", set_preload_status)
    app.router.add_get("/get_preload_status", get_preload_status)
    app.router.add_post("/clear_cache", clear_cache)


from __future__ import annotations

import asyncio
import json
import os
import torch
import time
import uuid

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceServer, RTCConfiguration
from aiortc.rtcrtpsender import RTCRtpSender

from logger import logger
from webrtc import HumanPlayer

from livetalking.server.state import AppState
from livetalking.services.chat_service import llm_response_with_identity


def setup_routes(app: web.Application, state: AppState) -> None:
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

                def is_current():
                    return state.chat_gen_ids.get(sessionid) == gen_id

                def _run_llm_job():
                    try:
                        t_job0 = time.perf_counter()
                        logger.info(f"[trace={trace_id}] chat start session={sessionid} len={len(user_text)}")
                        llm_response_with_identity(
                            user_text,
                            nerfreal,
                            identity,
                            is_current,
                            rag_evidence=rag_evidence,
                        )
                        logger.info(f"[trace={trace_id}] chat done dt={(time.perf_counter() - t_job0):.3f}s")
                    except Exception as e:
                        logger.exception(f"[trace={trace_id}] chat error: {e}")

                # Run and wait for LLM completion up to timeout; if exceeded, do degradation.
                try:
                    future = asyncio.get_event_loop().run_in_executor(None, _run_llm_job)
                    await asyncio.wait_for(future, timeout=llm_timeout_s)
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
                    return web.Response(
                        content_type="application/json",
                        text=json.dumps(
                            {
                                "code": -3,
                                "msg": "llm timeout, fallback spoken",
                                "trace_id": trace_id,
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
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {
                        "code": 0,
                        "msg": "ok",
                        "trace_id": trace_id,
                        "dt_ms": int(dt * 1000),
                        "rag_evidence": rag_evidence_for_resp,
                    }
                ),
            )
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
                            avatar_type = state.avatar_manager.infer_model_type(item, getattr(state.opt, "model", "musetalk"))
                            avatar_list.append({"id": item, "name": item, "type": avatar_type})
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

    async def set_avatar_identity(request: web.Request) -> web.Response:
        try:
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

    # route registrations
    app.router.add_post("/offer", offer)
    app.router.add_post("/human", human)
    app.router.add_post("/humanaudio", humanaudio)
    app.router.add_post("/set_audiotype", set_audiotype)
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


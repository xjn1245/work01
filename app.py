###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku@foxmail.com
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################

"""
LiveTalking 主应用文件

该文件是 LiveTalking 项目的主入口，负责初始化和运行数字人服务，处理 WebRTC 连接，
提供 HTTP API 接口，并管理数字人会话。
"""

# server.py
from flask import Flask, render_template, send_from_directory, request, jsonify
# from flask_sockets import Sockets
import base64
import json
# import gevent
# from gevent import pywsgi
# from geventwebsocket.handler import WebSocketHandler
import re
import copy
import numpy as np
from threading import Thread, Event
# import multiprocessing
import torch.multiprocessing as mp

from aiohttp import web
import aiohttp
import aiohttp_cors
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceServer, RTCConfiguration
from aiortc.rtcrtpsender import RTCRtpSender
from webrtc import HumanPlayer
from basereal import BaseReal
from llm import llm_response


def llm_response_with_identity(message, nerfreal, identity=None):
    """
    支持身份信息的LLM响应函数 - 性能优化版本

    Args:
        message: 用户消息
        nerfreal: 数字人实例
        identity: 身份信息
    """
    import time
    from logger import logger

    # 获取性能优化配置
    perf_config = get_performance_config()

    start_total = time.perf_counter()
    logger.info(f"开始处理用户消息: {message[:50]}... (优化级别: {perf_config.level})")

    # 应用性能优化配置
    llm_config = optimize_llm_response(perf_config)
    tts_config = optimize_tts_config(perf_config)

    logger.info(f"TTS引擎: {tts_config['engine']}, LLM模型: {llm_config['model']}")

    # 使用优化后的LLM配置
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    init_end = time.perf_counter()
    logger.info(f"LLM初始化时间: {init_end - start_total:.3f}s")

    # 构建系统消息，如果提供了身份信息则使用，否则使用默认身份
    if identity and identity.strip():
        system_message = identity.strip()
        logger.info(f"使用预设身份: {system_message[:100]}...")
    else:
        system_message = '您是一位专业的留学顾问，拥有丰富的留学咨询经验，擅长解答留学申请、院校选择、专业规划等问题。'

    # 使用性能优化配置
    completion = client.chat.completions.create(
        model=llm_config["model"],
        messages=[{'role': 'system', 'content': system_message},
                  {'role': 'user', 'content': message}],
        stream=True,
        max_tokens=llm_config["max_tokens"],
        temperature=llm_config["temperature"],
        stream_options={"include_usage": True}
    )

    result = ""
    first = True
    chunk_count = 0
    start_llm = time.perf_counter()

    for chunk in completion:
        chunk_count += 1
        if len(chunk.choices) > 0 and chunk.choices[0].delta.content:
            if first:
                first_chunk_end = time.perf_counter()
                logger.info(f"LLM首字节时间: {first_chunk_end - start_llm:.3f}s")
                first = False

            msg = chunk.choices[0].delta.content
            result += msg

            # 优化5: 使用性能配置的智能文本分割
            chunk_size = llm_config.get("text_chunk_size", 30)
            enable_smart_seg = llm_config.get("enable_smart_segmentation", True)

            # 智能文本分割策略
            should_send = False

            if enable_smart_seg:
                # 智能分割：基于长度和自然断句
                if len(result) >= chunk_size:
                    # 检查是否在自然断句处
                    if any(punct in result for punct in "，。！？；"):
                        should_send = True
                    # 如果达到两倍chunk_size，强制发送避免过长等待
                    elif len(result) >= chunk_size * 2:
                        should_send = True
            else:
                # 简单分割：仅基于长度
                if len(result) >= chunk_size:
                    should_send = True

            if should_send:
                nerfreal.put_msg_txt(result)
                result = ""

    # 发送剩余内容
    if result:
        nerfreal.put_msg_txt(result)

    llm_end = time.perf_counter()
    total_time = llm_end - start_total
    logger.info(f"LLM处理完成 - 总时间: {total_time:.3f}s, 处理chunks: {chunk_count}")


import argparse
import random
import shutil
import asyncio
import torch
import time
from typing import Dict, List, Tuple
from logger import logger
import gc
from performance_config import get_performance_config, optimize_llm_response, optimize_tts_config
from config_loader import get_config, get_config_value
from livetalking.services.avatar_manager import AvatarManager
from livetalking.server.state import AppState
from livetalking.server.routes import setup_routes
from knowledge_base import StudyAbroadKnowledgeBase
from livetalking.server.auth_store import AuthStore
from livetalking.server.chat_history import ChatHistoryStore
from livetalking.server.avatar_admin_store import AvatarAdminStore

app = Flask(__name__)
# sockets = Sockets(app)
nerfreals: Dict[int, BaseReal] = {}  # 存储会话ID到数字人实例的映射
identities: Dict[int, str] = {}  # 存储会话ID到身份信息的映射
opt = None  # 命令行参数
model = None  # 数字人模型
avatar = None  # 数字人形象

# 预加载/资源管理（模块化）
avatar_manager = None  # type: ignore[assignment]
preload_queue: List[Tuple[int, str]] = []  # 预加载队列
preload_in_progress = False  # 预加载是否正在进行

##### WebRTC 相关功能 ###############################
pcs = set()  # 存储所有活动的 PeerConnection


def randN(N) -> int:
    """
    生成长度为 N 的随机数

    Args:
        N: 随机数的位数

    Returns:
        生成的随机数
    """
    min_val = pow(10, N - 1)
    max_val = pow(10, N)
    return random.randint(min_val, max_val - 1)


def infer_model_type_by_avatar_id(avatar_id: str, fallback_model: str) -> str:
    """
    根据 avatar 目录文件存在情况推断数字人实现类型。

    兼容不同形象模型混用的场景：wav2lip / musetalk / ultralight
    """
    if not avatar_id:
        return fallback_model

    import os

    avatar_path = f"./data/avatars/{avatar_id}"
    # ultralight
    if os.path.exists(os.path.join(avatar_path, "ultralight.pth")):
        return "ultralight"

    # musetalk（musetalk 需要 latents.pt / mask_coords.pkl 等）
    if os.path.exists(os.path.join(avatar_path, "latents.pt")) and os.path.exists(
        os.path.join(avatar_path, "mask_coords.pkl")
    ):
        return "musetalk"

    # wav2lip（wav2lip 只需要 coords.pkl + face_imgs）
    face_imgs_dir = os.path.join(avatar_path, "face_imgs")
    coords_path = os.path.join(avatar_path, "coords.pkl")
    if os.path.exists(face_imgs_dir) and os.path.exists(coords_path):
        return "wav2lip"

    # 兜底：如果用户仍希望按 avatar_id 名称做推断，则走兼容逻辑
    if get_config_value('avatar.model_infer_from_id', True):
        aid = (avatar_id or "").lower()
        if "wav2lip" in aid:
            return "wav2lip"
        if "ultralight" in aid:
            return "ultralight"

    return fallback_model


def preload_avatar_resources(avatar_id: str) -> Dict:
    """
    预加载头像和模型资源

    Args:
        avatar_id: 头像ID

    Returns:
        Dict: 包含模型和头像的字典
    """
    global avatar_manager
    if avatar_manager is None:
        raise RuntimeError("avatar_manager is not initialized")
    res = avatar_manager.preload_avatar_resources(avatar_id)
    return None if res is None else {"model": res.model, "avatar": res.avatar, "loaded_at": res.loaded_at}


def build_nerfreal(sessionid: int, avatar_id=None) -> BaseReal:
    """
    根据指定的模型类型创建数字人实例

    Args:
        sessionid: 会话ID
        avatar_id: 头像ID，如果为None则使用opt.avatar_id

    Returns:
        BaseReal: 数字人实例
    """
    global avatar_manager
    if avatar_manager is None:
        raise RuntimeError("avatar_manager is not initialized")
    return avatar_manager.build_nerfreal(sessionid, avatar_id)


async def process_preload_queue():
    """
    处理预加载队列
    """
    global preload_in_progress, preload_queue

    if preload_in_progress or not preload_queue:
        return

    preload_in_progress = True

    try:
        # 从配置文件获取队列大小限制
        preload_queue_size = get_config_value('avatar.preload_queue_size', 10)

        # 限制队列处理数量
        processing_count = 0
        while preload_queue and processing_count < preload_queue_size:
            sessionid, avatar_id = preload_queue.pop(0)
            logger.info(f"处理预加载请求: session={sessionid}, avatar={avatar_id}")

            # 在后台线程中预加载资源
            await asyncio.get_event_loop().run_in_executor(
                None, preload_avatar_resources, avatar_id
            )

            processing_count += 1

            # 检查缓存大小并清理
            await check_and_clean_cache()

    except Exception as e:
        logger.exception(f"预加载队列处理错误: {e}")
    finally:
        preload_in_progress = False


async def check_and_clean_cache():
    """
    检查并清理缓存，确保不超过配置的最大大小
    """
    global avatar_manager
    if avatar_manager is None:
        return

    # 如果用户开启“启动时预加载全部”，则不做缓存淘汰
    if avatar_manager.preload_all_on_start:
        return

    cache_max_size = get_config_value('avatar.cache_max_size', 5)

    if len(avatar_manager.avatar_cache) > cache_max_size:
        # 按加载时间排序，删除最旧的缓存
        sorted_cache = sorted(avatar_manager.avatar_cache.items(), key=lambda x: x[1].loaded_at)
        items_to_remove = len(avatar_manager.avatar_cache) - cache_max_size

        for i in range(items_to_remove):
            if sorted_cache[i][0] in avatar_manager.avatar_cache:
                del avatar_manager.avatar_cache[sorted_cache[i][0]]
                logger.info(f"清理缓存: {sorted_cache[i][0]}")

        # 清理GPU资源
        import gc
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


# NOTE: 以下 aiohttp handlers 已迁移到 `livetalking/server/routes.py`。
# 这里暂时保留旧实现，避免一次性大删改造成冲突；实际运行已不再注册这些 handler。
async def offer(request):
    """
    处理 WebRTC 连接请求

    Args:
        request: HTTP 请求对象

    Returns:
        web.Response: 包含 SDP 响应和会话 ID 的 JSON 响应
    """
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    # 正常连接模式：只有点击开始连接时才建立连接
    # 切换数字人时不会建立连接，因此不需要自动清理

    # 生成会话 ID 并创建数字人实例
    sessionid = randN(6)  # 使用随机数作为会话ID
    nerfreals[sessionid] = None
    logger.info('sessionid=%d, session num=%d', sessionid, len(nerfreals))

    # 检查请求中是否包含avatar_id参数，如果没有则使用默认值
    avatar_id = params.get('avatar_id', opt.avatar_id)
    logger.info(f'Creating nerfreal instance with avatar_id: {avatar_id}')

    # 在后台线程中创建数字人实例，传入avatar_id
    nerfreal = await asyncio.get_event_loop().run_in_executor(None, build_nerfreal, sessionid, avatar_id)
    nerfreals[sessionid] = nerfreal

    # 配置 ICE 服务器
    # ice_server = RTCIceServer(urls='stun:stun.l.google.com:19302')
    ice_server = RTCIceServer(urls='stun:stun.miwifi.com:3478')
    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[ice_server]))
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        """
        处理连接状态变化
        """
        logger.info("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)
            del nerfreals[sessionid]
        if pc.connectionState == "closed":
            pcs.discard(pc)
            del nerfreals[sessionid]
            # gc.collect()

    # 创建 HumanPlayer 并添加音视频轨道
    player = HumanPlayer(nerfreals[sessionid])
    audio_sender = pc.addTrack(player.audio)
    video_sender = pc.addTrack(player.video)

    # 配置视频编解码器偏好
    capabilities = RTCRtpSender.getCapabilities("video")
    preferences = list(filter(lambda x: x.name == "H264", capabilities.codecs))
    preferences += list(filter(lambda x: x.name == "VP8", capabilities.codecs))
    preferences += list(filter(lambda x: x.name == "rtx", capabilities.codecs))
    transceiver = pc.getTransceivers()[1]
    transceiver.setCodecPreferences(preferences)

    # 设置远程描述并创建响应
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "sessionid": sessionid}
        ),
    )


async def human(request):
    """
    处理文本输入请求

    Args:
        request: HTTP 请求对象

    Returns:
        web.Response: 操作结果的 JSON 响应
    """
    try:
        params = await request.json()

        sessionid = params.get('sessionid', 0)

        # 处理打断请求
        if params.get('interrupt'):
            nerfreals[sessionid].flush_talk()

        # 处理不同类型的请求
        if params['type'] == 'echo':
            # 直接播报文本
            nerfreals[sessionid].put_msg_txt(params['text'])
        elif params['type'] == 'chat':
            # 通过 LLM 处理文本并播报，使用预设的身份信息
            identity = identities.get(sessionid)
            asyncio.get_event_loop().run_in_executor(None, llm_response_with_identity, params['text'],
                                                     nerfreals[sessionid], identity)

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg": "ok"}
            ),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg": str(e)}
            ),
        )


async def interrupt_talk(request):
    """
    处理打断数字人说话的请求

    Args:
        request: HTTP 请求对象

    Returns:
        web.Response: 操作结果的 JSON 响应
    """
    try:
        params = await request.json()
        sessionid = params.get('sessionid', 0)
        nerfreals[sessionid].flush_talk()

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg": "ok"}
            ),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg": str(e)}
            ),
        )


async def humanaudio(request):
    """
    处理音频文件输入请求

    Args:
        request: HTTP 请求对象

    Returns:
        web.Response: 操作结果的 JSON 响应
    """
    try:
        form = await request.post()
        sessionid = int(form.get('sessionid', 0))
        fileobj = form["file"]
        # filename = fileobj.filename
        filebytes = fileobj.file.read()

        # 将音频文件传递给数字人实例
        nerfreals[sessionid].put_audio_file(filebytes)

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg": "ok"}
            ),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg": str(e)}
            ),
        )


async def set_audiotype(request):
    """
    设置音频类型

    Args:
        request: HTTP 请求对象

    Returns:
        web.Response: 操作结果的 JSON 响应
    """
    try:
        params = await request.json()
        sessionid = params.get('sessionid', 0)
        nerfreals[sessionid].set_custom_state(params['audiotype'], params['reinit'])

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg": "ok"}
            ),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg": str(e)}
            ),
        )


async def record(request):
    """
    处理录制请求

    Args:
        request: HTTP 请求对象

    Returns:
        web.Response: 操作结果的 JSON 响应
    """
    try:
        params = await request.json()
        sessionid = params.get('sessionid', 0)

        if params['type'] == 'start_record':
            # 开始录制
            nerfreals[sessionid].start_recording()
        elif params['type'] == 'end_record':
            # 结束录制
            nerfreals[sessionid].stop_recording()

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg": "ok"}
            ),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg": str(e)}
            ),
        )


async def is_speaking(request):
    """
    检查数字人是否正在说话

    Args:
        request: HTTP 请求对象

    Returns:
        web.Response: 包含说话状态的 JSON 响应
    """
    params = await request.json()
    sessionid = params.get('sessionid', 0)
    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"code": 0, "data": nerfreals[sessionid].is_speaking()}
        ),
    )


async def get_avatars(request):
    """
    获取可用的数字人形象列表

    Args:
        request: HTTP 请求对象

    Returns:
        web.Response: 包含数字人形象列表的 JSON 响应
    """
    try:
        import os
        avatars_dir = "data/avatars"
        avatar_list = []

        # 扫描 avatars 目录
        if os.path.exists(avatars_dir):
            logger.info(f'Scanning avatars directory: {avatars_dir}')
            for item in os.listdir(avatars_dir):
                item_path = os.path.join(avatars_dir, item)
                logger.info(f'Checking item: {item}, is_dir: {os.path.isdir(item_path)}')
                if os.path.isdir(item_path) and item != ".gitkeep":
                    # 检查是否包含必要的文件
                    face_imgs_path = os.path.join(item_path, "face_imgs")
                    coords_path = os.path.join(item_path, "coords.pkl")

                    logger.info(
                        f'Checking files for {item}: face_imgs exists: {os.path.exists(face_imgs_path)}, coords exists: {os.path.exists(coords_path)}')

                    if os.path.exists(face_imgs_path) and os.path.exists(coords_path):
                        # 根据目录文件推断实现类型
                        avatar_type = avatar_manager.infer_model_type(item, getattr(opt, "model", "musetalk")) if avatar_manager else getattr(opt, "model", "musetalk")
                        avatar_list.append({
                            "id": item,
                            "name": item,
                            "type": avatar_type
                        })
                        logger.info(f'Added avatar: {item}')

        logger.info(f'Found {len(avatar_list)} available avatars: {[avatar["id"] for avatar in avatar_list]}')

        return web.Response(
            content_type="application/json",
            text=json.dumps({
                "code": 0,
                "msg": "ok",
                "data": avatar_list
            }),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            text=json.dumps({
                "code": -1,
                "msg": str(e)
            }),
        )


async def switch_avatar(request):
    """
    切换数字人形象

    Args:
        request: HTTP 请求对象

    Returns:
        web.Response: 操作结果的 JSON 响应
    """
    try:
        params = await request.json()
        sessionid = params.get('sessionid', 0)
        avatar_id = params.get('avatar_id', 'wav2lip256_avatar1')

        logger.info(f'Switching avatar to {avatar_id} for session {sessionid}')

        # 保存旧的数字人实例
        old_nerfreal = nerfreals.get(sessionid)
        if old_nerfreal:
            logger.info(f'Found old nerfreal instance for session {sessionid}')
            # 完全清理旧实例
            old_nerfreal.flush_talk()

            # 强制清理GPU资源
            if hasattr(old_nerfreal, 'asr') and old_nerfreal.asr:
                if hasattr(old_nerfreal.asr, 'flush_talk'):
                    old_nerfreal.asr.flush_talk()

            # 清理模型引用
            if hasattr(old_nerfreal, 'model'):
                old_nerfreal.model = None
            if hasattr(old_nerfreal, 'avatar'):
                old_nerfreal.avatar = None

            # 强制垃圾回收
            del old_nerfreal
            import gc
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            logger.info(f'Old nerfreal instance for session {sessionid} fully cleaned')
        else:
            logger.info(f'No old nerfreal instance found for session {sessionid}')

        # 创建新的数字人实例，直接传入avatar_id
        logger.info(f'Building new nerfreal instance with avatar_id: {avatar_id}')
        new_nerfreal = await asyncio.get_event_loop().run_in_executor(None, build_nerfreal, sessionid, avatar_id)
        nerfreals[sessionid] = new_nerfreal

        logger.info(
            f'Successfully switched to {avatar_id} avatar for session {sessionid}. New instance created: {new_nerfreal is not None}')

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg": "ok"}
            ),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg": str(e)}
            ),
        )


async def set_identity(request):
    """
    设置数字人身份信息

    Args:
        request: HTTP 请求对象

    Returns:
        web.Response: 操作结果的 JSON 响应
    """
    try:
        params = await request.json()
        sessionid = params.get('sessionid', 0)
        identity = params.get('identity', '')

        logger.info(f'Setting identity for session {sessionid}')

        # 保存身份信息
        identities[sessionid] = identity

        logger.info(f'Successfully set identity for session {sessionid}. Identity length: {len(identity)}')

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg": "ok"}
            ),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg": str(e)}
            ),
        )


async def clear_identity(request):
    """
    清除数字人身份信息

    Args:
        request: HTTP 请求对象

    Returns:
        web.Response: 操作结果的 JSON 响应
    """
    try:
        params = await request.json()
        sessionid = params.get('sessionid', 0)

        logger.info(f'Clearing identity for session {sessionid}')

        # 清除身份信息
        if sessionid in identities:
            del identities[sessionid]

        logger.info(f'Successfully cleared identity for session {sessionid}')

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg": "ok"}
            ),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg": str(e)}
            ),
        )


async def preload_avatar(request):
    """
    预加载数字人头像资源

    Args:
        request: HTTP 请求对象

    Returns:
        web.Response: 操作结果的 JSON 响应
    """
    try:
        params = await request.json()
        sessionid = params.get('sessionid', 0)
        avatar_id = params.get('avatar_id', '')

        logger.info(f'Preloading avatar {avatar_id} for session {sessionid}')

        if not avatar_id:
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {"code": -1, "msg": "avatar_id is required"}
                ),
            )

        # 如果当前未启用“预加载模式”，拒绝执行预加载，避免重复加载浪费资源
        global avatar_manager
        if avatar_manager is None or not avatar_manager.preload_enabled:
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {"code": -1, "msg": "preload disabled by server config"}
                ),
            )

        # 添加到预加载队列
        global preload_queue
        preload_queue.append((sessionid, avatar_id))

        # 启动预加载处理
        asyncio.create_task(process_preload_queue())

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg": "preload started"}
            ),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg": str(e)}
            ),
        )


async def set_preload_status(request):
    """
    设置预加载功能状态

    Args:
        request: HTTP 请求对象

    Returns:
        web.Response: 操作结果的 JSON 响应
    """
    try:
        params = await request.json()
        enabled = params.get('enabled', False)

        global avatar_manager
        if avatar_manager is not None:
            avatar_manager.preload_enabled = enabled

        logger.info(f'Set preload status to {enabled}')

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg": f"preload {'enabled' if enabled else 'disabled'}"}
            ),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg": str(e)}
            ),
        )


async def get_preload_status(request):
    """
    获取预加载功能状态

    Args:
        request: HTTP 请求对象

    Returns:
        web.Response: 包含预加载状态的 JSON 响应
    """
    try:
        global avatar_manager, preload_queue

        # 计算缓存大小
        cache_size = len(avatar_manager.avatar_cache) if avatar_manager is not None else 0
        queue_size = len(preload_queue)

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {
                    "code": 0,
                    "data": {
                        "enabled": bool(avatar_manager.preload_enabled) if avatar_manager is not None else False,
                        "cache_size": cache_size,
                        "queue_size": queue_size,
                        "cached_avatars": list(avatar_manager.avatar_cache.keys()) if avatar_manager is not None else []
                    }
                }
            ),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg": str(e)}
            ),
        )


async def clear_cache(request):
    """
    清除预加载缓存

    Args:
        request: HTTP 请求对象

    Returns:
        web.Response: 操作结果的 JSON 响应
    """
    try:
        global avatar_manager

        # 清除缓存
        cache_size = len(avatar_manager.avatar_cache) if avatar_manager is not None else 0
        if avatar_manager is not None:
            avatar_manager.avatar_cache.clear()

        # 清理GPU资源
        import gc
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        logger.info(f'Cleared cache, removed {cache_size} items')

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg": f"cache cleared, removed {cache_size} items"}
            ),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg": str(e)}
            ),
        )


async def on_shutdown(app):
    """
    应用关闭时的清理操作

    Args:
        app: 应用实例
    """
    # 关闭所有 PeerConnection
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


async def post(url, data):
    """
    发送 POST 请求

    Args:
        url: 请求 URL
        data: 请求数据

    Returns:
        str: 响应文本
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data) as response:
                return await response.text()
    except aiohttp.ClientError as e:
        logger.info(f'Error: {e}')


async def run(push_url, sessionid):
    """
    运行 RTCPUSH 模式的数字人服务

    Args:
        push_url: 推送 URL
        sessionid: 会话 ID
    """
    # 创建数字人实例
    nerfreal = await asyncio.get_event_loop().run_in_executor(None, build_nerfreal, sessionid)
    nerfreals[sessionid] = nerfreal

    # 创建 PeerConnection
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        """
        处理连接状态变化
        """
        logger.info("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    # 创建 HumanPlayer 并添加音视频轨道
    player = HumanPlayer(nerfreals[sessionid])
    audio_sender = pc.addTrack(player.audio)
    video_sender = pc.addTrack(player.video)

    # 创建 offer 并发送
    await pc.setLocalDescription(await pc.createOffer())
    answer = await post(push_url, pc.localDescription.sdp)
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer, type='answer'))


##########################################
# os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# os.environ['MULTIPROCESSING_METHOD'] = 'forkserver'
if __name__ == '__main__':
    """
    主函数，初始化并启动数字人服务
    """
    # 设置多进程启动方法
    mp.set_start_method('spawn')

    # 解析命令行参数
    parser = argparse.ArgumentParser()

    # 音频 FPS
    parser.add_argument('--fps', type=int, default=get_config_value('model.fps', 50), help="audio fps,must be 50")
    # 滑动窗口左-中-右长度 (单位: 20ms)
    parser.add_argument('-l', type=int, default=get_config_value('sliding_window.left', 10))
    parser.add_argument('-m', type=int, default=get_config_value('sliding_window.middle', 8))
    parser.add_argument('-r', type=int, default=get_config_value('sliding_window.right', 10))

    # GUI 尺寸
    parser.add_argument('--W', type=int, default=get_config_value('gui.width', 450), help="GUI width")
    parser.add_argument('--H', type=int, default=get_config_value('gui.height', 450), help="GUI height")

    # musetalk 选项
    parser.add_argument('--avatar_id', type=str, default=get_config_value('avatar.default_avatar_id', 'avator_1'),
                        help="define which avatar in data/avatars")
    # parser.add_argument('--bbox_shift', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=get_config_value('model.batch_size', 16), help="infer batch")

    # 自定义视频配置
    parser.add_argument('--customvideo_config', type=str, default='', help="custom action json")

    # TTS 配置
    parser.add_argument('--tts', type=str, default=get_config_value('tts.default_engine', 'edgetts'),
                        help="tts service type")  # xtts gpt-sovits cosyvoice fishtts tencent doubao indextts2 azuretts
    parser.add_argument('--REF_FILE', type=str, default=get_config_value('tts.default_voice', "zh-CN-YunxiaNeural"),
                        help="参考文件名或语音模型ID，默认值为 edgetts的语音模型ID zh-CN-YunxiaNeural, 若--tts指定为azuretts, 可以使用Azure语音模型ID, 如zh-CN-XiaoxiaoMultilingualNeural")
    parser.add_argument('--REF_TEXT', type=str, default=None)
    parser.add_argument('--TTS_SERVER', type=str,
                        default=get_config_value('tts.server', 'http://127.0.0.1:9880'))  # http://localhost:9000
    # parser.add_argument('--CHARACTER', type=str, default='test')
    # parser.add_argument('--EMOTION', type=str, default='default')

    # 模型选择
    parser.add_argument('--model', type=str,
                        default=get_config_value('model.default_model', 'musetalk'))  # musetalk wav2lip ultralight

    # 传输方式
    parser.add_argument('--transport', type=str,
                        default=get_config_value('server.transport', 'rtcpush'))  # webrtc rtcpush virtualcam
    parser.add_argument('--push_url', type=str,
                        default=get_config_value('server.push_url',
                                                 'http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream'))  # rtmp://localhost/live/livestream

    # 会话配置
    parser.add_argument('--max_session', type=int,
                        default=get_config_value('server.max_session', 1))  # multi session count
    parser.add_argument('--listenport', type=int, default=get_config_value('server.listenport', 8010),
                        help="web listen port")
    # 预加载配置
    parser.add_argument('--preload', action='store_true', default=get_config_value('avatar.preload_enabled', False),
                        help="enable avatar preload feature")
    # 测试模式
    parser.add_argument('--test', action='store_true', default=False, help="test mode, print info and exit")

    opt = parser.parse_args()

    # 加载自定义视频配置
    opt.customopt = []
    if opt.customvideo_config != '':
        with open(opt.customvideo_config, 'r') as file:
            opt.customopt = json.load(file)

    # 应用性能优化配置
    perf_config = get_performance_config()
    perf_config.apply_to_opt(opt)

    # 初始化模块化的 avatar/资源管理器
    avatar_manager = AvatarManager(opt, get_config_value)
    kb = StudyAbroadKnowledgeBase(get_config_value("rag.db_path", "study_abroad_kb.db"))

    # -------------------------
    # auth (admin/student)
    # -------------------------
    auth_token_secret = str(get_config_value("auth.token_secret", "dev-only-change-me"))
    auth_token_ttl_seconds = int(get_config_value("auth.token_ttl_seconds", 24 * 3600))
    auth_db_path = str(get_config_value("auth.db_path", "data/auth.db"))
    auth_store = AuthStore(auth_db_path)
    # Ensure default accounts exist
    auth_store.ensure_user("admin", "123456", "admin")
    auth_store.ensure_user("student", "123456", "student")

    # chat history
    history_db_path = str(get_config_value("history.db_path", "data/chat_history.db"))
    chat_history = ChatHistoryStore(history_db_path)

    # avatar admin config store
    avatar_admin_db_path = str(get_config_value("avatar_admin.db_path", "data/avatar_admin.db"))
    avatar_admin_store = AvatarAdminStore(avatar_admin_db_path)
    # auto register existing avatars
    try:
        avatars_dir = "data/avatars"
        if os.path.exists(avatars_dir):
            for item in os.listdir(avatars_dir):
                p = os.path.join(avatars_dir, item)
                if os.path.isdir(p) and item != ".gitkeep":
                    inferred = avatar_manager.infer_model_type(item, getattr(opt, "model", "musetalk"))
                    avatar_admin_store.ensure_avatar(item, item, inferred)
    except Exception as e:
        logger.warning(f"avatar admin init warn: {e}")

    avatar_manager.set_custom_action_provider(lambda aid: avatar_admin_store.get_actions(aid))
    avatar_manager.set_tts_config_provider(lambda aid: avatar_admin_store.get_tts(aid))

    # 设置预加载功能状态
    preload_enabled = opt.preload
    logger.info(f"预加载功能: {'启用' if preload_enabled else '禁用'}")

    # 从配置文件获取预加载相关设置
    import os
    preload_all_on_start = bool(get_config_value('avatar.preload_all_on_start', False))
    logger.info(f"启动时预加载全部形象: {'启用' if preload_all_on_start else '禁用'}")

    preload_queue_size = get_config_value('avatar.preload_queue_size', 10)
    cache_max_size = get_config_value('avatar.cache_max_size', 5)
    logger.info(f"预加载队列大小: {preload_queue_size}, 缓存最大大小: {cache_max_size}")

    avatar_manager.configure(preload_enabled=preload_enabled, preload_all_on_start=preload_all_on_start)

    # 启动时一次性预加载所有形象资源
    if avatar_manager.preload_enabled and avatar_manager.preload_all_on_start:
        logger.info("开始启动时全量预加载...")
        loaded = avatar_manager.preload_all_avatars_on_start()
        logger.info(f"启动时全量预加载完成，加载数量: {loaded}")

    # 测试模式，只打印信息然后退出
    if opt.test:
        logger.info("测试模式: 服务器配置信息")
        logger.info(f"  传输方式: {opt.transport}")
        logger.info(f"  模型: {opt.model}")
        logger.info(f"  Avatar ID: {opt.avatar_id}")
        logger.info(f"  端口: {opt.listenport}")
        logger.info("  注意: 模型和头像将在需要时动态加载，而不是在启动时加载")
        logger.info("测试完成，退出")
        exit(0)

    # 延迟加载模型和头像，只有在需要时才加载
    logger.info("服务器启动完成，模型和头像将在需要时动态加载")

    # 启动虚拟摄像头模式
    if opt.transport == 'virtualcam' and opt.avatar_id:
        thread_quit = Event()
        nerfreals[0] = build_nerfreal(0)
        rendthrd = Thread(target=nerfreals[0].render, args=(thread_quit,))
        rendthrd.start()
    elif opt.transport == 'virtualcam' and not opt.avatar_id:
        logger.warning("虚拟摄像头模式需要指定avatar_id，将使用默认值: %s",
                       get_config_value('avatar.default_avatar_id', 'avator_1'))
        thread_quit = Event()
        nerfreals[0] = build_nerfreal(0)
        rendthrd = Thread(target=nerfreals[0].render, args=(thread_quit,))
        rendthrd.start()

    #############################################################################
    # 创建 aiohttp 应用
    appasync = web.Application(client_max_size=1024 ** 2 * 100)  # 100MB
    appasync.on_shutdown.append(on_shutdown)

    # 头像身份预设持久化（按 avatar_id）
    avatar_identity_path = str(get_config_value("identity.store_path", "data/avatar_identities.json"))
    avatar_identities: Dict[str, str] = {}
    if os.path.exists(avatar_identity_path):
        try:
            with open(avatar_identity_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    avatar_identities = {str(k): str(v) for k, v in data.items()}
        except Exception as e:
            logger.warning(f"读取头像身份预设失败: {e}")

    def _save_avatar_identities() -> bool:
        try:
            parent_dir = os.path.dirname(avatar_identity_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            with open(avatar_identity_path, "w", encoding="utf-8") as f:
                json.dump(avatar_identities, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.warning(f"保存头像身份预设失败: {e}")
            return False

    state = AppState(
        opt=opt,
        avatar_manager=avatar_manager,
        kb=kb,
        get_config_value=get_config_value,
        rand_sessionid=lambda: randN(6),
        nerfreals=nerfreals,
        identities=identities,
        avatar_identities=avatar_identities,
        save_avatar_identities=_save_avatar_identities,
        auth_token_secret=auth_token_secret,
        auth_token_ttl_seconds=auth_token_ttl_seconds,
        auth_store=auth_store,
        chat_history=chat_history,
        avatar_admin_store=avatar_admin_store,
        preload_queue=preload_queue,
        preload_in_progress=preload_in_progress,
        pcs=pcs,
        chat_semaphore=asyncio.Semaphore(int(get_config_value("server.max_inflight_chats", 2))),
        chat_gen_ids={},
    )

    # 注册路由（模块化）
    setup_routes(appasync, state)
    appasync.router.add_static('/', path='web')

    # 配置 CORS
    cors = aiohttp_cors.setup(appasync, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
    })
    for route in list(appasync.router.routes()):
        cors.add(route)

    # 确定首页文件名
    pagename = 'webrtcapi.html'
    if opt.transport == 'rtmp':
        pagename = 'echoapi.html'
    elif opt.transport == 'rtcpush':
        pagename = 'rtcpushapi.html'

    # 打印启动信息
    logger.info('start http server; http://<serverip>:' + str(opt.listenport) + '/' + pagename)
    logger.info('如果使用webrtc，推荐访问webrtc集成前端: http://<serverip>:' + str(opt.listenport) + '/dashboard.html')


    def run_server(runner):
        """
        运行服务器

        Args:
            runner: 应用运行器
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, '0.0.0.0', opt.listenport)
        loop.run_until_complete(site.start())

        # 启动 RTCPUSH 模式的会话
        if opt.transport == 'rtcpush':
            for k in range(opt.max_session):
                push_url = opt.push_url
                if k != 0:
                    push_url = opt.push_url + str(k)
                loop.run_until_complete(run(push_url, k))

        loop.run_forever()

        # 启动服务器


    run_server(web.AppRunner(appasync))

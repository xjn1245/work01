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
from flask_sockets import Sockets
import base64
import json
# import gevent
# from gevent import pywsgi
# from geventwebsocket.handler import WebSocketHandler
import re
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
    import os
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

            # 优化5: 更智能的文本分割，减少TTS等待时间
            # 当积累到一定长度或遇到自然断句时发送
            if len(result) >= 50 or (len(result) > 10 and any(punct in result for punct in "，。！？；")):
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
from typing import Dict
from logger import logger
import gc
from performance_config import get_performance_config, optimize_llm_response, optimize_tts_config

app = Flask(__name__)
# sockets = Sockets(app)
nerfreals: Dict[int, BaseReal] = {}  # 存储会话ID到数字人实例的映射
identities: Dict[int, str] = {}  # 存储会话ID到身份信息的映射
opt = None  # 命令行参数
model = None  # 数字人模型
avatar = None  # 数字人形象

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


def build_nerfreal(sessionid: int, avatar_id=None) -> BaseReal:
    """
    根据指定的模型类型创建数字人实例

    Args:
        sessionid: 会话ID
        avatar_id: 头像ID，如果为None则使用opt.avatar_id

    Returns:
        BaseReal: 数字人实例
    """
    opt.sessionid = sessionid

    # 如果指定了avatar_id，则使用该ID，否则使用opt.avatar_id
    target_avatar_id = avatar_id if avatar_id is not None else opt.avatar_id

    # 根据当前模型重新加载对应的模型和头像
    if opt.model == 'wav2lip':
        from lipreal import LipReal, load_model, load_avatar
        current_model = load_model("./models/wav2lip.pth")
        current_avatar = load_avatar(target_avatar_id)
        nerfreal = LipReal(opt, current_model, current_avatar)
    elif opt.model == 'musetalk':
        from musereal import MuseReal, load_model, load_avatar
        current_model = load_model()
        current_avatar = load_avatar(target_avatar_id)
        nerfreal = MuseReal(opt, current_model, current_avatar)
    # elif opt.model == 'ernerf':
    #     from nerfreal import NeRFReal, load_model, load_avatar
    #     current_model = load_model(opt)
    #     current_avatar = load_avatar(target_avatar_id)
    #     nerfreal = NeRFReal(opt, current_model, current_avatar)
    elif opt.model == 'ultralight':
        from lightreal import LightReal, load_model, load_avatar
        current_model = load_model(opt)
        current_avatar = load_avatar(target_avatar_id)
        nerfreal = LightReal(opt, current_model, current_avatar)
    return nerfreal


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
                        avatar_list.append({
                            "id": item,
                            "name": item,
                            "type": "wav2lip" if "wav2lip" in item else "musetalk"
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
    parser.add_argument('--fps', type=int, default=50, help="audio fps,must be 50")
    # 滑动窗口左-中-右长度 (单位: 20ms)
    parser.add_argument('-l', type=int, default=10)
    parser.add_argument('-m', type=int, default=8)
    parser.add_argument('-r', type=int, default=10)

    # GUI 尺寸
    parser.add_argument('--W', type=int, default=450, help="GUI width")
    parser.add_argument('--H', type=int, default=450, help="GUI height")

    # musetalk 选项
    parser.add_argument('--avatar_id', type=str, default='avator_1', help="define which avatar in data/avatars")
    # parser.add_argument('--bbox_shift', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16, help="infer batch")

    # 自定义视频配置
    parser.add_argument('--customvideo_config', type=str, default='', help="custom action json")

    # TTS 配置
    parser.add_argument('--tts', type=str, default='edgetts',
                        help="tts service type")  # xtts gpt-sovits cosyvoice fishtts tencent doubao indextts2 azuretts
    parser.add_argument('--REF_FILE', type=str, default="zh-CN-YunxiaNeural",
                        help="参考文件名或语音模型ID，默认值为 edgetts的语音模型ID zh-CN-YunxiaNeural, 若--tts指定为azuretts, 可以使用Azure语音模型ID, 如zh-CN-XiaoxiaoMultilingualNeural")
    parser.add_argument('--REF_TEXT', type=str, default=None)
    parser.add_argument('--TTS_SERVER', type=str, default='http://127.0.0.1:9880')  # http://localhost:9000
    # parser.add_argument('--CHARACTER', type=str, default='test')
    # parser.add_argument('--EMOTION', type=str, default='default')

    # 模型选择
    parser.add_argument('--model', type=str, default='musetalk')  # musetalk wav2lip ultralight

    # 传输方式
    parser.add_argument('--transport', type=str, default='rtcpush')  # webrtc rtcpush virtualcam
    parser.add_argument('--push_url', type=str,
                        default='http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream')  # rtmp://localhost/live/livestream

    # 会话配置
    parser.add_argument('--max_session', type=int, default=1)  # multi session count
    parser.add_argument('--listenport', type=int, default=8010, help="web listen port")

    opt = parser.parse_args()

    # 加载自定义视频配置
    opt.customopt = []
    if opt.customvideo_config != '':
        with open(opt.customvideo_config, 'r') as file:
            opt.customopt = json.load(file)

    # 应用性能优化配置
    perf_config = get_performance_config()
    perf_config.apply_to_opt(opt)

    # 加载模型和头像
    # if opt.model == 'ernerf':
    #     from nerfreal import NeRFReal, load_model, load_avatar
    #     model = load_model(opt)
    #     avatar = load_avatar(opt)
    if opt.model == 'musetalk':
        from musereal import MuseReal, load_model, load_avatar, warm_up

        logger.info(opt)
        model = load_model()
        avatar = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size, model)
    elif opt.model == 'wav2lip':
        from lipreal import LipReal, load_model, load_avatar, warm_up

        logger.info(opt)
        model = load_model("./models/wav2lip.pth")
        avatar = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size, model, 256)
    elif opt.model == 'ultralight':
        from lightreal import LightReal, load_model, load_avatar, warm_up

        logger.info(opt)
        model = load_model(opt)
        avatar = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size, avatar, 160)

    # 启动虚拟摄像头模式
    if opt.transport == 'virtualcam':
        thread_quit = Event()
        nerfreals[0] = build_nerfreal(0)
        rendthrd = Thread(target=nerfreals[0].render, args=(thread_quit,))
        rendthrd.start()

    #############################################################################
    # 创建 aiohttp 应用
    appasync = web.Application(client_max_size=1024 ** 2 * 100)  # 100MB
    appasync.on_shutdown.append(on_shutdown)

    # 注册路由
    appasync.router.add_post("/offer", offer)
    appasync.router.add_post("/human", human)
    appasync.router.add_post("/humanaudio", humanaudio)
    appasync.router.add_post("/set_audiotype", set_audiotype)
    appasync.router.add_post("/record", record)
    appasync.router.add_post("/interrupt_talk", interrupt_talk)
    appasync.router.add_post("/is_speaking", is_speaking)
    appasync.router.add_post("/switch_avatar", switch_avatar)
    appasync.router.add_post("/set_identity", set_identity)
    appasync.router.add_post("/clear_identity", clear_identity)
    appasync.router.add_get("/get_avatars", get_avatars)
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

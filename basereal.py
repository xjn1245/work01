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

import math
import torch
import numpy as np

import subprocess
import os
import time
import copy
import cv2
import glob
import resampy

import queue
from queue import Queue
from threading import Thread, Event
from io import BytesIO
import soundfile as sf

import asyncio
from av import AudioFrame, VideoFrame

import av
from fractions import Fraction

from ttsreal import EdgeTTS,SovitsTTS,XTTS,CosyVoiceTTS,FishTTS,TencentTTS,DoubaoTTS,IndexTTS2,AzureTTS
from logger import logger
from config_loader import get_config_value

from tqdm import tqdm
def read_imgs(img_list):
    frames = []
    logger.info('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

def play_audio(quit_event,queue):        
    import pyaudio
    p = pyaudio.PyAudio()
    stream = p.open(
        rate=16000,
        channels=1,
        format=8,
        output=True,
        output_device_index=1,
    )
    stream.start_stream()
    # while queue.qsize() <= 0:
    #     time.sleep(0.1)
    while not quit_event.is_set():
        stream.write(queue.get(block=True))
    stream.close()

class BaseReal:
    def __init__(self, opt):
        self.opt = opt
        self.sample_rate = 16000
        self.chunk = self.sample_rate // opt.fps # 320 samples per chunk (20ms * 16000 / 1000)
        self.sessionid = self.opt.sessionid

        self._fallback_in_progress = False
        self.fallback_tts_list = []
        self.tts = self._build_tts_with_fallback()

        self.speaking = False

        self.recording = False
        self._record_video_pipe = None
        self._record_audio_pipe = None
        self.width = self.height = 0

        self.curr_state=0
        self.custom_img_cycle = {}
        self.custom_audio_cycle = {}
        self.custom_audio_index = {}
        self.custom_index = {}
        self.custom_opt = {}
        self._metric_tts_ms = 0
        self._metric_action_ms = 0

        # 运行时开关：B 模式默认关闭 LivePortrait 表情驱动
        self._enable_liveportrait_expression = bool(
            get_config_value("liveportrait.expression_enabled", False)
        )
        # 当启用 LivePortrait 时，是否连嘴部也用 LivePortrait（忽略口型模型输出）
        self._liveportrait_mouth_from_liveportrait = bool(
            get_config_value("liveportrait.mouth_from_liveportrait", True)
        )

        # 表情=C：LivePortrait 表情引擎（按需懒加载）
        self._liveportrait_engine = None
        self._current_expression_id = None
        # speaking 时复用的表情底图（只在 eventpoint.start 时生成一次）
        self._liveportrait_expr_face_crop = None
        # 防止 LivePortrait 引擎初始化阻塞渲染/音频链路
        self._liveportrait_engine_init_thread = None
        # 表情+C/口型=1：用 TTS 的 eventpoint start/end 来判定 speaking，避免 ASR 队列空导致频繁切换编排动作
        self._tts_segment_active: bool = False
        self.__loadcustom()

    def reset_runtime_metrics(self):
        self._metric_tts_ms = 0
        self._metric_action_ms = 0

    def add_tts_metric(self, elapsed_ms: int):
        try:
            self._metric_tts_ms += max(0, int(elapsed_ms))
        except Exception:
            pass

    def add_action_metric(self, elapsed_ms: int):
        try:
            self._metric_action_ms += max(0, int(elapsed_ms))
        except Exception:
            pass

    def snapshot_runtime_metrics(self):
        return {
            "tts_ms": int(self._metric_tts_ms),
            "action_ms": int(self._metric_action_ms),
        }

    def _get_tts_class(self, engine_name: str):
        mapping = {
            "edgetts": EdgeTTS,
            "gpt-sovits": SovitsTTS,
            "xtts": XTTS,
            "cosyvoice": CosyVoiceTTS,
            "fishtts": FishTTS,
            "tencent": TencentTTS,
            "doubao": DoubaoTTS,
            "indextts2": IndexTTS2,
            "azuretts": AzureTTS,
        }
        return mapping.get(engine_name)

    def _is_engine_available(self, engine_name: str) -> bool:
        if engine_name == "azuretts":
            return bool(os.getenv("AZURE_SPEECH_KEY")) and bool(os.getenv("AZURE_TTS_REGION"))
        if engine_name == "doubao":
            return bool(os.getenv("DOUBAO_APPID")) and bool(os.getenv("DOUBAO_TOKEN"))
        return True

    def _get_voice_for_engine(self, engine_name: str, default_voice: str) -> str:
        voice_map = get_config_value("tts.fallback_voice_map", {}) or {}
        if isinstance(voice_map, dict) and engine_name in voice_map and voice_map[engine_name]:
            return str(voice_map[engine_name])
        return default_voice

    def _make_tts_instance(self, engine_name: str):
        tts_cls = self._get_tts_class(engine_name)
        if tts_cls is None:
            return None
        if not self._is_engine_available(engine_name):
            logger.warning(f"[TTS-Fallback] engine not available (missing env): {engine_name}")
            return None

        local_opt = copy.copy(self.opt)
        local_opt.tts = engine_name
        base_voice = str(getattr(self.opt, "REF_FILE", "") or "")
        primary_engine = str(getattr(self.opt, "tts", "edgetts") or "edgetts").strip().lower()
        # 优先级：
        # 1) 当前会话/数字人显式 voice（self.opt.REF_FILE）
        # 2) fallback_voice_map（主要给回退引擎）
        if engine_name == primary_engine and base_voice.strip():
            local_opt.REF_FILE = base_voice
        else:
            local_opt.REF_FILE = self._get_voice_for_engine(engine_name, base_voice)
        logger.info(f"[TTS] init engine={engine_name}, voice={getattr(local_opt, 'REF_FILE', '')}")
        try:
            inst = tts_cls(local_opt, self)
            # attach helper tag for logging
            setattr(inst, "_engine_name", engine_name)
            return inst
        except Exception as e:
            logger.warning(f"[TTS-Fallback] init failed for {engine_name}: {e}")
            return None

    def _build_tts_with_fallback(self):
        primary_engine = getattr(self.opt, "tts", "edgetts")
        primary = self._make_tts_instance(primary_engine)
        fallback_enabled = bool(get_config_value("tts.fallback_enabled", True))

        fallback_engines = get_config_value("tts.fallback_engines", ["azuretts", "doubao", "edgetts"]) or []
        if not isinstance(fallback_engines, list):
            fallback_engines = ["azuretts", "doubao", "edgetts"]

        # keep order, remove duplicates, exclude primary
        seen = set()
        normalized = []
        for e in fallback_engines:
            en = str(e).strip().lower()
            if not en or en == primary_engine or en in seen:
                continue
            seen.add(en)
            normalized.append(en)

        if fallback_enabled:
            for en in normalized:
                inst = self._make_tts_instance(en)
                if inst is not None:
                    self.fallback_tts_list.append(inst)
        else:
            self.fallback_tts_list = []

        if primary is not None:
            logger.info(
                f"[TTS-Fallback] enabled={fallback_enabled}, primary={primary_engine}, "
                f"fallbacks={[getattr(x, '_engine_name', '?') for x in self.fallback_tts_list]}"
            )
            return primary

        # primary unavailable: choose first fallback as active
        if self.fallback_tts_list:
            promoted = self.fallback_tts_list.pop(0)
            logger.warning(
                f"[TTS-Fallback] primary '{primary_engine}' unavailable, promote fallback '{getattr(promoted, '_engine_name', '?')}'"
            )
            return promoted

        raise RuntimeError("No available TTS engine after fallback checks.")

    def try_fallback_tts(self, msg, failed_engine: str = "") -> bool:
        if not bool(get_config_value("tts.fallback_enabled", True)):
            return False
        if self._fallback_in_progress:
            return False
        if not self.fallback_tts_list:
            return False

        self._fallback_in_progress = True
        try:
            for fb in self.fallback_tts_list:
                engine = getattr(fb, "_engine_name", "")
                if failed_engine and engine == failed_engine:
                    continue
                try:
                    logger.warning(f"[TTS-Fallback] try fallback engine: {engine}")
                    fb.txt_to_audio(msg)
                    return True
                except Exception as e:
                    logger.warning(f"[TTS-Fallback] fallback engine failed {engine}: {e}")
                    continue
            return False
        finally:
            self._fallback_in_progress = False

    def put_msg_txt(self,msg,datainfo:dict={}):
        self.tts.put_msg_txt(msg,datainfo)
    
    def put_audio_frame(self,audio_chunk,datainfo:dict={}): #16khz 20ms pcm
        self.asr.put_audio_frame(audio_chunk,datainfo)

    def put_audio_file(self,filebyte,datainfo:dict={}): 
        input_stream = BytesIO(filebyte)
        stream = self.__create_bytes_stream(input_stream)
        streamlen = stream.shape[0]
        idx=0
        while streamlen >= self.chunk:  #and self.state==State.RUNNING
            self.put_audio_frame(stream[idx:idx+self.chunk],datainfo)
            streamlen -= self.chunk
            idx += self.chunk
    
    def __create_bytes_stream(self,byte_stream):
        #byte_stream=BytesIO(buffer)
        stream, sample_rate = sf.read(byte_stream) # [T*sample_rate,] float64
        logger.info(f'[INFO]put audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            logger.info(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]
    
        if sample_rate != self.sample_rate and stream.shape[0]>0:
            logger.info(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream

    def flush_talk(self):
        self.tts.flush_talk()
        self.asr.flush_talk()

    def is_speaking(self)->bool:
        return self.speaking
    
    def __loadcustom(self):
        for item in self.opt.customopt:
            logger.info(item)
            input_img_list = glob.glob(os.path.join(item['imgpath'], '*.[jpJP][pnPN]*[gG]'))
            input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            self.custom_img_cycle[item['audiotype']] = read_imgs(input_img_list)
            self.custom_audio_cycle[item['audiotype']], sample_rate = sf.read(item['audiopath'], dtype='float32')
            self.custom_audio_index[item['audiotype']] = 0
            self.custom_index[item['audiotype']] = 0
            self.custom_opt[item['audiotype']] = item

    def init_customindex(self):
        self.curr_state=0
        for key in self.custom_audio_index:
            self.custom_audio_index[key]=0
        for key in self.custom_index:
            self.custom_index[key]=0

    def notify(self,eventpoint):
        logger.info("notify:%s",eventpoint)

    def start_recording(self):
        """开始录制视频"""
        if self.recording:
            return

        command = ['ffmpeg',
                    '-y', '-an',
                    '-f', 'rawvideo',
                    '-vcodec','rawvideo',
                    '-pix_fmt', 'bgr24', #像素格式
                    '-s', "{}x{}".format(self.width, self.height),
                    '-r', str(25),
                    '-i', '-',
                    '-pix_fmt', 'yuv420p', 
                    '-vcodec', "h264",
                    #'-f' , 'flv',                  
                    f'temp{self.opt.sessionid}.mp4']
        self._record_video_pipe = subprocess.Popen(command, shell=False, stdin=subprocess.PIPE)

        acommand = ['ffmpeg',
                    '-y', '-vn',
                    '-f', 's16le',
                    #'-acodec','pcm_s16le',
                    '-ac', '1',
                    '-ar', '16000',
                    '-i', '-',
                    '-acodec', 'aac',
                    #'-f' , 'wav',                  
                    f'temp{self.opt.sessionid}.aac']
        self._record_audio_pipe = subprocess.Popen(acommand, shell=False, stdin=subprocess.PIPE)

        self.recording = True
        # self.recordq_video.queue.clear()
        # self.recordq_audio.queue.clear()
        # self.container = av.open(path, mode="w")
    
        # process_thread = Thread(target=self.record_frame, args=())
        # process_thread.start()
    
    def record_video_data(self,image):
        if self.width == 0:
            print("image.shape:",image.shape)
            self.height,self.width,_ = image.shape
        if self.recording:
            self._record_video_pipe.stdin.write(image.tostring())

    def record_audio_data(self,frame):
        if self.recording:
            self._record_audio_pipe.stdin.write(frame.tostring())
    
    # def record_frame(self): 
    #     videostream = self.container.add_stream("libx264", rate=25)
    #     videostream.codec_context.time_base = Fraction(1, 25)
    #     audiostream = self.container.add_stream("aac")
    #     audiostream.codec_context.time_base = Fraction(1, 16000)
    #     init = True
    #     framenum = 0       
    #     while self.recording:
    #         try:
    #             videoframe = self.recordq_video.get(block=True, timeout=1)
    #             videoframe.pts = framenum #int(round(framenum*0.04 / videostream.codec_context.time_base))
    #             videoframe.dts = videoframe.pts
    #             if init:
    #                 videostream.width = videoframe.width
    #                 videostream.height = videoframe.height
    #                 init = False
    #             for packet in videostream.encode(videoframe):
    #                 self.container.mux(packet)
    #             for k in range(2):
    #                 audioframe = self.recordq_audio.get(block=True, timeout=1)
    #                 audioframe.pts = int(round((framenum*2+k)*0.02 / audiostream.codec_context.time_base))
    #                 audioframe.dts = audioframe.pts
    #                 for packet in audiostream.encode(audioframe):
    #                     self.container.mux(packet)
    #             framenum += 1
    #         except queue.Empty:
    #             print('record queue empty,')
    #             continue
    #         except Exception as e:
    #             print(e)
    #             #break
    #     for packet in videostream.encode(None):
    #         self.container.mux(packet)
    #     for packet in audiostream.encode(None):
    #         self.container.mux(packet)
    #     self.container.close()
    #     self.recordq_video.queue.clear()
    #     self.recordq_audio.queue.clear()
    #     print('record thread stop')
		
    def stop_recording(self):
        """停止录制视频"""
        if not self.recording:
            return
        self.recording = False 
        self._record_video_pipe.stdin.close()  #wait() 
        self._record_video_pipe.wait()
        self._record_audio_pipe.stdin.close()
        self._record_audio_pipe.wait()
        cmd_combine_audio = f"ffmpeg -y -i temp{self.opt.sessionid}.aac -i temp{self.opt.sessionid}.mp4 -c:v copy -c:a copy data/record.mp4"
        os.system(cmd_combine_audio) 
        #os.remove(output_path)

    def mirror_index(self,size, index):
        #size = len(self.coord_list_cycle)
        turn = index // size
        res = index % size
        if turn % 2 == 0:
            return res
        else:
            return size - res - 1 
    
    def get_audio_stream(self,audiotype):
        idx = self.custom_audio_index[audiotype]
        stream = self.custom_audio_cycle[audiotype][idx:idx+self.chunk]
        self.custom_audio_index[audiotype] += self.chunk
        if self.custom_audio_index[audiotype]>=self.custom_audio_cycle[audiotype].shape[0]:
            self.curr_state = 1  #当前视频不循环播放，切换到静音状态
        return stream
    
    def set_custom_state(self,audiotype, reinit=True):
        print('set_custom_state:',audiotype)
        if self.custom_audio_index.get(audiotype) is None:
            return
        self.curr_state = audiotype
        if reinit:
            self.custom_audio_index[audiotype] = 0
            self.custom_index[audiotype] = 0

    def process_frames(self,quit_event,loop=None,audio_track=None,video_track=None):
        # 启用静音/说话切换过渡，让画面更自然、减少“僵硬卡顿感”
        enable_transition = True
        
        if enable_transition:
            _last_speaking = False
            _transition_start = time.time()
            # 过渡时间（越短越不影响实时性，但融合更弱；越长越柔和但延迟更明显）
            _transition_duration = 0.08
            _last_silent_frame = None  # 静音帧缓存
            _last_speaking_frame = None  # 说话帧缓存
        
        if self.opt.transport=='virtualcam':
            import pyvirtualcam
            vircam = None

            audio_tmp = queue.Queue(maxsize=3000)
            audio_thread = Thread(target=play_audio, args=(quit_event,audio_tmp,), daemon=True, name="pyaudio_stream")
            audio_thread.start()
        
        while not quit_event.is_set():
            try:
                res_frame,idx,audio_frames = self.res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
            
            if enable_transition:
                # 检测状态变化
                # speaking 判定与主分支保持一致：用 TTS segment + res_frame 是否可用
                # 只要 TTS segment 处于 start/end 之间，就尽量保持“说话状态”，
                # 即使中间偶发 res_frame 为 None，也用上一帧缓存避免跳回编排动作。
                current_speaking = bool(self._tts_segment_active) and (
                    (res_frame is not None)
                    or (
                        self._enable_liveportrait_expression
                        and self._liveportrait_mouth_from_liveportrait
                    )
                )
                if current_speaking != _last_speaking:
                    logger.info(f"状态切换：{'说话' if _last_speaking else '静音'} → {'说话' if current_speaking else '静音'}")
                    _transition_start = time.time()
                    try:
                        self.add_action_metric(int(_transition_duration * 1000))
                    except Exception:
                        pass
                _last_speaking = current_speaking

            # 统一做索引归一化，避免并发切换形象时偶发越界
            frame_count = len(self.frame_list_cycle) if hasattr(self, 'frame_list_cycle') and self.frame_list_cycle is not None else 0
            if frame_count <= 0:
                logger.warning("frame_list_cycle is empty, skip one frame")
                continue
            safe_idx = idx % frame_count

            # 更新当前 speaking 的表情 id（由 TTS 的 eventpoint.start 携带）
            try:
                eps = []
                if len(audio_frames) > 0:
                    eps.append(audio_frames[0][2])
                if len(audio_frames) > 1:
                    eps.append(audio_frames[1][2])

                # 兼容 eventpoint 可能落在前两帧中的任意一帧
                for ep in eps:
                    if not isinstance(ep, dict):
                        continue
                    if ep.get("status") == "start":
                        self._tts_segment_active = True
                        self._current_expression_id = ep.get("expression_id") or ep.get("facial_expression") or "neutral"
                        # 只在 start 时生成一次表情底图，后续 speaking 帧复用，避免每帧 LivePortrait 推理导致 GPU 99%
                        expr_id_now = str(self._current_expression_id)
                        if (
                            self._enable_liveportrait_expression
                            and self._liveportrait_engine is None
                            and hasattr(self, "face_list_cycle")
                        ):
                            faces = getattr(self, "face_list_cycle", None)
                            if isinstance(faces, list) and len(faces) > 0:
                                # 非阻塞：引擎初始化放到后台线程
                                if (
                                    self._liveportrait_engine_init_thread is None
                                    or not self._liveportrait_engine_init_thread.is_alive()
                                ):
                                    def _init_engine():
                                        try:
                                            from livetalking.services.liveportrait_expression_engine import (
                                                LivePortraitExpressionEngine,
                                            )
                                            self._liveportrait_engine = LivePortraitExpressionEngine(faces)
                                        except Exception as e:
                                            logger.warning(f"[LivePortraitExpressionEngine] init failed: {e}")
                                            self._liveportrait_engine = None

                                    self._liveportrait_engine_init_thread = Thread(
                                        target=_init_engine,
                                        daemon=True,
                                        name="liveportrait_init",
                                    )
                                    self._liveportrait_engine_init_thread.start()
                            # 引擎尚未就绪：先不要阻塞音频输出
                            self._liveportrait_expr_face_crop = None
                        else:
                            if self._liveportrait_engine is not None:
                                # 引擎就绪后再生成一次表情底图
                                self._liveportrait_expr_face_crop = self._liveportrait_engine.get_expression_face(
                                    safe_idx, expr_id_now
                                )
                    elif ep.get("status") == "end":
                        self._current_expression_id = None
                        self._liveportrait_expr_face_crop = None
                        self._tts_segment_active = False
            except Exception:
                pass

            # 新的 speaking 判断：只有当在 TTS segment 内且推理输出帧有效（res_frame != None）时才认为“说话”
            # 这样可以避免一旦 eventpoint 丢失/延迟导致 segment active 卡住时，永远不播放预设静音动作。
            should_speak = bool(self._tts_segment_active) and (
                (res_frame is not None)
                or (self._enable_liveportrait_expression and self._liveportrait_mouth_from_liveportrait)
            )

            if not should_speak:  # 静音 / 或推理认为全静音 => 取 fullimg（可能是编排动作）
                self.speaking = False
                audiotype = audio_frames[0][1] if len(audio_frames) > 0 else 1
                if self.custom_index.get(audiotype) is not None: #有自定义视频
                    mirindex = self.mirror_index(len(self.custom_img_cycle[audiotype]),self.custom_index[audiotype])
                    target_frame = self.custom_img_cycle[audiotype][mirindex]
                    self.custom_index[audiotype] += 1
                else:
                    target_frame = self.frame_list_cycle[safe_idx]
                
                if enable_transition:
                    # 说话→静音过渡
                    if time.time() - _transition_start < _transition_duration and _last_speaking_frame is not None:
                        alpha = min(1.0, (time.time() - _transition_start) / _transition_duration)
                        # 过渡融合要求两帧 shape 完全一致；不一致时退化为当前帧，避免线程崩溃
                        if _last_speaking_frame.shape == target_frame.shape:
                            combine_frame = cv2.addWeighted(_last_speaking_frame, 1-alpha, target_frame, alpha, 0)
                        else:
                            # 形状不一致通常来自不同来源帧分辨率差异。
                            # 这里仅做降级，不要高频 warning 影响实时线程稳定性。
                            logger.debug(
                                f"transition shape mismatch(speaking->silent): "
                                f"{_last_speaking_frame.shape} vs {target_frame.shape}, fallback to target_frame"
                            )
                            combine_frame = target_frame
                    else:
                        combine_frame = target_frame
                    # 缓存静音帧
                    _last_silent_frame = combine_frame.copy()
                else:
                    combine_frame = target_frame
            else:
                self.speaking = True
                try:
                    if res_frame is None:
                        # 推理偶发输出 None，但我们仍在 TTS segment 内：使用上一帧说话内容兜底
                        if (
                            _last_speaking_frame is not None
                            and _last_speaking_frame.shape == self.frame_list_cycle[safe_idx].shape
                        ):
                            current_frame = _last_speaking_frame.copy()
                        else:
                            current_frame = self.frame_list_cycle[safe_idx]
                    else:
                        # LivePortrait 表情底图：
                        # - mouth_from_liveportrait=True：按帧生成，避免整段复用导致“嘴不动”
                        # - 否则：沿用分段缓存，降低 GPU 开销
                        expression_face_crop = self._liveportrait_expr_face_crop
                        if (
                            self._enable_liveportrait_expression
                            and self._liveportrait_engine is not None
                            and self._current_expression_id is not None
                        ):
                            if self._liveportrait_mouth_from_liveportrait:
                                expression_face_crop = self._liveportrait_engine.get_expression_face(
                                    safe_idx, str(self._current_expression_id)
                                )
                            elif expression_face_crop is None:
                                expression_face_crop = self._liveportrait_engine.get_expression_face(
                                    safe_idx, str(self._current_expression_id)
                                )
                                self._liveportrait_expr_face_crop = expression_face_crop

                        # 口型也用 LivePortrait：忽略 res_frame（口型模型输出）
                        if (
                            self._enable_liveportrait_expression
                            and self._liveportrait_mouth_from_liveportrait
                            and expression_face_crop is not None
                        ):
                            current_frame = self.paste_back_frame(
                                None,
                                safe_idx,
                                expression_face_crop=expression_face_crop,
                            )
                        else:
                            current_frame = self.paste_back_frame(
                                res_frame,
                                safe_idx,
                                expression_face_crop=expression_face_crop,
                            )
                except Exception as e:
                    logger.warning(f"paste_back_frame error: {e}")
                    continue
                if enable_transition:
                    # 静音→说话过渡
                    if time.time() - _transition_start < _transition_duration and _last_silent_frame is not None:
                        alpha = min(1.0, (time.time() - _transition_start) / _transition_duration)
                        # 过渡融合要求两帧 shape 完全一致；不一致时退化为当前帧，避免线程崩溃
                        if _last_silent_frame.shape == current_frame.shape:
                            combine_frame = cv2.addWeighted(_last_silent_frame, 1-alpha, current_frame, alpha, 0)
                        else:
                            # 仅降级，不做高频 warning
                            logger.debug(
                                f"transition shape mismatch(silent->speaking): "
                                f"{_last_silent_frame.shape} vs {current_frame.shape}, fallback to current_frame"
                            )
                            combine_frame = current_frame
                    else:
                        combine_frame = current_frame
                    # 缓存说话帧
                    _last_speaking_frame = combine_frame.copy()
                else:
                    combine_frame = current_frame

            cv2.putText(combine_frame, "LiveTalking", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128,128,128), 1)
            if self.opt.transport=='virtualcam':
                if vircam==None:
                    height, width,_= combine_frame.shape
                    vircam = pyvirtualcam.Camera(width=width, height=height, fps=25, fmt=pyvirtualcam.PixelFormat.BGR,print_fps=True)
                vircam.send(combine_frame)
            else: #webrtc
                image = combine_frame
                new_frame = VideoFrame.from_ndarray(image, format="bgr24")
                asyncio.run_coroutine_threadsafe(video_track._queue.put((new_frame,None)), loop)
            self.record_video_data(combine_frame)

            for audio_frame in audio_frames:
                frame,type,eventpoint = audio_frame
                frame = (frame * 32767).astype(np.int16)

                if self.opt.transport=='virtualcam':
                    audio_tmp.put(frame.tobytes()) #TODO
                else: #webrtc
                    new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
                    new_frame.planes[0].update(frame.tobytes())
                    new_frame.sample_rate=16000
                    asyncio.run_coroutine_threadsafe(audio_track._queue.put((new_frame,eventpoint)), loop)
                self.record_audio_data(frame)
            if self.opt.transport=='virtualcam':
                vircam.sleep_until_next_frame()
        if self.opt.transport=='virtualcam':
            audio_thread.join()
            vircam.close()
        logger.info('basereal process_frames thread stop') 
    
    # def process_custom(self,audiotype:int,idx:int):
    #     if self.curr_state!=audiotype: #从推理切到口播
    #         if idx in self.switch_pos:  #在卡点位置可以切换
    #             self.curr_state=audiotype
    #             self.custom_index=0
    #     else:
    #         self.custom_index+=1
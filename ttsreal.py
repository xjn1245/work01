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
from __future__ import annotations
import time
import numpy as np
import soundfile as sf
import resampy
import asyncio
import edge_tts

import os
import hmac
import hashlib
import base64
import json
import uuid

from typing import Iterator

import requests

import queue
from queue import Queue
from io import BytesIO
import copy, websockets, gzip
import azure.cognitiveservices.speech as speechsdk

from threading import Thread, Event
from enum import Enum

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from basereal import BaseReal

from logger import logger


class State(Enum):
    RUNNING = 0
    PAUSE = 1


class BaseTTS:
    def __init__(self, opt, parent: BaseReal):
        self.opt = opt
        self.parent = parent

        self.fps = opt.fps  # 20 ms per frame
        self.sample_rate = 16000
        self.chunk = self.sample_rate // self.fps  # 320 samples per chunk (20ms * 16000 / 1000)
        self.input_stream = BytesIO()

        self.msgqueue = Queue()
        self.state = State.RUNNING

    def flush_talk(self):
        self.msgqueue.queue.clear()
        self.state = State.PAUSE

    def put_msg_txt(self, msg: str, datainfo: dict = {}):
        if len(msg) > 0:
            self.msgqueue.put((msg, datainfo))

    def _emit_pre_speech_padding(self, eventpoint: dict | None = None):
        """
        在每段语音前补一小段静音，避免 WebRTC/播放器起播阶段吞掉首音节。
        """
        try:
            from config_loader import get_config_value
            import math

            pre_silence_ms = int(get_config_value("tts.pre_speech_silence_ms", 120))
            if pre_silence_ms <= 0:
                return

            frame_ms = max(1.0, 1000.0 / float(self.fps))
            pad_frames = max(1, int(math.ceil(pre_silence_ms / frame_ms)))
            for i in range(pad_frames):
                ep = eventpoint if i == 0 else None
                self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), ep)
        except Exception:
            # padding 失败不影响主流程
            pass

    def render(self, quit_event):
        process_thread = Thread(target=self.process_tts, args=(quit_event,))
        process_thread.start()

    def process_tts(self, quit_event):
        while not quit_event.is_set():
            try:
                msg: tuple[str, dict] = self.msgqueue.get(block=True, timeout=1)
                self.state = State.RUNNING
            except queue.Empty:
                continue
            t0 = time.perf_counter()
            self.txt_to_audio(msg)
            try:
                self.parent.add_tts_metric(int((time.perf_counter() - t0) * 1000))
            except Exception:
                pass
        logger.info('ttsreal thread stop')

    def txt_to_audio(self, msg: tuple[str, dict]):
        pass


###########################################################################################
class EdgeTTS(BaseTTS):
    def txt_to_audio(self, msg: tuple[str, dict]):
        text, textevent = msg
        # BaseReal.opt 为会话真源；TTS 内部 opt 是 copy.copy 的副本，可能与 SET 不同步
        session_opt = getattr(getattr(self, "parent", None), "opt", None)
        voicename = (
            str(getattr(session_opt, "REF_FILE", "") or "").strip()
            or str(getattr(self.opt, "REF_FILE", "") or "").strip()
        )
        try:
            from config_loader import get_config_value
            ui_lang = str((textevent or {}).get("ui_lang", "") or "").strip().lower()
            # 支持配置覆盖；未配置时使用内置兜底
            voice_by_lang = get_config_value("tts.edge_voice_by_lang", {}) or {}
            if not isinstance(voice_by_lang, dict):
                voice_by_lang = {}
            fallback_map = {
                "zh-cn": "zh-CN-YunxiaNeural",
                "en": "en-US-JennyNeural",
                "ja": "ja-JP-NanamiNeural",
                "ko": "ko-KR-SunHiNeural",
            }
            selected = ""
            if ui_lang.startswith("ja"):
                selected = str(voice_by_lang.get("ja") or fallback_map["ja"])
            elif ui_lang.startswith("ko"):
                selected = str(voice_by_lang.get("ko") or fallback_map["ko"])
            elif ui_lang.startswith("en"):
                selected = str(voice_by_lang.get("en") or fallback_map["en"])
            elif ui_lang.startswith("zh"):
                selected = str(voice_by_lang.get("zh-CN") or fallback_map["zh-cn"])

            # 规则（与「语言听感」强相关）：
            # - Edge 的 zh-* 神经网络读日文/韩文文本时，听感会像中文或发音错误，这是模型限制。
            # - edge_voice_force_locale_match=true（默认）：当本段对话 ui_lang 与所选音色语种不一致时，
            #   合成时临时改用 edge_voice_by_lang 对应语种（会话 REF_FILE / 下拉框仍保留，不写回）。
            # - edge_voice_force_locale_match=false：即使用户锁了音色也绝不按语种覆盖（听感可能「语言不对」）。
            # 默认 false：优先使用会话里 /set_tts_voice 与后台配置的音色，避免被 edge_voice_by_lang「顶掉」。
            # 需要日文界面仍强制用日语音色合成时，再在 config 里设为 true。
            force_locale = bool(get_config_value("tts.edge_voice_force_locale_match", False))
            current_voice = str(voicename or "").strip()
            user_locked = bool(
                getattr(session_opt, "_tts_voice_user_locked", False)
                if session_opt is not None
                else getattr(self.opt, "_tts_voice_user_locked", False)
            )

            target_family = ""
            if ui_lang.startswith("ja"):
                target_family = "ja"
            elif ui_lang.startswith("ko"):
                target_family = "ko"
            elif ui_lang.startswith("en"):
                target_family = "en"
            elif ui_lang.startswith("zh"):
                target_family = "zh"

            voice_lower = current_voice.lower()
            current_family = ""
            if voice_lower.startswith("ja-"):
                current_family = "ja"
            elif voice_lower.startswith("ko-"):
                current_family = "ko"
            elif voice_lower.startswith("en-"):
                current_family = "en"
            elif voice_lower.startswith("zh-"):
                current_family = "zh"

            # 当前音色语种与对话 ui_lang 不一致（含无前缀、跨语种）
            auto_fix_lang = bool(
                selected
                and target_family
                and (not current_family or current_family != target_family)
            )

            if auto_fix_lang and ((not user_locked) or force_locale):
                voicename = selected

            logger.info(
                f"EdgeTTS language={ui_lang or 'n/a'}, user_locked={user_locked}, "
                f"force_locale={force_locale}, using voice={voicename}"
            )
            # 仅更新 TTS 实例上的 opt，避免把「临时按 UI 语种切出的 voice」写回会话 opt，
            # 否则从日韩切回中文时会丢失用户/后台配置的中文音色。
            try:
                self.opt.REF_FILE = voicename
            except Exception:
                pass
        except Exception:
            pass
        # True streaming path: EdgeTTS mp3 chunks -> ffmpeg decode -> 20ms PCM frames
        try:
            import shutil
            import subprocess
            from threading import Thread, Event
            import numpy as np
            from config_loader import get_config_value

            ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg:
                raise FileNotFoundError("ffmpeg not found")

            cmd = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-fflags",
                "nobuffer",
                "-flags",
                "low_delay",
                "-probesize",
                "32k",
                "-analyzeduration",
                "0",
                "-i",
                "pipe:0",
                "-f",
                "s16le",
                "-ac",
                "1",
                "-ar",
                str(self.sample_rate),
                "pipe:1",
            ]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            pcm_buf = bytearray()
            chunk_bytes = self.chunk * 2  # s16le mono
            started = False
            stop_read = Event()
            audio_received = False
            tts_timeout_s = float(get_config_value("server.tts_timeout_seconds", 20))
            deadline = time.time() + tts_timeout_s

            def _drain_stderr():
                try:
                    if proc.stderr:
                        proc.stderr.read()
                except Exception:
                    pass

            Thread(target=_drain_stderr, daemon=True).start()

            def _stdout_reader():
                nonlocal started, pcm_buf
                try:
                    while (not stop_read.is_set()) and self.state == State.RUNNING and time.time() < deadline:
                        if not proc.stdout:
                            break
                        data = proc.stdout.read(4096)
                        if not data:
                            break
                        pcm_buf.extend(data)
                        while len(pcm_buf) >= chunk_bytes and self.state == State.RUNNING:
                            frame_bytes = bytes(pcm_buf[:chunk_bytes])
                            del pcm_buf[:chunk_bytes]
                            frame = (np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32767.0)
                            eventpoint = None
                            if not started:
                                eventpoint = {"status": "start", "text": text}
                                eventpoint.update(**textevent)
                                self._emit_pre_speech_padding(eventpoint)
                                started = True
                                eventpoint = None
                            self.parent.put_audio_frame(frame, eventpoint)
                except Exception as e:
                    logger.warning(f"EdgeTTS stdout reader error: {e}")

            reader_thread = Thread(target=_stdout_reader, daemon=True, name="edgetts_ffmpeg_stdout")
            reader_thread.start()

            async def _stream_and_feed():
                nonlocal audio_received

                # 复用 __main 的校验逻辑，但不要写入 input_stream
                if not text or len(text.strip()) == 0:
                    logger.error("EdgeTTS: 文本内容为空")
                    return False

                # 不再使用硬编码白名单（会误伤如 zh-CN-YunjianNeural 等合法音色）。
                # 直接使用配置音色，若无效由 edge_tts/回退链路处理。
                vn = voicename
                logger.info(f"EdgeTTS streaming using voice={vn}")

                connect_timeout = int(get_config_value("tts.edgetts.connect_timeout", 10))
                receive_timeout = int(get_config_value("tts.edgetts.receive_timeout", 60))
                rate = str(getattr(self.opt, "TTS_RATE", "+0%"))
                communicate = edge_tts.Communicate(
                    text,
                    vn,
                    rate=rate,
                    connect_timeout=connect_timeout,
                    receive_timeout=receive_timeout,
                )

                async for chunk in communicate.stream():
                    if self.state != State.RUNNING:
                        break
                        if time.time() >= deadline:
                            logger.warning("EdgeTTS tts timeout reached, stop streaming")
                            break
                    if chunk["type"] == "audio":
                        audio_received = True
                        if proc.stdin:
                            proc.stdin.write(chunk["data"])
                    # ignore boundaries

                return audio_received

            t = time.time()
            audio_received = asyncio.new_event_loop().run_until_complete(_stream_and_feed())
            logger.info(f"-------edge tts (stream) time:{time.time() - t:.4f}s")
            if time.time() >= deadline and not started:
                logger.warning("EdgeTTS timed out before start frame; degrade without audio")

            # close stdin, flush remaining decoded data
            try:
                if proc.stdin:
                    proc.stdin.close()
            except Exception:
                pass

            # allow reader thread to drain remaining stdout quickly
            try:
                proc.wait(timeout=2)
            except Exception:
                pass
            stop_read.set()
            try:
                reader_thread.join(timeout=1)
            except Exception:
                pass

            while len(pcm_buf) >= chunk_bytes and self.state == State.RUNNING:
                frame_bytes = bytes(pcm_buf[:chunk_bytes])
                del pcm_buf[:chunk_bytes]
                frame = (np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32767.0)
                eventpoint = None
                if not started:
                    eventpoint = {"status": "start", "text": text}
                    eventpoint.update(**textevent)
                    started = True
                self.parent.put_audio_frame(frame, eventpoint)

            if self.state == State.RUNNING and started:
                eventpoint = {"status": "end", "text": text}
                eventpoint.update(**textevent)
                self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)

            try:
                proc.kill()
            except Exception:
                pass

            if not audio_received:
                logger.error("edgetts err!!!!!")
                try:
                    self.parent.try_fallback_tts(msg, failed_engine="edgetts")
                except Exception:
                    pass
            return

        except Exception as e:
            # Fallback to old behaviour (buffer then decode)
            logger.warning(f"EdgeTTS streaming decode unavailable, fallback. reason={e}")

        t = time.time()
        asyncio.new_event_loop().run_until_complete(self.__main(voicename, text))
        logger.info(f'-------edge tts time:{time.time() - t:.4f}s')
        if self.input_stream.getbuffer().nbytes <= 0:  # edgetts err
            logger.error('edgetts err!!!!!')
            try:
                self.parent.try_fallback_tts(msg, failed_engine="edgetts")
            except Exception:
                pass
            return

        self.input_stream.seek(0)
        stream = self.__create_bytes_stream(self.input_stream)
        streamlen = stream.shape[0]
        idx = 0
        while streamlen >= self.chunk and self.state == State.RUNNING:
            eventpoint = {}
            streamlen -= self.chunk
            if idx == 0:
                eventpoint = {'status': 'start', 'text': text}
                eventpoint.update(**textevent)
                self._emit_pre_speech_padding(eventpoint)
                eventpoint = {}
            elif streamlen < self.chunk:
                eventpoint = {'status': 'end', 'text': text}
                eventpoint.update(**textevent)
            self.parent.put_audio_frame(stream[idx:idx + self.chunk], eventpoint)
            idx += self.chunk
        self.input_stream.seek(0)
        self.input_stream.truncate()

    def __create_bytes_stream(self, byte_stream):
        # byte_stream=BytesIO(buffer)
        stream, sample_rate = sf.read(byte_stream)  # [T*sample_rate,] float64
        logger.info(f'[INFO]tts audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            logger.info(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]

        if sample_rate != self.sample_rate and stream.shape[0] > 0:
            logger.info(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream

    async def __main(self, voicename: str, text: str):
        try:
            # 修复1: 检查文本有效性
            if not text or len(text.strip()) == 0:
                logger.error("EdgeTTS: 文本内容为空")
                return

            # 修复2: 验证语音名称有效性
            # 不限制为硬编码白名单，避免把合法新音色错误回退为默认女声
            logger.info(f"EdgeTTS fallback path using voice={voicename}")

            # 修复3: 添加超时和重试机制
            import asyncio
            from edge_tts.exceptions import NoAudioReceived

            rate = str(getattr(self.opt, "TTS_RATE", "+0%"))
            communicate = edge_tts.Communicate(text, voicename, rate=rate)

            # with open(OUTPUT_FILE, "wb") as file:
            first = True
            audio_received = False

            try:
                async for chunk in communicate.stream():
                    if first:
                        first = False
                    if chunk["type"] == "audio" and self.state == State.RUNNING:
                        # self.push_audio(chunk["data"])
                        self.input_stream.write(chunk["data"])
                        audio_received = True
                        # file.write(chunk["data"])
                    elif chunk["type"] == "WordBoundary":
                        pass
            except NoAudioReceived:
                logger.error(f"EdgeTTS: 未收到音频，语音: {voicename}, 文本: {text[:50]}...")
                # 尝试使用默认语音重试
                if voicename != "zh-CN-YunxiaNeural":
                    logger.info("EdgeTTS: 尝试使用默认语音重试")
                    await self.__main("zh-CN-YunxiaNeural", text)
                return
            except Exception as e:
                logger.error(f"EdgeTTS流式处理错误: {e}")
                return

            if not audio_received:
                logger.error("EdgeTTS: 未收到任何音频数据")

        except Exception as e:
            logger.exception('edgetts')


###########################################################################################
class FishTTS(BaseTTS):
    def txt_to_audio(self, msg: tuple[str, dict]):
        text, textevent = msg
        self.stream_tts(
            self.fish_speech(
                text,
                self.opt.REF_FILE,
                self.opt.REF_TEXT,
                "zh",  # en args.language,
                self.opt.TTS_SERVER,  # "http://127.0.0.1:5000", #args.server_url,
            ),
            msg
        )

    def fish_speech(self, text, reffile, reftext, language, server_url) -> Iterator[bytes]:
        start = time.perf_counter()
        req = {
            'text': text,
            'reference_id': reffile,
            'format': 'wav',
            'streaming': True,
            'use_memory_cache': 'on'
        }
        try:
            res = requests.post(
                f"{server_url}/v1/tts",
                json=req,
                stream=True,
                headers={
                    "content-type": "application/json",
                },
            )
            end = time.perf_counter()
            logger.info(f"fish_speech Time to make POST: {end - start}s")

            if res.status_code != 200:
                logger.error("Error:%s", res.text)
                return

            first = True

            for chunk in res.iter_content(chunk_size=17640):  # 1764 44100*20ms*2
                # print('chunk len:',len(chunk))
                if first:
                    end = time.perf_counter()
                    logger.info(f"fish_speech Time to first chunk: {end - start}s")
                    first = False
                if chunk and self.state == State.RUNNING:
                    yield chunk
            # print("gpt_sovits response.elapsed:", res.elapsed)
        except Exception as e:
            logger.exception('fishtts')

    def stream_tts(self, audio_stream, msg: tuple[str, dict]):
        text, textevent = msg
        first = True
        for chunk in audio_stream:
            if chunk is not None and len(chunk) > 0:
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = resampy.resample(x=stream, sr_orig=44100, sr_new=self.sample_rate)
                # byte_stream=BytesIO(buffer)
                # stream = self.__create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx = 0
                while streamlen >= self.chunk:
                    eventpoint = {}
                    if first:
                        eventpoint = {'status': 'start', 'text': text}
                        eventpoint.update(**textevent)  # eventpoint={'status':'start','text':text,'msgevent':textevent}
                        first = False
                    self.parent.put_audio_frame(stream[idx:idx + self.chunk], eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
        eventpoint = {'status': 'end', 'text': text}
        eventpoint.update(**textevent)  # eventpoint={'status':'end','text':text,'msgevent':textevent}
        self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)

    ###########################################################################################


class SovitsTTS(BaseTTS):
    def txt_to_audio(self, msg: tuple[str, dict]):
        text, textevent = msg
        self.stream_tts(
            self.gpt_sovits(
                text=text,
                reffile=self.opt.REF_FILE,
                reftext=self.opt.REF_TEXT,
                language="zh",  # en args.language,
                server_url=self.opt.TTS_SERVER,  # "http://127.0.0.1:5000", #args.server_url,
            ),
            msg
        )

    def gpt_sovits(self, text, reffile, reftext, language, server_url) -> Iterator[bytes]:
        start = time.perf_counter()
        req = {
            'text': text,
            'text_lang': language,
            'ref_audio_path': reffile,
            'prompt_text': reftext,
            'prompt_lang': language,
            'media_type': 'ogg',
            'streaming_mode': True
        }
        # req["text"] = text
        # req["text_language"] = language
        # req["character"] = character
        # req["emotion"] = emotion
        # #req["stream_chunk_size"] = stream_chunk_size  # you can reduce it to get faster response, but degrade quality
        # req["streaming_mode"] = True
        try:
            res = requests.post(
                f"{server_url}/tts",
                json=req,
                stream=True,
            )
            end = time.perf_counter()
            logger.info(f"gpt_sovits Time to make POST: {end - start}s")

            if res.status_code != 200:
                logger.error("Error:%s", res.text)
                return

            first = True

            for chunk in res.iter_content(chunk_size=None):  # 12800 1280 32K*20ms*2
                logger.info('chunk len:%d', len(chunk))
                if first:
                    end = time.perf_counter()
                    logger.info(f"gpt_sovits Time to first chunk: {end - start}s")
                    first = False
                if chunk and self.state == State.RUNNING:
                    yield chunk
            # print("gpt_sovits response.elapsed:", res.elapsed)
        except Exception as e:
            logger.exception('sovits')

    def __create_bytes_stream(self, byte_stream):
        # byte_stream=BytesIO(buffer)
        stream, sample_rate = sf.read(byte_stream)  # [T*sample_rate,] float64
        logger.info(f'[INFO]tts audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            logger.info(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]

        if sample_rate != self.sample_rate and stream.shape[0] > 0:
            logger.info(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream

    def stream_tts(self, audio_stream, msg: tuple[str, dict]):
        text, textevent = msg
        first = True
        for chunk in audio_stream:
            if chunk is not None and len(chunk) > 0:
                # stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                # stream = resampy.resample(x=stream, sr_orig=32000, sr_new=self.sample_rate)
                byte_stream = BytesIO(chunk)
                stream = self.__create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx = 0
                while streamlen >= self.chunk:
                    eventpoint = {}
                    if first:
                        eventpoint = {'status': 'start', 'text': text}
                        eventpoint.update(**textevent)
                        first = False
                    self.parent.put_audio_frame(stream[idx:idx + self.chunk], eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
        eventpoint = {'status': 'end', 'text': text}
        eventpoint.update(**textevent)
        self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)


###########################################################################################
class CosyVoiceTTS(BaseTTS):
    def txt_to_audio(self, msg: tuple[str, dict]):
        text, textevent = msg
        self.stream_tts(
            self.cosy_voice(
                text,
                self.opt.REF_FILE,
                self.opt.REF_TEXT,
                "zh",  # en args.language,
                self.opt.TTS_SERVER,  # "http://127.0.0.1:5000", #args.server_url,
            ),
            msg
        )

    def cosy_voice(self, text, reffile, reftext, language, server_url) -> Iterator[bytes]:
        start = time.perf_counter()
        payload = {
            'tts_text': text,
            'prompt_text': reftext
        }
        try:
            files = [('prompt_wav', ('prompt_wav', open(reffile, 'rb'), 'application/octet-stream'))]
            res = requests.request("GET", f"{server_url}/inference_zero_shot", data=payload, files=files, stream=True)

            end = time.perf_counter()
            logger.info(f"cosy_voice Time to make POST: {end - start}s")

            if res.status_code != 200:
                logger.error("Error:%s", res.text)
                return

            first = True

            for chunk in res.iter_content(chunk_size=9600):  # 960 24K*20ms*2
                if first:
                    end = time.perf_counter()
                    logger.info(f"cosy_voice Time to first chunk: {end - start}s")
                    first = False
                if chunk and self.state == State.RUNNING:
                    yield chunk
        except Exception as e:
            logger.exception('cosyvoice')

    def stream_tts(self, audio_stream, msg: tuple[str, dict]):
        text, textevent = msg
        first = True
        for chunk in audio_stream:
            if chunk is not None and len(chunk) > 0:
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = resampy.resample(x=stream, sr_orig=24000, sr_new=self.sample_rate)
                # byte_stream=BytesIO(buffer)
                # stream = self.__create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx = 0
                while streamlen >= self.chunk:
                    eventpoint = {}
                    if first:
                        eventpoint = {'status': 'start', 'text': text}
                        eventpoint.update(**textevent)
                        first = False
                    self.parent.put_audio_frame(stream[idx:idx + self.chunk], eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
        eventpoint = {'status': 'end', 'text': text}
        eventpoint.update(**textevent)
        self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)

    ###########################################################################################


_PROTOCOL = "https://"
_HOST = "tts.cloud.tencent.com"
_PATH = "/stream"
_ACTION = "TextToStreamAudio"


class TencentTTS(BaseTTS):
    def __init__(self, opt, parent):
        super().__init__(opt, parent)
        self.appid = os.getenv("TENCENT_APPID")
        self.secret_key = os.getenv("TENCENT_SECRET_KEY")
        self.secret_id = os.getenv("TENCENT_SECRET_ID")
        self.voice_type = int(opt.REF_FILE)
        self.codec = "pcm"
        self.sample_rate = 16000
        self.volume = 0
        self.speed = 0

    def __gen_signature(self, params):
        sort_dict = sorted(params.keys())
        sign_str = "POST" + _HOST + _PATH + "?"
        for key in sort_dict:
            sign_str = sign_str + key + "=" + str(params[key]) + '&'
        sign_str = sign_str[:-1]
        hmacstr = hmac.new(self.secret_key.encode('utf-8'),
                           sign_str.encode('utf-8'), hashlib.sha1).digest()
        s = base64.b64encode(hmacstr)
        s = s.decode('utf-8')
        return s

    def __gen_params(self, session_id, text):
        params = dict()
        params['Action'] = _ACTION
        params['AppId'] = int(self.appid)
        params['SecretId'] = self.secret_id
        params['ModelType'] = 1
        params['VoiceType'] = self.voice_type
        params['Codec'] = self.codec
        params['SampleRate'] = self.sample_rate
        params['Speed'] = self.speed
        params['Volume'] = self.volume
        params['SessionId'] = session_id
        params['Text'] = text

        timestamp = int(time.time())
        params['Timestamp'] = timestamp
        params['Expired'] = timestamp + 24 * 60 * 60
        return params

    def txt_to_audio(self, msg: tuple[str, dict]):
        text, textevent = msg
        self.stream_tts(
            self.tencent_voice(
                text,
                self.opt.REF_FILE,
                self.opt.REF_TEXT,
                "zh",  # en args.language,
                self.opt.TTS_SERVER,  # "http://127.0.0.1:5000", #args.server_url,
            ),
            msg
        )

    def tencent_voice(self, text, reffile, reftext, language, server_url) -> Iterator[bytes]:
        start = time.perf_counter()
        session_id = str(uuid.uuid1())
        params = self.__gen_params(session_id, text)
        signature = self.__gen_signature(params)
        headers = {
            "Content-Type": "application/json",
            "Authorization": str(signature)
        }
        url = _PROTOCOL + _HOST + _PATH
        try:
            res = requests.post(url, headers=headers,
                                data=json.dumps(params), stream=True)

            end = time.perf_counter()
            logger.info(f"tencent Time to make POST: {end - start}s")

            first = True

            for chunk in res.iter_content(chunk_size=6400):  # 640 16K*20ms*2
                # logger.info('chunk len:%d',len(chunk))
                if first:
                    try:
                        rsp = json.loads(chunk)
                        # response["Code"] = rsp["Response"]["Error"]["Code"]
                        # response["Message"] = rsp["Response"]["Error"]["Message"]
                        logger.error("tencent tts:%s", rsp["Response"]["Error"]["Message"])
                        return
                    except:
                        end = time.perf_counter()
                        logger.info(f"tencent Time to first chunk: {end - start}s")
                        first = False
                if chunk and self.state == State.RUNNING:
                    yield chunk
        except Exception as e:
            logger.exception('tencent')

    def stream_tts(self, audio_stream, msg: tuple[str, dict]):
        text, textevent = msg
        first = True
        last_stream = np.array([], dtype=np.float32)
        for chunk in audio_stream:
            if chunk is not None and len(chunk) > 0:
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = np.concatenate((last_stream, stream))
                # stream = resampy.resample(x=stream, sr_orig=24000, sr_new=self.sample_rate)
                # byte_stream=BytesIO(buffer)
                # stream = self.__create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx = 0
                while streamlen >= self.chunk:
                    eventpoint = {}
                    if first:
                        eventpoint = {'status': 'start', 'text': text}
                        eventpoint.update(**textevent)
                        first = False
                    self.parent.put_audio_frame(stream[idx:idx + self.chunk], eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
                last_stream = stream[idx:]  # get the remain stream
        eventpoint = {'status': 'end', 'text': text}
        eventpoint.update(**textevent)
        self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)

    ###########################################################################################


class DoubaoTTS(BaseTTS):
    def __init__(self, opt, parent):
        super().__init__(opt, parent)
        # 从配置中读取火山引擎参数
        self.appid = os.getenv("DOUBAO_APPID")
        self.token = os.getenv("DOUBAO_TOKEN")
        if not self.appid or not self.token:
            logger.error(
                f"[DoubaoTTS] Missing DOUBAO_APPID/DOUBAO_TOKEN. appid is None={not bool(self.appid)}, token is None={not bool(self.token)}"
            )
        _cluster = 'volcano_tts'
        _host = "openspeech.bytedance.com"
        self.api_url = f"wss://{_host}/api/v1/tts/ws_binary"

        self.request_json = {
            "app": {
                "appid": self.appid,
                "token": "access_token",
                "cluster": _cluster
            },
            "user": {
                "uid": "xxx"
            },
            "audio": {
                "voice_type": "xxx",
                "encoding": "pcm",
                "rate": 16000,
                "speed_ratio": 1.0,
                "volume_ratio": 1.0,
                "pitch_ratio": 1.0,
            },
            "request": {
                "reqid": "xxx",
                "text": "字节跳动语音合成。",
                "text_type": "plain",
                "operation": "xxx"
            }
        }

    async def doubao_voice(self, text):  # -> Iterator[bytes]:
        start = time.perf_counter()
        voice_type = self.opt.REF_FILE

        try:
            default_header = bytearray(b'\x11\x10\x11\x00')
            submit_request_json = copy.deepcopy(self.request_json)
            submit_request_json["user"]["uid"] = self.parent.sessionid
            submit_request_json["audio"]["voice_type"] = voice_type
            submit_request_json["request"]["text"] = text
            submit_request_json["request"]["reqid"] = str(uuid.uuid4())
            submit_request_json["request"]["operation"] = "submit"

            raw_payload = str.encode(json.dumps(submit_request_json))

            # 方案优先级：
            # 1) 不压缩（很多协议本身就要求明文 JSON）
            # 2) gzip 压缩（兼容部分服务器实现）
            attempts = [
                ("no_compress", raw_payload),
                ("gzip", gzip.compress(raw_payload)),
            ]

            # 注意：豆包接口对鉴权头格式较敏感，项目原实现使用 "Bearer; <token>"
            header = {"Authorization": f"Bearer; {self.token}"}

            first = True
            for attempt_name, payload_bytes in attempts:
                got_audio = False
                try:
                    full_client_request = bytearray(default_header)
                    # payload length 字节序尝试：使用 little-endian 以适配部分协议实现
                    full_client_request.extend((len(payload_bytes)).to_bytes(4, "little"))
                    full_client_request.extend(payload_bytes)

                    logger.info(f"[DoubaoTTS] submit attempt={attempt_name}, text_len={len(text)}")

                    async with websockets.connect(
                        self.api_url, extra_headers=header, ping_interval=None
                    ) as ws:
                        await ws.send(full_client_request)

                        while True:
                            res = await ws.recv()

                            # ws.recv should normally return bytes for binary frames.
                            if isinstance(res, str):
                                # 如果服务端异常返回文本帧，直接跳过该帧
                                logger.warning(f"[DoubaoTTS] got text frame unexpectedly, len={len(res)}")
                                continue
                            if not isinstance(res, (bytes, bytearray)):
                                continue

                            if len(res) < 8:
                                continue

                            header_size = res[0] & 0x0F
                            message_type = res[1] >> 4
                            message_type_specific_flags = res[1] & 0x0F
                            payload = res[header_size * 4 :]

                            if message_type == 0xB:  # audio-only server response
                                if message_type_specific_flags == 0:
                                    # ACK frame (no sequence number)
                                    continue

                                if len(payload) < 8:
                                    continue

                                sequence_number = int.from_bytes(payload[:4], "big", signed=True)
                                payload_size = int.from_bytes(payload[4:8], "big", signed=False)
                                payload = payload[8:]

                                got_audio = True

                                if first:
                                    end = time.perf_counter()
                                    logger.info(f"doubao tts Time to first chunk: {end - start}s")
                                    first = False

                                yield payload
                                if sequence_number < 0:
                                    break
                            else:
                                break

                    # 如果这次已经拿到音频，直接退出重试循环
                    if got_audio:
                        break
                except Exception as e:
                    logger.warning(f"[DoubaoTTS] attempt={attempt_name} failed: {e}")
                    continue

            # 允许外层在无音频时打日志降级
        except Exception as e:
            logger.exception(f"doubao outer error: {e}")
        # # 检查响应状态码
        # if response.status_code == 200:
        #     # 处理响应数据
        #     audio_data = base64.b64decode(response.json().get('data'))
        #     yield audio_data
        # else:
        #     logger.error(f"请求失败，状态码: {response.status_code}")
        #     return

    def txt_to_audio(self, msg: tuple[str, dict]):
        text, textevent = msg
        asyncio.new_event_loop().run_until_complete(
            self.stream_tts(
                self.doubao_voice(text),
                msg
            )
        )

    async def stream_tts(self, audio_stream, msg: tuple[str, dict]):
        text, textevent = msg
        first = True
        last_stream = np.array([], dtype=np.float32)
        audio_received = False
        async for chunk in audio_stream:
            if chunk is not None and len(chunk) > 0:
                audio_received = True
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = np.concatenate((last_stream, stream))
                # stream = resampy.resample(x=stream, sr_orig=24000, sr_new=self.sample_rate)
                # byte_stream=BytesIO(buffer)
                # stream = self.__create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx = 0
                while streamlen >= self.chunk:
                    eventpoint = {}
                    if first:
                        eventpoint = {'status': 'start', 'text': text}
                        eventpoint.update(**textevent)
                        first = False
                    self.parent.put_audio_frame(stream[idx:idx + self.chunk], eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
                last_stream = stream[idx:]  # get the remain stream
        eventpoint = {'status': 'end', 'text': text}
        eventpoint.update(**textevent)
        self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)
        if not audio_received:
            logger.error(
                f"[DoubaoTTS] no audio chunks received. voice_type(opt.REF_FILE)='{getattr(self.opt, 'REF_FILE', None)}'"
            )
            try:
                self.parent.try_fallback_tts(msg, failed_engine="doubao")
            except Exception:
                pass


###########################################################################################
class IndexTTS2(BaseTTS):
    def __init__(self, opt, parent):
        super().__init__(opt, parent)
        # IndexTTS2 配置参数
        self.server_url = opt.TTS_SERVER  # Gradio服务器地址，如 "http://127.0.0.1:7860/"
        self.ref_audio_path = opt.REF_FILE  # 参考音频文件路径
        self.max_tokens = getattr(opt, 'MAX_TOKENS', 120)  # 最大token数

        # 初始化Gradio客户端
        try:
            from gradio_client import Client, handle_file
            self.client = Client(self.server_url)
            self.handle_file = handle_file
            logger.info(f"IndexTTS2 Gradio客户端初始化成功: {self.server_url}")
        except ImportError:
            logger.error("IndexTTS2 需要安装 gradio_client: pip install gradio_client")
            raise
        except Exception as e:
            logger.error(f"IndexTTS2 Gradio客户端初始化失败: {e}")
            raise

    def txt_to_audio(self, msg):
        text, textevent = msg
        try:
            # 先进行文本分割
            segments = self.split_text(text)
            if not segments:
                logger.error("IndexTTS2 文本分割失败")
                return

            logger.info(f"IndexTTS2 文本分割为 {len(segments)} 个片段")

            # 循环生成每个片段的音频
            for i, segment_text in enumerate(segments):
                if self.state != State.RUNNING:
                    break

                logger.info(f"IndexTTS2 正在生成第 {i + 1}/{len(segments)} 段音频...")
                audio_file = self.indextts2_generate(segment_text)

                if audio_file:
                    # 为每个片段创建事件信息
                    segment_msg = (segment_text, textevent)
                    self.file_to_stream(audio_file, segment_msg, is_first=(i == 0), is_last=(i == len(segments) - 1))
                else:
                    logger.error(f"IndexTTS2 第 {i + 1} 段音频生成失败")

        except Exception as e:
            logger.exception(f"IndexTTS2 txt_to_audio 错误: {e}")

    def split_text(self, text):
        """使用 IndexTTS2 API 分割文本"""
        try:
            logger.info(f"IndexTTS2 开始分割文本，长度: {len(text)}")

            # 调用文本分割 API
            result = self.client.predict(
                text=text,
                max_text_tokens_per_segment=self.max_tokens,
                api_name="/on_input_text_change"
            )

            # 解析分割结果
            if 'value' in result and 'data' in result['value']:
                data = result['value']['data']
                logger.info(f"IndexTTS2 共分割为 {len(data)} 个片段")

                segments = []
                for i, item in enumerate(data):
                    序号 = item[0] + 1
                    分句内容 = item[1]
                    token数 = item[2]
                    logger.info(f"片段 {序号}: {len(分句内容)} 字符, {token数} tokens")
                    segments.append(分句内容)

                return segments
            else:
                logger.error(f"IndexTTS2 文本分割结果格式异常: {result}")
                return [text]  # 如果分割失败，返回原文本

        except Exception as e:
            logger.exception(f"IndexTTS2 文本分割失败: {e}")
            return [text]  # 如果分割失败，返回原文本

    def indextts2_generate(self, text):
        """调用 IndexTTS2 Gradio API 生成语音"""
        start = time.perf_counter()

        try:
            # 调用 gen_single API
            result = self.client.predict(
                emo_control_method="Same as the voice reference",
                prompt=self.handle_file(self.ref_audio_path),
                text=text,
                emo_ref_path=self.handle_file(self.ref_audio_path),
                emo_weight=0.8,
                vec1=0.5,
                vec2=0,
                vec3=0,
                vec4=0,
                vec5=0,
                vec6=0,
                vec7=0,
                vec8=0,
                emo_text="",
                emo_random=False,
                max_text_tokens_per_segment=self.max_tokens,
                param_16=True,
                param_17=0.8,
                param_18=30,
                param_19=0.8,
                param_20=0,
                param_21=3,
                param_22=10,
                param_23=1500,
                api_name="/gen_single"
            )

            end = time.perf_counter()
            logger.info(f"IndexTTS2 片段生成完成，耗时: {end - start:.2f}s")

            # 返回生成的音频文件路径
            if 'value' in result:
                audio_file = result['value']
                return audio_file
            else:
                logger.error(f"IndexTTS2 结果格式异常: {result}")
                return None

        except Exception as e:
            logger.exception(f"IndexTTS2 API调用失败: {e}")
            return None

    def file_to_stream(self, audio_file, msg, is_first=False, is_last=False):
        """将音频文件转换为音频流"""
        text, textevent = msg

        try:
            # 读取音频文件
            stream, sample_rate = sf.read(audio_file)
            logger.info(f'IndexTTS2 音频文件 {sample_rate}Hz: {stream.shape}')

            # 转换为float32
            stream = stream.astype(np.float32)

            # 如果是多声道，只取第一个声道
            if stream.ndim > 1:
                logger.info(f'IndexTTS2 音频有 {stream.shape[1]} 个声道，只使用第一个')
                stream = stream[:, 0]

            # 重采样到目标采样率
            if sample_rate != self.sample_rate and stream.shape[0] > 0:
                logger.info(f'IndexTTS2 重采样: {sample_rate}Hz -> {self.sample_rate}Hz')
                stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

            # 分块发送音频流
            streamlen = stream.shape[0]
            idx = 0
            first_chunk = True

            while streamlen >= self.chunk and self.state == State.RUNNING:
                eventpoint = None

                # 只在第一个片段的第一个chunk发送start事件
                if is_first and first_chunk:
                    eventpoint = {'status': 'start', 'text': text, 'msgevent': textevent}
                    first_chunk = False

                self.parent.put_audio_frame(stream[idx:idx + self.chunk], eventpoint)
                idx += self.chunk
                streamlen -= self.chunk

            # 只在最后一个片段发送end事件
            if is_last:
                eventpoint = {'status': 'end', 'text': text, 'msgevent': textevent}
                self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)

            # 清理临时文件
            try:
                if os.path.exists(audio_file):
                    os.remove(audio_file)
                    logger.info(f"IndexTTS2 已删除临时文件: {audio_file}")
            except Exception as e:
                logger.warning(f"IndexTTS2 删除临时文件失败: {e}")

        except Exception as e:
            logger.exception(f"IndexTTS2 音频流处理失败: {e}")


###########################################################################################
class XTTS(BaseTTS):
    def __init__(self, opt, parent):
        super().__init__(opt, parent)
        self.speaker = self.get_speaker(opt.REF_FILE, opt.TTS_SERVER)

    def txt_to_audio(self, msg: tuple[str, dict]):
        text, textevent = msg
        self.stream_tts(
            self.xtts(
                text,
                self.speaker,
                "zh-cn",  # en args.language,
                self.opt.TTS_SERVER,  # "http://localhost:9000", #args.server_url,
                "20"  # args.stream_chunk_size
            ),
            msg
        )

    def get_speaker(self, ref_audio, server_url):
        files = {"wav_file": ("reference.wav", open(ref_audio, "rb"))}
        response = requests.post(f"{server_url}/clone_speaker", files=files)
        return response.json()

    def xtts(self, text, speaker, language, server_url, stream_chunk_size) -> Iterator[bytes]:
        start = time.perf_counter()
        speaker["text"] = text
        speaker["language"] = language
        speaker[
            "stream_chunk_size"] = stream_chunk_size  # you can reduce it to get faster response, but degrade quality
        try:
            res = requests.post(
                f"{server_url}/tts_stream",
                json=speaker,
                stream=True,
            )
            end = time.perf_counter()
            logger.info(f"xtts Time to make POST: {end - start}s")

            if res.status_code != 200:
                print("Error:", res.text)
                return

            first = True

            for chunk in res.iter_content(chunk_size=9600):  # 24K*20ms*2
                if first:
                    end = time.perf_counter()
                    logger.info(f"xtts Time to first chunk: {end - start}s")
                    first = False
                if chunk:
                    yield chunk
        except Exception as e:
            print(e)

    def stream_tts(self, audio_stream, msg: tuple[str, dict]):
        text, textevent = msg
        first = True
        for chunk in audio_stream:
            if chunk is not None and len(chunk) > 0:
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = resampy.resample(x=stream, sr_orig=24000, sr_new=self.sample_rate)
                # byte_stream=BytesIO(buffer)
                # stream = self.__create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx = 0
                while streamlen >= self.chunk:
                    eventpoint = {}
                    if first:
                        eventpoint = {'status': 'start', 'text': text}
                        eventpoint.update(**textevent)
                        first = False
                    self.parent.put_audio_frame(stream[idx:idx + self.chunk], eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
        eventpoint = {'status': 'end', 'text': text}
        eventpoint.update(**textevent)
        self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)

    ###########################################################################################


class AzureTTS(BaseTTS):
    CHUNK_SIZE = 640  # 16kHz, 20ms, 16-bit Mono PCM size

    def __init__(self, opt, parent):
        super().__init__(opt, parent)
        self.audio_buffer = b''
        voicename = self.opt.REF_FILE  # 比如"zh-CN-XiaoxiaoMultilingualNeural"
        speech_key = os.getenv("AZURE_SPEECH_KEY")
        tts_region = os.getenv("AZURE_TTS_REGION")
        speech_endpoint = f"wss://{tts_region}.tts.speech.microsoft.com/cognitiveservices/websocket/v2"
        speech_config = speechsdk.SpeechConfig(subscription=speech_key, endpoint=speech_endpoint)
        speech_config.speech_synthesis_voice_name = voicename
        speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm)

        # 获取内存中流形式的结果
        self.speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
        self.speech_synthesizer.synthesizing.connect(self._on_synthesizing)

    def txt_to_audio(self, msg: tuple[str, dict]):
        msg_text: str = msg[0]
        result = self.speech_synthesizer.speak_text(msg_text)

        # 延迟指标
        fb_latency = int(result.properties.get_property(
            speechsdk.PropertyId.SpeechServiceResponse_SynthesisFirstByteLatencyMs
        ))
        fin_latency = int(result.properties.get_property(
            speechsdk.PropertyId.SpeechServiceResponse_SynthesisFinishLatencyMs
        ))
        logger.info(
            f"azure音频生成相关：首字节延迟: {fb_latency} ms, 完成延迟: {fin_latency} ms, result_id: {result.result_id}")

    # === 回调 ===
    def _on_synthesizing(self, evt: speechsdk.SpeechSynthesisEventArgs):
        if evt.result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            logger.info("SynthesizingAudioCompleted")
        elif evt.result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = evt.result.cancellation_details
            logger.info(f"Speech synthesis canceled: {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                if cancellation_details.error_details:
                    logger.info(f"Error details: {cancellation_details.error_details}")
        if self.state != State.RUNNING:
            self.audio_buffer = b''
            return

        # evt.result.audio_data 是刚到的一小段原始 PCM
        self.audio_buffer += evt.result.audio_data
        while len(self.audio_buffer) >= self.CHUNK_SIZE:
            chunk = self.audio_buffer[:self.CHUNK_SIZE]
            self.audio_buffer = self.audio_buffer[self.CHUNK_SIZE:]

            frame = (np.frombuffer(chunk, dtype=np.int16)
                     .astype(np.float32) / 32767.0)
            self.parent.put_audio_frame(frame)
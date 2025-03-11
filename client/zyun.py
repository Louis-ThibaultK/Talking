# TODO - zyun avatar

import asyncio
import concurrent.futures
import fractions
import random
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

import numpy as np
from av import Packet
from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    OutputImageRawFrame,
    StartInterruptionFrame,
    TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, StartFrame

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCConfiguration, RTCIceServer
    from av.audio.frame import AudioFrame
    from av.audio.resampler import AudioResampler
    import aiohttp
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Zyun, you need to `pip install pipecat-ai[zyun]`.")
    raise Exception(f"Missing module: {e}")


import logging
logging.basicConfig(level=logging.DEBUG)
# 只打印 aiortc 相关日志（可选）
# logging.getLogger("aiortc").setLevel(logging.DEBUG)
# logging.getLogger("aiortc").setLevel(logging.INFO)
# logging.getLogger("aioice").setLevel(logging.INFO)
# logging.getLogger("websockets").setLevel(logging.INFO)
logging.getLogger("openai").setLevel(logging.INFO)
logging.getLogger("httpcore").setLevel(logging.INFO)


AUDIO_PTIME = 0.020  # 20ms audio packetization
SAMPLE_RATE = 16000
AUDIO_TIME_BASE = fractions.Fraction(1, SAMPLE_RATE)

@dataclass
class ZyunConfig:
    """Zyun 服务配置"""
    api_url: str = "http://10.176.205.11:8010/offer"  # 默认 API 地址


class AudioTrack(MediaStreamTrack):
    """音频轨道实现"""
    def __init__(self, kind="audio", sample_rate=SAMPLE_RATE, channels=1):
        super().__init__()
        self.kind = kind
        self.sample_rate = sample_rate
        self.channels = channels
        self._queue = asyncio.Queue()
        self.timelist = []  # 记录最近包的时间戳

    _start: float
    _timestamp: int

    async def next_timestamp(self) -> Tuple[int, fractions.Fraction]:
        """计算下一个时间戳"""
        try:
            if self.readyState != "live":
                logger.warning("Track is not live, initializing timestamp anyway")
                # 继续执行而不是抛出异常

            if self.kind == 'audio':
                if hasattr(self, "_timestamp"):
                    self._timestamp += int(AUDIO_PTIME * SAMPLE_RATE)
                    current_time = time.time()
                    wait = self._start + (self._timestamp / SAMPLE_RATE) - current_time

                    # 如果等待时间过长，可能是时钟漂移，重置时间戳
                    if wait > 1.0:  # 等待超过1秒，可能有问题
                        logger.warning(f"Wait time too long ({wait:.2f}s), resetting timestamp")
                        self._start = current_time
                        self._timestamp = 0
                        wait = 0

                    if wait > 0:
                        await asyncio.sleep(wait)
                else:
                    self._start = time.time()
                    self._timestamp = 0
                    self.timelist.append(self._start)
                    logger.info(f"Audio track started at: {self._start}")
                return self._timestamp, AUDIO_TIME_BASE
        except Exception as e:
            logger.error(f"Error in next_timestamp: {e}")
            # 出错时返回一个基于当前时间的时间戳
            current_time = time.time()
            if not hasattr(self, "_start"):
                self._start = current_time
                self._timestamp = 0
            else:
                self._timestamp = int((current_time - self._start) * SAMPLE_RATE)
            return self._timestamp, AUDIO_TIME_BASE

    async def recv(self) -> Union[Frame, Packet]:
        """获取音频帧 - 被 WebRTC 内部调用"""
        try:
            # 记录队列大小
            # current_size = self._queue.qsize()
            # logger.debug(f"Before Recv: Queue size: {current_size}")

            # 从队列获取帧，设置超时
            # try:
            frame = await self._queue.get()
            # except asyncio.TimeoutError:
            #     # 队列为空或超时，生成静默帧
            #     # logger.debug("Queue get timeout, generating silence frame")
            #     frame = AudioFrame(format="s16", layout="mono", samples=int(AUDIO_PTIME * SAMPLE_RATE))
            #     for p in frame.planes:
            #         p.update(bytes(p.buffer_size))
            #     frame.sample_rate = self.sample_rate

            # 更新队列大小日志
            # current_size = self._queue.qsize()
            # logger.debug(f"After Recv: Queue size: {current_size}")

            # 确保帧不为 None
            if frame is None:
                logger.warning("Received None frame, generating silence instead")
                frame = AudioFrame(format="s16", layout="mono", samples=int(AUDIO_PTIME * SAMPLE_RATE))
                for p in frame.planes:
                    p.update(bytes(p.buffer_size))
                frame.sample_rate = self.sample_rate

            # 设置时间戳
            pts, time_base = await self.next_timestamp()
            frame.pts = pts
            frame.time_base = time_base

            logger.debug(f"Frame received: pts={frame.pts}, samples={frame.samples if hasattr(frame, 'samples') else 'N/A'}")

            return frame

        except Exception as e:
            logger.error(f"Error in recv: {e}")
            # 出错时也返回静默帧而不是抛出异常
            silent_frame = AudioFrame(format="s16", layout="mono", samples=int(AUDIO_PTIME * SAMPLE_RATE))
            for p in silent_frame.planes:
                p.update(bytes(p.buffer_size))
            silent_frame.sample_rate = self.sample_rate

            # 设置时间戳
            try:
                pts, time_base = await self.next_timestamp()
                silent_frame.pts = pts
                silent_frame.time_base = time_base
            except:
                # 如果时间戳获取失败，使用当前时间
                silent_frame.pts = int(time.time() * self.sample_rate)
                silent_frame.time_base = AUDIO_TIME_BASE

            logger.debug(f"Returning fallback silence frame after error")
            return silent_frame

    async def add_frame(self, frame):
        """添加音频帧到队列"""
        current_size = self._queue.qsize()
        if current_size != 0:
            logger.debug(f"Add: Queue size: {current_size}")

        # 添加新帧
        await self._queue.put(frame)


class StreamBuffer:
    """音视频数据缓冲区"""
    def __init__(self):
        self.audio_buffer = asyncio.Queue()
        self.video_buffer = asyncio.Queue()

    def get_audio(self):
        return self.audio_buffer

    def get_video(self):
        return self.video_buffer

    def clear_buffer(self):
        """清理缓冲区内容"""
        self.audio_buffer = asyncio.Queue()
        self.video_buffer = asyncio.Queue()


class ZyunVideoService(FrameProcessor):
    """Zyun 视频服务实现"""
    def __init__(self, zyun_config: ZyunConfig):
        super().__init__()
        self._zyun_config = zyun_config

        # 创建音频轨道
        self._audio_track = AudioTrack(kind="audio", sample_rate=SAMPLE_RATE, channels=1)

        # 初始化连接和缓冲区
        self._push_pc = None
        self._pull_pc = None
        self._pcs = set()
        self._stream_buffer = StreamBuffer()

        # 初始化任务
        self._audio_task = None
        self._video_task = None
        self._monitor_task = None

        # 初始化接收器
        self._audio_receiver = None
        self._video_receiver = None

        # 初始化状态
        self._initialized = False

        # 创建执行器
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        # 创建重采样器事件
        self._digital_person_resampler_event = asyncio.Event()
        self._pipecat_resampler = None
        self._digital_person_resampler = None

        self.buffer_lock = asyncio.Lock()  # 线程安全
        # 初始化缓冲区
        self.audio_buffer = bytearray()
        # 测试使用，用于保存wav
        self.buffer = bytearray()

    async def _post(self, url, data):
        """发送 HTTP POST 请求"""
        try:
            async with aiohttp.ClientSession() as session:
                logger.debug(f"Sending POST request to {url}")
                async with session.post(url, json=data) as response:
                    if response.status != 200:
                        logger.error(f"HTTP error: {response.status}")
                        text = await response.text()
                        logger.error(f"Response: {text}")
                        raise Exception(f"HTTP error: {response.status}")

                    logger.info(f"Server response: {await response.json()}")
                    return await response.json()
        except Exception as e:
            logger.error(f"Error in POST request: {e}")
            raise

    async def _start_connection(self):
        """初始化与 Zyun 服务的连接"""
        if not self._initialized:
            try:
                # 清理旧连接
                for pc in self._pcs:
                    await pc.close()
                self._pcs.clear()

                # 创建新连接
                self._push_pc = RTCPeerConnection()
                self._pull_pc = RTCPeerConnection()
                self._pcs.add(self._push_pc)
                self._pcs.add(self._pull_pc)

                # 设置连接状态监听
                @self._push_pc.on("connectionstatechange")
                async def on_push_connectionstatechange():
                    state = self._push_pc.connectionState
                    logger.info(f"Push connection state changed to: {state}")

                    if state == "connected":
                        logger.info("Push connection established successfully")
                    elif state == "failed" or state == "disconnected":
                        logger.error(f"Push connection {state}, attempting recovery")
                        # 尝试重新启动连接
                        self.create_task(self._recover_connection())
                    elif state == "closed":
                        logger.debug("Push connection closed")
                        # self.save_buffer_to_wav(24000, 1)

                @self._push_pc.on("iceconnectionstatechange")
                async def on_push_iceconnectionstatechange():
                    logger.info(f"Push ICE connection state: {self._push_pc.iceConnectionState}")

                # 添加音频发送轨道 - 使用 addTransceiver 而不是 addTrack
                audio_transceiver = self._push_pc.addTransceiver(self._audio_track, direction="sendonly")
                logger.info(f"Added audio track to peer connection: {audio_transceiver}")

                # 添加音频和视频接收轨道
                self._pull_pc.addTransceiver("audio", direction="recvonly")
                self._pull_pc.addTransceiver("video", direction="recvonly")

                # 设置接收轨道处理
                @self._pull_pc.on("track")
                async def on_track(track):
                    if track.kind == "audio":
                        logger.info(f"Audio track received: {track.id}")
                        self.create_task(self._process_audio_track(track))
                    elif track.kind == "video":
                        logger.info(f"Video track received: {track.id}")
                        self.create_task(self._process_video_track(track))

                # 创建并设置本地描述
                offer = await self._push_pc.createOffer()
                await self._push_pc.setLocalDescription(offer)

                pull_offer = await self._pull_pc.createOffer()
                await self._pull_pc.setLocalDescription(pull_offer)

                # 等待 ICE 收集完成
                await self._wait_for_ice_complete(self._push_pc)
                await self._wait_for_ice_complete(self._pull_pc)

                # 准备 SDP 数据
                push_sdp = self._push_pc.localDescription.sdp
                pull_sdp = self._pull_pc.localDescription.sdp

                logger.info(f"Push SDP offer: {push_sdp}")
                logger.info(f"Pull SDP offer: {pull_sdp}")

                sdp_data = {
                    "push_sdp": push_sdp,
                    "pull_sdp": pull_sdp,
                    "type": "offer",
                }

                logger.info("Sending SDP offers to Zyun service")

                # 发送请求并获取响应
                response = await self._post(self._zyun_config.api_url, sdp_data)

                # 设置远程描述
                push_answer = RTCSessionDescription(sdp=response["pull_sdp"], type="answer")
                await self._push_pc.setRemoteDescription(push_answer)

                pull_answer = RTCSessionDescription(sdp=response["push_sdp"], type="answer")
                await self._pull_pc.setRemoteDescription(pull_answer)

                logger.info("Remote descriptions set successfully")

                # 创建任务来消费和处理音频和视频
                self._audio_task = self.create_task(self._consume_and_process_audio())
                self._video_task = self.create_task(self._consume_and_process_video())
                self.create_task(self._consume_audio_from_buffer())

                self._initialized = True

                # 启动保活任务
                # self._keep_alive_task = self.create_task(self._keep_audio_alive())

                # 启用监控任务
                # self._monitor_task = self.create_task(self._monitor_connection())

            except Exception as conne:
                logger.exception(f"Error initializing Zyun connection: {conne}")
                # 清理资源
                for pc in self._pcs:
                    await pc.close()
                self._pcs.clear()

    async def _process_audio_track(self, track):
        """处理接收到的音频轨道"""
        while True:
            try:
                frame = await track.recv()
                logger.debug(f"receiver audio data process, pts: {frame.pts}, audio_buffer length: {self._stream_buffer.audio_buffer.qsize()}")
                await self._stream_buffer.audio_buffer.put(frame)
            except Exception as ate:
                logger.error(f"Error processing audio track: {ate}")
                break

    async def _process_video_track(self, track):
        """处理接收到的视频轨道"""
        while True:
            try:
                frame = await track.recv()
                logger.debug(f"receiver video data process, pts: {frame.pts}, video_buffer length: {self._stream_buffer.video_buffer.qsize()}")
                await self._stream_buffer.video_buffer.put(frame)
            except Exception as vte:
                logger.error(f"Error processing video track: {vte}")
                break


    async def _consume_and_process_audio(self):
        """消费并处理音频数据"""
        # await self._digital_person_resampler_event.wait()

        while True:
            try:
                audio_frame = await self._stream_buffer.audio_buffer.get()
                logger.debug(f"receiver audio data consume, pts: {audio_frame.pts}")
                # 输出给下一级
                if self._pipecat_resampler:
                    resampled_frames = self._pipecat_resampler.resample(audio_frame)
                    # self._pipecat_resampler_event.set()
                    for resampled_frame in resampled_frames:
                        await self.push_frame(
                            TTSAudioRawFrame(
                                audio=resampled_frame.to_ndarray().tobytes(),
                                sample_rate=self._pipecat_resampler.rate,
                                num_channels=1,
                            ),
                        )
                # 输出给sfu  48000/ 2
                else:
                    await self.push_frame(
                       TTSAudioRawFrame(
                            audio=audio_frame.to_ndarray().tobytes(),
                            sample_rate= audio_frame.sample_rate,
                            num_channels=1,
                        ), 
                    )
            except Exception as afe:
                logger.error(f"Error consuming audio: {afe}")
                await asyncio.sleep(0.1)

    async def _consume_and_process_video(self):
        """消费并处理视频数据"""
        # await self._digital_person_resampler_event.wait()
        while True:
            try:
                video_frame = await self._stream_buffer.video_buffer.get()
                logger.debug(f"receiver video data consume, pts: {video_frame.pts}")
                # 转换视频帧为 OutputImageRawFrame
                rgb_frame = video_frame.to_rgb().to_image()
                converted_frame = OutputImageRawFrame(
                    image=rgb_frame.tobytes(),
                    size=(video_frame.width, video_frame.height),
                    format="RGB",
                )
                converted_frame.pts = video_frame.pts
                await self.push_frame(converted_frame)
            except Exception as vfe:
                logger.error(f"Error consuming video: {vfe}")
                await asyncio.sleep(0.1)


    async def send_audio(self, audio_data: bytes):
        """发送音频数据到 Zyun 服务"""
        try:
            # 检查连接状态
            if not self._push_pc or self._push_pc.connectionState != "connected":
                logger.warning(f"Cannot send audio, connection state: {self._push_pc.connectionState if self._push_pc else 'None'}")
                return

            # 检查音频数据
            if not audio_data or len(audio_data) == 0:
                logger.warning("Empty audio data, skipping")
                return

            # 确保音频数据长度是偶数（16位采样）
            if len(audio_data) % 2 != 0:
                logger.warning(f"Audio data length {len(audio_data)} is not even, padding")
                audio_data += b'\x00'

            # 创建音频帧
            try:
                # 将字节转换为 numpy 数组
                audio_array = np.frombuffer(audio_data, dtype=np.int16)

                # 确保数组不为空
                if len(audio_array) == 0:
                    logger.warning("Empty audio array, skipping")
                    return

                # 创建帧
                audio_frame = AudioFrame.from_ndarray(
                    audio_array.reshape(1, -1),  # 确保是 2D 数组 [channels, samples]
                    layout="mono",
                    format="s16",
                )

                # 设置采样率
                audio_frame.sample_rate = self._audio_track.sample_rate

                # logger.debug(f"Sending audio frame with {len(audio_data)} bytes, samples={len(audio_array)}")

                # 添加帧到队列
                await self._audio_track.add_frame(audio_frame)

            except ValueError as ve:
                logger.error(f"ValueError creating audio frame: {ve}")
                # 可能是数据格式问题，尝试不同的方法
                try:
                    # 创建一个新的静默帧
                    silent_frame = AudioFrame(format="s16", layout="mono", samples=len(audio_data)//2)
                    # 直接复制数据
                    for p in silent_frame.planes:
                        p.update(audio_data[:p.buffer_size])
                    silent_frame.sample_rate = self._audio_track.sample_rate

                    await self._audio_track.add_frame(silent_frame)
                    logger.debug(f"Sent audio frame using alternative method")
                except Exception as alt_e:
                    logger.error(f"Alternative method also failed: {alt_e}")

        except Exception as sae:
            logger.error(f"Error sending audio to Zyun: {sae}")

    async def save_audio_to_wav_async(self, audio_data, sample_rate, channels=1, prefix="audio"):
        """异步保存音频文件"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.save_audio_to_wav,
            audio_data,
            sample_rate,
            channels,
            prefix
        )

    def save_audio_to_wav(self, audio_data: bytes, sample_rate: int, channels: int = 1, prefix="audio"):
        output_dir = Path("./audio_logs")
        output_dir.mkdir(exist_ok=True)

        timestamp = int(time.time())
        random_num = random.randint(1000, 9999)
        filename = f"{prefix}_{timestamp}_{random_num}.wav"
        filepath = output_dir / filename

        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        with wave.open(str(filepath), 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # 16位音频，每个样本2字节
            wf.setframerate(sample_rate)
            wf.writeframes(audio_array.tobytes())

        logger.info(f"Saved audio to {filepath}")
        return str(filepath)
    
    def save_buffer_to_wav(self, sample_rate: int = 24000, channels: int = 1, prefix="audio"):
        output_dir = Path("./audio_logs")
        output_dir.mkdir(exist_ok=True)

        timestamp = int(time.time())
        random_num = random.randint(1000, 9999)
        filename = f"{prefix}_{timestamp}_{random_num}.wav"
        filepath = output_dir / filename

        with wave.open(str(filepath), 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # 16位音频，每个样本2字节
            wf.setframerate(sample_rate)
            wf.writeframes(self.buffer) 

        logger.info(f"Saved audio to {filepath}")
        return str(filepath)

    async def _process_audio2buffer(self, audio_data: bytes):
        async with self.buffer_lock:  # 确保并发安全
            self.audio_buffer.extend(audio_data)

    async def _consume_audio_from_buffer(self):
        chunk_size = int(SAMPLE_RATE * AUDIO_PTIME)
        while True:
            async with self.buffer_lock:
                if len(self.audio_buffer) >= chunk_size * 2:
                    chunk = self.audio_buffer[:chunk_size * 2]  # s16取 640 字节, 320个采样点
                    self.audio_buffer = self.audio_buffer[chunk_size * 2:]  # 移除已读取部分
                    await self.send_audio(chunk)
                else:
                    chunk = None
                
                if chunk is None:
                    await asyncio.sleep(0.01) #避免忙等
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """处理帧"""
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            await self._start_connection()
        elif isinstance(frame, TTSAudioRawFrame):
            # 保存原始 TTS 音频数据
            # self.save_audio_to_wav(
            #     audio_data=frame.audio,
            #     sample_rate=frame.sample_rate,
            #     channels=frame.num_channels,
            #     prefix="tts_original"
            # )

            # 发送音频帧到 Zyun
            try:
                # 创建 PyAV 音频帧
                old_frame = AudioFrame.from_ndarray(
                    np.frombuffer(frame.audio, dtype=np.int16)[None, :],
                    layout="mono" if frame.num_channels == 1 else "stereo",
                )
                old_frame.sample_rate = frame.sample_rate

                # 初始化 pipecat 重采样器（用于重采样发送给pipecat）
                if self._pipecat_resampler is None:
                    self._pipecat_resampler = AudioResampler(
                        "s16", old_frame.layout, old_frame.sample_rate
                    )
                   

                # 用于重采样发送给 数字人
                if self._digital_person_resampler is None:
                    self._digital_person_resampler = AudioResampler(
                        "s16", "mono",SAMPLE_RATE 
                    )
                    # self._digital_person_resampler_event.set()

                # 目标采样点大小
                # 重采样到 Zyun 期望的格式（确保与 AudioTrack 一致）
                resampled_frames = self._digital_person_resampler.resample(old_frame)
                logger.info(f"resample frames")
                for resampled_frame in resampled_frames:
                    # 保存重采样后的音频数据
                    resampled_audio = resampled_frame.to_ndarray().astype(np.int16).tobytes()
                    
                    # self.save_audio_to_wav(
                    #     audio_data=resampled_audio,
                    #     sample_rate=self._audio_track.sample_rate,
                    #     channels=1,
                    #     prefix="tts_resampled"
                    # )
                    # self.buffer.extend(resampled_audio)
                    await self._process_audio2buffer(resampled_audio)
                    # await self.send_audio(resampled_audio)
            except Exception as pfe:
                logger.exception(f"{self} exception: {pfe}")
        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self._stop()
            await self.push_frame(frame, direction)
        elif isinstance(frame, StartInterruptionFrame):
            await self._clear_buffer()
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def _stop(self):
        """停止所有连接和任务"""
        # 关闭 WebRTC 连接
        for pc in self._pcs:
            await pc.close()
        self._pcs.clear()

        # 取消任务
        if self._audio_task:
            await self.cancel_task(self._audio_task)
            self._audio_task = None
        if self._video_task:
            await self.cancel_task(self._video_task)
            self._video_task = None

        # 重置状态
        self._initialized = False
        self._stream_buffer.clear_buffer()

    async def _clear_buffer(self):
        """清除缓冲区"""
        self._stream_buffer.clear_buffer()

    async def wait_for_connection(self, timeout=10):
        """等待 WebRTC 连接建立，超时返回 False"""
        if not self._push_pc:
            return False

        start_time = asyncio.get_event_loop().time()
        while (asyncio.get_event_loop().time() - start_time) < timeout:
            if self._push_pc.connectionState == "connected":
                return True
            await asyncio.sleep(1)
        return False

    async def _monitor_connection(self):
        """监控连接状态和队列状态"""
        while self._initialized:
            try:
                if self._push_pc:
                    # 检查连接状态
                    conn_state = self._push_pc.connectionState
                    ice_state = self._push_pc.iceConnectionState

                    logger.debug(f"Connection state: {conn_state}, ICE state: {ice_state}")

                    # 检查队列大小
                    if hasattr(self._audio_track, '_queue'):
                        queue_size = self._audio_track._queue.qsize()
                        logger.debug(f"Monitor: Audio queue size: {queue_size}")

                        # 如果队列持续增长，尝试清空队列
                        if queue_size > 10:
                            logger.warning(f"Queue size too large ({queue_size}), clearing excess frames")
                            # 保留最新的几帧
                            keep_frames = 3
                            to_remove = queue_size - keep_frames
                            if to_remove > 0:
                                for _ in range(to_remove):
                                    try:
                                        self._audio_track._queue.get_nowait()
                                    except asyncio.QueueEmpty:
                                        break
                                logger.info(f"Cleared {to_remove} frames from queue")

                        # 如果连接状态不是 connected 且队列很大，重置连接
                        if queue_size > 20 and conn_state != "connected":
                            logger.warning("Queue size too large and connection not established, resetting connection")
                            await self._stop()
                            await asyncio.sleep(1)
                            await self._start_connection()
            except Exception as e:
                logger.error(f"Error in connection monitor: {e}")

            await asyncio.sleep(1)  # 每秒检查一次

    async def _keep_audio_alive(self):
        """保持音频流活跃的任务"""
        import fractions

        # 等待连接建立
        while not self._push_pc or self._push_pc.connectionState != "connected":
            await asyncio.sleep(0.5)
            if not self._initialized:
                return

        logger.info("Starting audio keep-alive task")

        # 主循环
        while self._initialized and self._push_pc and self._push_pc.connectionState == "connected":
            try:
                # 检查队列
                if self._audio_track._queue.empty():
                    # 创建静默帧
                    silent_frame = AudioFrame(format="s16", layout="mono", samples=int(AUDIO_PTIME * SAMPLE_RATE))
                    for p in silent_frame.planes:
                        p.update(bytes(p.buffer_size))
                    silent_frame.sample_rate = SAMPLE_RATE

                    # 添加到队列
                    await self._audio_track.add_frame(silent_frame)
                    logger.debug("Added silence frame to keep stream alive")
            except Exception as e:
                logger.error(f"Error in audio keep-alive: {e}")

            await asyncio.sleep(0.1)  # 每100ms检查一次

    async def _wait_for_ice_complete(self, pc):
        """等待 ICE 收集完成"""
        waiter = asyncio.Future()

        @pc.on("icegatheringstatechange")
        def on_ice_gathering_state_change():
            if pc.iceGatheringState == "complete":
                waiter.set_result(None)

        if pc.iceGatheringState == "complete":
            waiter.set_result(None)

        # 设置超时
        try:
            await asyncio.wait_for(waiter, timeout=10)
        except asyncio.TimeoutError:
            logger.warning("ICE gathering timed out, proceeding anyway")

    async def _recover_connection(self):
        """尝试恢复失败的连接"""
        logger.info("Attempting to recover connection")

        # 等待一段时间
        await asyncio.sleep(2)

        # 关闭现有连接
        await self._stop()

        # 等待一段时间
        await asyncio.sleep(2)

        # 重新启动连接
        await self._start_connection()



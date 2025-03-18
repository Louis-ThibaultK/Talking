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

import asyncio
import io
import json
import logging
import threading
import time
from typing import Tuple, Dict, Optional, Set, Union
import wave
from av.frame import Frame
from av.packet import Packet
from av import AudioFrame
import fractions
import numpy as np

AUDIO_PTIME = 0.020  # 20ms audio packetization
VIDEO_CLOCK_RATE = 90000
VIDEO_PTIME = 1 / (12.5)  # 30fps
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)
SAMPLE_RATE = 48000
AUDIO_TIME_BASE = fractions.Fraction(1, SAMPLE_RATE)

#from aiortc.contrib.media import MediaPlayer, MediaRelay
#from aiortc.rtcrtpsender import RTCRtpSender
from aiortc import (
    MediaStreamTrack,
)

logging.basicConfig()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)  # 默认 INFO 级别
# logging.getLogger("aiortc.rtcrtpreceiver").setLevel(logging.DEBUG)  # 打印 RTP 接收日志
class AudioBuffer:
    """
    �~F�~N��~T��~H��~Z~D�~_��~Q�~U��~M��~]�~X�~H��~S�~F��~L��~@~B
    """
    def __init__(self):
        self.buffer = bytearray()
        self.num_channels = None
        self.sample_rate = None
        self.sample_width = None

    async def write(self, data, num_channels, sample_rate, sample_width):
        if self.num_channels is None:
            self.num_channels = num_channels
        if self.sample_rate is None:
            self.sample_rate = sample_rate
        if self.sample_width is None:
            self.sample_width = sample_width

        self.buffer.extend(data)

    def get_data(self):
        return self.buffer
class PlayerStreamTrack(MediaStreamTrack):
    """
    A video track that returns an animated flag.
    """

    def __init__(self, player, kind):
        super().__init__()  # don't forget this!
        self.kind = kind
        self._player = player
        self._queue = asyncio.Queue()
        self.timelist = [] #记录最近包的时间戳
        if self.kind == 'video':
            self.framecount = 0
            self.lasttime = time.perf_counter()
            self.totaltime = 0
        else:
            self.buffer = AudioBuffer()
    
    _start: float
    _timestamp: int

    async def next_timestamp(self) -> Tuple[int, fractions.Fraction]:
        if self.readyState != "live":
            raise Exception

        if self.kind == 'video':
            if hasattr(self, "_timestamp"):
                #self._timestamp = (time.time()-self._start) * VIDEO_CLOCK_RATE
                self._timestamp += int(VIDEO_PTIME * VIDEO_CLOCK_RATE)
                wait = self._start + (self._timestamp / VIDEO_CLOCK_RATE) - time.time()
                # wait = self.timelist[0] + len(self.timelist)*VIDEO_PTIME - time.time()               
                if wait>0:
                    await asyncio.sleep(wait)
                # if len(self.timelist)>=100:
                #     self.timelist.pop(0)
                # self.timelist.append(time.time())
            else:
                self._start = time.time()
                self._timestamp = 0
                self.timelist.append(self._start)
                print('video start:',self._start)
            return self._timestamp, VIDEO_TIME_BASE
        else: #audio
            if hasattr(self, "_timestamp"):
                #self._timestamp = (time.time()-self._start) * SAMPLE_RATE
                self._timestamp += int(AUDIO_PTIME * SAMPLE_RATE)
                wait = self._start + (self._timestamp / SAMPLE_RATE) - time.time()
                # wait = self.timelist[0] + len(self.timelist)*AUDIO_PTIME - time.time()
                if wait>0:
                    await asyncio.sleep(wait)
                # if len(self.timelist)>=200:
                #     self.timelist.pop(0)
                #     self.timelist.pop(0)
                # self.timelist.append(time.time())
            else:
                self._start = time.time()
                self._timestamp = 0
                self.timelist.append(self._start)
                print('audio start:',self._start)
            return self._timestamp, AUDIO_TIME_BASE

    async def recv(self) -> Union[Frame, Packet]:
        # frame = self.frames[self.counter % 30]            
        self._player._start(self)
        # if self.kind == 'video':
        #     frame = await self._queue.get()
        # else: #audio
        #     if hasattr(self, "_timestamp"):
        #         wait = self._start + self._timestamp / SAMPLE_RATE + AUDIO_PTIME - time.time()
        #         if wait>0:
        #             await asyncio.sleep(wait)
        #         if self._queue.qsize()<1:
        #             #frame = AudioFrame(format='s16', layout='mono', samples=320)
        #             audio = np.zeros((1, 320), dtype=np.int16)
        #             frame = AudioFrame.from_ndarray(audio, layout='mono', format='s16')
        #             frame.sample_rate=16000
        #         else:
        #             frame = await self._queue.get()
        #     else:
        #         frame = await self._queue.get()
        frame = await self._queue.get()
        if self.kind == 'audio':
            data = frame.to_ndarray().tobytes()
            await self.buffer.write(data, 1, 16000, 2)
        pts, time_base = await self.next_timestamp()

        frame.pts = pts
        frame.time_base = time_base
        if frame is None:
            self.stop()
            raise Exception
        if self.kind == 'video':
            self.totaltime += (time.perf_counter() - self.lasttime)
            self.framecount += 1
            self.lasttime = time.perf_counter()
            if self.framecount==100:
                print(f"------actual avg final fps:{self.framecount/self.totaltime:.4f}")
                self.framecount = 0
                self.totaltime=0
        return frame
    
    def stop(self):
        super().stop()
        if self._player is not None:
            self._player._stop(self)
            self._player = None


async def player_worker_thread(
    quit_event,
    loop,
    container,
    audio_track,
    video_track
):
    await container.render(quit_event,loop,audio_track,video_track)

def save_audio_to_file(frames, filename):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(frames.num_channels)  # Mono
        wf.setsampwidth(frames.sample_width)  # 16-bit PCM
        wf.setframerate(frames.sample_rate)  # Sample rate
        # for frame in frames:
        wf.writeframes(frames.get_data())
        print(f"Saved {len(frames.get_data())} audio frames to {filename}.")

class HumanPlayer:

    def __init__(
        self, nerfreal, format=None, options=None, timeout=None, loop=False, decode=True
    ):
        self.__thread: Optional[threading.Thread] = None
        self.__thread_quit: Optional[threading.Event] = None

        # examine streams
        self.__started: Set[PlayerStreamTrack] = set()
        self.__audio: Optional[PlayerStreamTrack] = None
        self.__video: Optional[PlayerStreamTrack] = None

        self.__audio = PlayerStreamTrack(self, kind="audio")
        self.__video = PlayerStreamTrack(self, kind="video")
        self._loop = loop

        self.__container = nerfreal


    @property
    def audio(self) -> MediaStreamTrack:
        """
        A :class:`aiortc.MediaStreamTrack` instance if the file contains audio.
        """
        return self.__audio
    
    def get_loop(self):
        return self._loop

    @property
    def video(self) -> MediaStreamTrack:
        """
        A :class:`aiortc.MediaStreamTrack` instance if the file contains video.
        """
        return self.__video

    def _start(self, track: PlayerStreamTrack) -> None:
        self.__started.add(track)
        if self.__thread is None:
            self.__log_debug("Starting worker thread")
            self.__thread_quit = threading.Event()
            # self.__thread = threading.Thread(
            #     name="media-player",
            #     target=player_worker_thread,
            #     args=(
            #         self.__thread_quit,
            #         asyncio.get_event_loop(),
            #         self.__container,
            #         self.__audio,
            #         self.__video                   
            #     ),
            # )
            # self.__thread.start()
            self.__thread = asyncio.create_task(player_worker_thread(self.__thread_quit, asyncio.get_event_loop(), self.__container, self.__audio, self.__video))

    def _stop(self, track: PlayerStreamTrack) -> None:
        if track == self.__audio:
            print("save push wav")
            save_audio_to_file(self.__audio.buffer, "push.wav")
        self.__started.discard(track)

        if not self.__started and self.__thread is not None:
            self.__log_debug("Stopping worker thread")
            self.__thread_quit.set()
            # self.__thread.join()
            self.__thread.cancel()  # 取消任务
            self.__thread = None

        if not self.__started and self.__container is not None:
            #self.__container.close()
            self.__container = None

    def __log_debug(self, msg: str, *args) -> None:
        logger.debug(f"HumanPlayer {msg}", *args)


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

import time
import numpy as np

import queue
from queue import Queue
import torch.multiprocessing as mp
import opuslib
from io import BytesIO
import soundfile as sf
import resampy


class BaseASR:
    def __init__(self, opt, parent=None):
        self.opt = opt
        self.parent = parent

        self.fps = opt.fps # 20 ms per frame
        self.sample_rate = 16000
        self.chunk = self.sample_rate // self.fps # 320 samples per chunk (20ms * 16000 / 1000)
        self.queue = Queue()
        self.output_queue = mp.Queue()

        self.batch_size = opt.batch_size

        self.frames = []
        self.stride_left_size = opt.l
        self.stride_right_size = opt.r
        #self.context_size = 10
        self.feat_queue = mp.Queue(2)

        #self.warm_up()

    def flush_talk(self):
        self.queue.queue.clear()

    def put_audio_frame(self,audio_chunk): #16khz 20ms pcm
        self.queue.put(audio_chunk)

    def get_audio_frame(self):        
        try:
            frame = self.queue.get(block=True,timeout=0.01)
            type = 0
            #print(f'[INFO] get frame {frame.shape}')
        except queue.Empty:
            if self.parent and self.parent.curr_state>1: #播放自定义音频
                frame = self.parent.get_audio_stream(self.parent.curr_state)
                type = self.parent.curr_state
            else:
                frame = np.zeros(self.chunk, dtype=np.float32)
                type = 1

        return frame,type 

    def is_audio_frame_empty(self)->bool:
        return self.queue.empty()

    def get_audio_out(self):  #get origin audio pcm to nerf
        return self.output_queue.get()
    
    def warm_up(self):
        for _ in range(self.stride_left_size + self.stride_right_size):
            audio_frame,type=self.get_audio_frame()
            self.frames.append(audio_frame)
            self.output_queue.put((audio_frame,type))
        for _ in range(self.stride_left_size):
            self.output_queue.get()

    def run_step(self):
        pass

    def get_next_feat(self,block,timeout):        
        return self.feat_queue.get(block,timeout)
    
    def channel_switch(self, frame):

        audio_data = frame.to_ndarray()
        print("old audio data:", audio_data[0][0], audio_data[0][1], audio_data[0][2], audio_data[0][3])
        if audio_data.shape[0] == 1 and len(frame.layout.channels) == 2:
            audio_data[0] = audio_data[0][0::2]
            audio_data[1] = audio_data[0][1::2]
        print("new audio data:", audio_data[0][0], audio_data[1][0], audio_data[0][1], audio_data[1][1])
        if audio_data.shape[0] == 2:
            # 将 2 通道数据求平均，合并为 1 个通道（Mono）
            audio_data = np.mean(audio_data, axis=0)
        print("audio_data:", audio_data.shape, frame.sample_rate)

        return audio_data, frame.sample_rate
    
    def create_bytes_stream(self, stream, sample_rate = None):
        #byte_stream=BytesIO(buffer)
        #必须wav格式
        if sample_rate is None:
            stream, sample_rate = sf.read(stream) # [T*sample_rate,] float64
        print(f'[INFO]put audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            print(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]
    
        if sample_rate != self.sample_rate and stream.shape[0]>0:
            print(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream
    
   
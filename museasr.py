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

from io import BytesIO
import time
import numpy as np

import queue
from queue import Queue
#import multiprocessing as mp
from baseasr import BaseASR
from musetalk.whisper.audio2feature import Audio2Feature

class MuseASR(BaseASR):
    def __init__(self, opt, parent,audio_processor:Audio2Feature):
        super().__init__(opt,parent)
        self.audio_processor = audio_processor
        self.audio_buffer = []

    def run_step(self):
        ############################################## extract audio feature ##############################################
        start_time = time.time()
        for _ in range(self.batch_size*2):
            audio_frame,type=self.get_audio_frame()
            self.frames.append(audio_frame)
            self.output_queue.put((audio_frame,type))
        
        if len(self.frames) <= self.stride_left_size + self.stride_right_size:
            return
        
        inputs = np.concatenate(self.frames) # [N * chunk]
        whisper_feature = self.audio_processor.audio2feat(inputs)
        # for feature in whisper_feature:
        #     self.audio_feats.append(feature)        
        #print(f"processing audio costs {(time.time() - start_time) * 1000}ms, inputs shape:{inputs.shape} whisper_feature len:{len(whisper_feature)}")
        whisper_chunks = self.audio_processor.feature2chunks(feature_array=whisper_feature,fps=self.fps/2,batch_size=self.batch_size,start=self.stride_left_size/2 )
        #print(f"whisper_chunks len:{len(whisper_chunks)},self.audio_feats len:{len(self.audio_feats)},self.output_queue len:{self.output_queue.qsize()}")
        #self.audio_feats = self.audio_feats[-(self.stride_left_size + self.stride_right_size):]
        self.feat_queue.put(whisper_chunks)
        # discard the old part to save memory
        self.frames = self.frames[-(self.stride_left_size + self.stride_right_size):]

    def put_frame(self,audio_stream, sample_rate):
        if audio_stream is not None and len(audio_stream)>0: 
            self.audio_buffer.append(audio_stream)
            if len(self.audio_buffer) >= 5:
                merged_audio = np.concatenate(self.audio_buffer, axis=0)
                merged_audio = merged_audio.flatten()
                stream = self.create_bytes_stream(merged_audio, sample_rate)
                streamlen = stream.shape[0]
                idx=0
                while streamlen >= self.chunk:
                        self.put_audio_frame(stream[idx:idx+self.chunk])
                        streamlen -= self.chunk
                        idx += self.chunk

                self.audio_buffer.clear()
    
    def stream_tts(self,audio_stream):
        chunk_count = 0
        for chunk in audio_stream:
            if chunk is not None and len(chunk)>0:          
                #stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                #stream = resampy.resample(x=stream, sr_orig=32000, sr_new=self.sample_rate)
                byte_stream=BytesIO(chunk)
                stream = self.create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx=0
                while streamlen >= self.chunk:
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk])
                    streamlen -= self.chunk
                    idx += self.chunk 
                    self.chunk_count += 1
        return chunk_count
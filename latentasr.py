from io import BytesIO
import time
import numpy as np

import queue
from queue import Queue
#import multiprocessing as mp
from baseasr import BaseASR
from latentsync.latentsync.whisper import Audio2Feature

class LatentsyncASR(BaseASR):
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
        whisper_chunks = self.audio_processor.feature2chunks(feature_array=whisper_feature,fps=self.fps/2,batch_size=self.batch_size,start=self.stride_left_size/2 )
        #print(f"whisper_chunks len:{len(whisper_chunks)},self.audio_feats len:{len(self.audio_feats)},self.output_queue len:{self.output_queue.qsize()}")
        #self.audio_feats = self.audio_feats[-(self.stride_left_size + self.stride_right_size):]
        self.feat_queue.put(whisper_chunks)
        # discard the old part to save memory
        self.frames = self.frames[-(self.stride_left_size + self.stride_right_size):]

    def put_frame(self,audio_stream, sample_rate):
        if audio_stream is not None and len(audio_stream)>0: 
            self.audio_buffer.append(audio_stream)
            if len(self.audio_buffer) >= 10:
                merged_audio = np.concatenate(self.audio_buffer, axis=0)
                merged_audio = merged_audio.flatten()
                stream = self.create_bytes_stream(merged_audio, sample_rate)
                streamlen = stream.shape[0]
                idx=0
                while streamlen >= self.chunk:
                        self.parent.put_audio_frame(stream[idx:idx+self.chunk])
                        streamlen -= self.chunk
                        idx += self.chunk 

                self.audio_buffer.clear()
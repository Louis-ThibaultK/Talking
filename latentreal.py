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

import torch
import numpy as np

#from .utils import *

import time

import queue
from threading import Thread
import torch.multiprocessing as mp
from museasr import MuseASR
import asyncio
from av import AudioFrame, VideoFrame
from basereal import BaseReal

#新添加
from basereal import BaseReal
from latentsync.pipeline import load_all_model, _load_avatar, Pipeline
from typing import Optional
from latentasr import LatentsyncASR
# from scipy.signal import resample
import resampy

def load_model():
    # load model weights
    audio_processor,vae, unet, pe = load_all_model()
    
    return vae, unet, pe, audio_processor

def load_avatar(video_path, pipeline: Pipeline):
    pipeline.warm_up(256, 256)
    return _load_avatar(video_path, pipeline=pipeline)

def warm_up(pe: Pipeline,  height: Optional[int] = None,
                width: Optional[int] = None,):
    pe.warm_up(height, width)


@torch.no_grad()
def resample_pcm_scipy(pcm_chunks, input_rate=16000, output_rate=48000, target_chunks=50, target_size=960):
    """
    用 scipy 高效重采样 32 个 16kHz PCM 数据包 → 50 个 48kHz PCM 数据包
    """
    # Step 1: 拼接 32 个 PCM 数据包
    pcm_data = np.concatenate(pcm_chunks, axis=0)  # shape = (10240,)

    # Step 2: 进行 16kHz → 48kHz 重采样
    # resampled_data = resampy.resample(x=pcm_data, sr_orig=9600, sr_new=target_chunks * target_size)
    # resampled_data = resample(pcm_data, target_chunks * target_size)  # shape = (48000,)

    # stereo_data = np.stack([resampled_data, resampled_data], axis=1) 
    # Step 3: 重新分割成 50 个数据包，每个包长 960
    # resampled_chunks = np.split(stereo_data, target_chunks, axis = 0)
    resampled_chunks = np.split(pcm_data, target_chunks, axis = 0)
    

    return resampled_chunks


class LatentReal(BaseReal):
    @torch.no_grad()
    def __init__(self, opt, model, avatar):
        super().__init__(opt)
        #self.opt = opt # shared with the trainer's opt to support in-place modification of rendering parameters.

        # self.fps = opt.fps # 20 ms per frame

        self.batch_size = opt.batch_size
        self.res_frame_queue = mp.Queue(self.batch_size*2)
        self.audio_frame_queue = mp.Queue(100)

        self.vae, self.unet, self.pe, self.audio_processor = model
        self.faces, self.original_video_frames, self.boxes, self.affine_matrices = avatar

        self.asr = LatentsyncASR(opt,self,self.audio_processor)

        self.render_event = mp.Event()

        self.index = 0

    def __del__(self):
        print(f'latentreal({self.sessionid}) delete')

    @torch.no_grad()
    def inference(self, pipeline: Pipeline, faces, original_video_frames, boxes, affine_matrices,
                render_event,batch_size,audio_feat_queue,audio_out_queue,res_frame_queue, audio_frame_queue
                ): #vae, unet, pe,timesteps
        
        count=0
        counttime=0
        length = len(original_video_frames)
        print('start inference')
        while render_event.is_set():
            try:
                whisper_chunks = audio_feat_queue.get(block=True, timeout=0.5)
            except queue.Empty:
                continue
            is_all_silence=True
            audio_frames = []
            for _ in range(batch_size*4):
                frame,type = audio_out_queue.get()
                audio_frames.append((frame,type))
                if type==0:
                    is_all_silence=False

            # audio_slice = [frame for frame, _ in audio_frames]
            # audios = resample_pcm_scipy(audio_slice)
            # print("333333", len(audios))

            # resample_frames= []
            # for _, audio_frame in enumerate(audios):
            #     if is_all_silence:
            #         resample_frames.append((audio_frame, 1))
            #     else:
            #         resample_frames.append((audio_frame, 0))

            t=time.perf_counter()
            split_faces = []
            split_boxes = []
            split_affine_matrices = []
            split_video_frames = []
            for i in range(batch_size):
                id = self.mirror_index(length, self.index + i)
                split_faces.append(faces[id])
                split_boxes.append(boxes[id])
                split_affine_matrices.append(affine_matrices[id])
                split_video_frames.append(original_video_frames[id])
            
            split_faces = torch.stack(split_faces) 
            split_video_frames = np.array(split_video_frames)
            # if is_all_silence:
            #     for i in range(batch_size):
            #         res_frame_queue.put((split_video_frames[i], audio_frames[i*2:i*2+2]))
            #         index = index + 1
            #     return
            start_time = time.perf_counter()
            videos = pipeline.inference(whisper_chunks, split_faces, split_video_frames, split_boxes, split_affine_matrices, num_frames=batch_size)
            end_time = time.perf_counter()
            print(f"inference代码执行时间: {end_time - start_time:.6f} 秒")

            self.index += batch_size
            counttime += (time.perf_counter() - t)
            count += batch_size

            if count>=100:
                print(f"------actual avg infer fps:{count/counttime:.4f}")
                count=0
                counttime=0

            for i,res_frame in enumerate(videos):
                #self.__pushmedia(res_frame,loop,audio_track,video_track)
                res_frame_queue.put((res_frame, audio_frames[i*2:i*2+2]))

            for _, audio_frame in enumerate(audio_frames):
                audio_frame_queue.put(audio_frame)
                
        print('latentreal inference processor stop')

    def process_frames(self,quit_event,loop=None,audio_track=None,video_track=None):
        res_frame = np.zeros((512, 512, 3), dtype=np.uint8)
        while not quit_event.is_set():
            try:
                res_frame,audio_frames = self.res_frame_queue.get(block=True, timeout=0.075)
                print("8888888")
            except queue.Empty:
                print("7777777")
                length = len(self.original_video_frames)
                id = self.mirror_index(length, self.index) 
                res_frame = self.original_video_frames[id]
                self.index += 1

            image = res_frame #(outputs['image'] * 255).astype(np.uint8)
            new_frame = VideoFrame.from_ndarray(image, format="rgb24")
            asyncio.run_coroutine_threadsafe(video_track._queue.put(new_frame), loop)
            self.record_video_data(image)
            #self.recordq_video.put(new_frame)  

            # if audio_frames[0][1]!=0 and audio_frames[1][1]!=0:
            #     continue 

            # for audio_frame in audio_frames:
            #     frame, type = audio_frame
            #     frame = frame.astype(np.int16)
                
            #     new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
            #     new_frame.planes[0].update(frame.tobytes())
            #     new_frame.sample_rate=16000
            #     # if audio_track._queue.qsize()>10:
            #     #     time.sleep(0.1)
            #     asyncio.run_coroutine_threadsafe(audio_track._queue.put(new_frame), loop)
            #     self.record_audio_data(frame)
            #     #self.recordq_audio.put(new_frame)
        print('latentreal process_frames thread stop') 

    def process_audio_frame(self, quit_event, loop = None, audio_track = None):
        while not quit_event.is_set():
            try:
                audio_frame = self.audio_frame_queue.get(block=True, timeout=0.02)
                print("999999")
            except queue.Empty:
                continue
            frame, type = audio_frame
            frame = frame.astype(np.int16)
        #    stereo 
            new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
            # new_frame.planes[0].update(frame.flatten().tobytes())
            new_frame.planes[0].update(frame.tobytes())
            new_frame.sample_rate=16000
            # if audio_track._queue.qsize()>10:
            #     time.sleep(0.1)
            asyncio.run_coroutine_threadsafe(audio_track._queue.put(new_frame), loop)
            self.record_audio_data(frame)
            #self.recordq_audio.put(new_frame)

    async def run_step(self, quit_event):
        while not quit_event.is_set(): #todo
            # update texture every frame
            # audio stream thread...
            t = time.perf_counter()
            await self.asr.run_step() 
        self.render_event.clear() #end infer process render 

    def render(self,quit_event,loop=None,audio_track=None,video_track=None):
        #if self.opt.asr:
        #     self.asr.warm_up()

        self.init_customindex()
        process_thread = Thread(target=self.process_frames, args=(quit_event,loop,audio_track,video_track))
        process_thread.start()
        Thread(target=self.process_audio_frame,  args=(quit_event,loop,audio_track)).start()

        self.render_event.set() #start infer process render
        Thread(target=self.inference, args=(self.pe, self.faces, self.original_video_frames, self.boxes, self.affine_matrices, self.render_event,
                                       self.batch_size, self.asr.feat_queue, self.asr.output_queue, self.res_frame_queue, self.audio_frame_queue)).start() #mp.Process

        
        # self.asr.loop.run_until_complete(self.run_step(quit_event))
        # lp.run_forever()

        #_totalframe=0
        while not quit_event.is_set(): #todo
            # update texture every frame
            # audio stream thread...
            # t = time.perf_counter()
            self.asr.run_step()
   
            # if video_track._queue.qsize()>=1.5*self.opt.batch_size:
            #     print('sleep qsize=',video_track._queue.qsize())
            #     time.sleep(0.06*video_track._queue.qsize()*0.8)
            # print("video_track buffer length:", video_track._queue.qsize())
            # time.sleep(0.1)
   
        # self.render_event.clear() #end infer process render
        # print('latentreal thread stop')
            
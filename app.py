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

# server.py
from flask import Flask, render_template,send_from_directory,request, jsonify
import json
#import gevent
#from gevent import pywsgi
#from geventwebsocket.handler import WebSocketHandler
import numpy as np
from threading import Thread,Event
#import multiprocessing
import torch.multiprocessing as mp

from aiohttp import web
import aiohttp
import aiohttp_cors
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.rtcrtpsender import RTCRtpSender
from webrtc import HumanPlayer, PlayerStreamTrack
from av import AudioFrame, VideoFrame

import argparse
import random

import asyncio
import torch


app = Flask(__name__)
#sockets = Sockets(app)
nerfreals = {}
opt = None
model = None
avatar = None
status = False
message_queue = None
player = None
      

#####webrtc###############################
pcs = set()

def randN(N):
    '''生成长度为 N的随机数 '''
    min = pow(10, N - 1)
    max = pow(10, N)
    return random.randint(min, max - 1)

def build_nerfreal(sessionid):
    opt.sessionid=sessionid
    if opt.model == 'musetalk':
        from musereal import MuseReal
        nerfreal = MuseReal(opt,model,avatar)
    return nerfreal

#@app.route('/offer', methods=['POST'])
async def offer(request):
    params = await request.json()
    pull_offer = RTCSessionDescription(sdp=params["push_sdp"], type=params["type"])
    print("pull offer:", pull_offer)
    push_offer = RTCSessionDescription(sdp=params["pull_sdp"], type=params["type"]) 
    print("push offer:", push_offer)

    if len(nerfreals) >= opt.max_session:
        print('reach max session')
        return -1
    
    sessionid = randN(6) #len(nerfreals)
    sessionid = 0
    print('sessionid=',sessionid)
    nerfreals[sessionid] = None
    nerfreal = await asyncio.get_event_loop().run_in_executor(None, build_nerfreal,sessionid)
    nerfreals[sessionid] = nerfreal
    
    push_pc = RTCPeerConnection()
    pull_pc = RTCPeerConnection() 
    pcs.add(push_pc)
    pcs.add(pull_pc)
    player = HumanPlayer(nerfreals[sessionid])

    @push_pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("push connection state is %s" % push_pc.connectionState)
        if push_pc.connectionState == "failed":
            await push_pc.close()
            pcs.discard(push_pc)
            del nerfreals[sessionid]
        if push_pc.connectionState == "closed":
            pcs.discard(push_pc)
            del nerfreals[sessionid]

    @pull_pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("pull connection state is %s" % pull_pc.connectionState)
        if pull_pc.connectionState == "failed":
            await pull_pc.close()
            pcs.discard(pull_pc)
        if pull_pc.connectionState == "closed":
            pcs.discard(pull_pc)

    @push_pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        print(f"push ICE connection state is {push_pc.iceConnectionState}")
        if push_pc.iceConnectionState == "failed":
            print("ICE connection failed.")
        elif push_pc.iceConnectionState == "disconnected":
            print("ICE connection disconnected.")
    
    @pull_pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        print(f"push ICE connection state is {pull_pc.iceConnectionState}")
        if pull_pc.iceConnectionState == "failed":
            print("ICE connection failed.")
        elif pull_pc.iceConnectionState == "disconnected":
            print("ICE connection disconnected.")

    #拉取音频流
    async def on_track(track):
        print("Track received", track.kind, track.id)
        if track.kind == "audio":
            while True:
                frame = await track.recv()
                # 这里可以处理音频数据，例如进行转换、保存等
                # print(f"Received audio frame with timestamp {frame.time}")
                pcm_frame, sample_rate = nerfreal.asr.decode_opus_to_pcm(frame)
                nerfreal.asr.put_frame(pcm_frame, sample_rate)

    pull_pc.on("track", on_track)
    
    #拉流
    pull_pc.addTransceiver("audio", direction="recvonly")
    #推流
    audio_transceiver = push_pc.addTransceiver(player.audio, direction="sendonly")
    video_transceiver = push_pc.addTransceiver(player.video, direction="sendonly")

    capabilities = RTCRtpSender.getCapabilities("video")
    preferences = list(filter(lambda x: x.name == "H264", capabilities.codecs))
    preferences += list(filter(lambda x: x.name == "VP8", capabilities.codecs))
    preferences += list(filter(lambda x: x.name == "rtx", capabilities.codecs))
    transceiver = push_pc.getTransceivers()[1]
    transceiver.setCodecPreferences(preferences)

    await push_pc.setRemoteDescription(push_offer)

    await pull_pc.setRemoteDescription(pull_offer)
   

    push_answer = await push_pc.createAnswer()
    await push_pc.setLocalDescription(push_answer)
    print("push answer", push_answer)

    pull_answer = await pull_pc.createAnswer()
    await pull_pc.setLocalDescription(pull_answer)
    print("pull answer", pull_answer)

    #return jsonify({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"push_sdp": push_pc.localDescription.sdp, "pull_sdp": pull_pc.localDescription.sdp,
             "type": push_pc.localDescription.type, "sessionid":sessionid}
        ),
    )

# 将 VideoFrame 和 AudioFrame 转换为字节流
def frame_to_bytes(frame):
    if isinstance(frame, VideoFrame):
        # 对于 VideoFrame，提取 NumPy 数组并转化为字节流
        return frame.to_ndarray(format="bgr24").tobytes()
    elif isinstance(frame, AudioFrame):
        # 对于 AudioFrame，直接提取 PCM 数据并转化为字节流
        return frame.to_ndarray().tobytes()

# 异步发送数据
async def stream_pcm(frames, url):
    # 将 frame 数据转换为字节流
    
    async with aiohttp.ClientSession() as session:
        video_combine = bytearray()
        audio_combine = bytearray()
        for video_frame, audio_frame in frames:
            video_data = frame_to_bytes(video_frame)
            video_combine.extend(video_data)
            audio_data = frame_to_bytes(audio_frame)
            audio_combine.extend(audio_data)

        async with session.post(url + "/video", data=video_combine) as response:
            print(f"Server response: {response.status}")
            
        async with session.post(url + "/audio", data=audio_combine) as response:
            print(f"Audio response status: {response.status}")

@app.route('/start', methods=['POST'])
async def start(request):
    params = await request.json()
    url = params['url']
    global nerfreals
    if nerfreals.get(0) is None:
        sessionid = 0
        nerfreal = await asyncio.get_event_loop().run_in_executor(None, build_nerfreal,sessionid)
        nerfreals[sessionid] = nerfreal

    def receive_stream():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        global player
        player = HumanPlayer(nerfreals[0], loop=loop)
        player._start(player.audio)
        player._start(player.video)
        frames = []
        while True:
            audio_frame = player.audio._queue.get()
            video_frame = player.video._queue.get()
            frames.append((audio_frame, video_frame))
            if len(frames) >= 10:
                stream_pcm(frames, url)
                frames.clear()

    rendthrd = Thread(target=receive_stream, args=())
    rendthrd.start()
    return "Start successfully", 200 
    

@app.route('/humanaudio', methods=['POST'])
async def humanaudio():
    global player
    loop = player.get_loop()

    frame_count = asyncio.run_coroutine_threadsafe(nerfreals[0].asr.stream_tts(request.stream) , loop) 
    return "Start successfully", 200 
    

async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

async def post(url,data):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url,data=data) as response:
                return await response.text()
    except aiohttp.ClientError as e:
        print(f'Error: {e}')

async def run(push_url,sessionid):
    nerfreal = await asyncio.get_event_loop().run_in_executor(None, build_nerfreal,sessionid)
    nerfreals[sessionid] = nerfreal

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("push Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    player = HumanPlayer(nerfreals[sessionid])
    audio_sender = pc.addTrack(player.audio)
    video_sender = pc.addTrack(player.video)

    await pc.setLocalDescription(await pc.createOffer())
    answer = await post(push_url,pc.localDescription.sdp)
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer,type='answer'))


##########################################
# os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# os.environ['MULTIPROCESSING_METHOD'] = 'forkserver'                                                    
if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose', type=str, default="data/data_kf.json", help="transforms.json, pose source")
    parser.add_argument('--au', type=str, default="data/au.csv", help="eye blink area")
    parser.add_argument('--torso_imgs', type=str, default="", help="torso images path")

    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --exp_eye")

    parser.add_argument('--data_range', type=int, nargs='*', default=[0, -1], help="data range to use")
    parser.add_argument('--workspace', type=str, default='data/video')
    parser.add_argument('--seed', type=int, default=0)

    ### training options
    parser.add_argument('--ckpt', type=str, default='data/pretrained/ngp_kf.pth')
   
    parser.add_argument('--num_rays', type=int, default=4096 * 16, help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=16, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=16, help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")

    ### loss set
    parser.add_argument('--warmup_step', type=int, default=10000, help="warm up steps")
    parser.add_argument('--amb_aud_loss', type=int, default=1, help="use ambient aud loss")
    parser.add_argument('--amb_eye_loss', type=int, default=1, help="use ambient eye loss")
    parser.add_argument('--unc_loss', type=int, default=1, help="use uncertainty loss")
    parser.add_argument('--lambda_amb', type=float, default=1e-4, help="lambda for ambient loss")

    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    
    parser.add_argument('--bg_img', type=str, default='white', help="background image")
    parser.add_argument('--fbg', action='store_true', help="frame-wise bg")
    parser.add_argument('--exp_eye', action='store_true', help="explicitly control the eyes")
    parser.add_argument('--fix_eye', type=float, default=-1, help="fixed eye area, negative to disable, set to 0-0.3 for a reasonable eye")
    parser.add_argument('--smooth_eye', action='store_true', help="smooth the eye area sequence")

    parser.add_argument('--torso_shrink', type=float, default=0.8, help="shrink bg coords to allow more flexibility in deform")

    ### dataset options
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', type=int, default=0, help="0 means load data from disk on-the-fly, 1 means preload to CPU, 2 means GPU.")
    # (the default value is for the fox dataset)
    parser.add_argument('--bound', type=float, default=1, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=4, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1/256, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.05, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied (sigma)")
    parser.add_argument('--density_thresh_torso', type=float, default=0.01, help="threshold for density grid to be occupied (alpha)")
    parser.add_argument('--patch_size', type=int, default=1, help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")

    parser.add_argument('--init_lips', action='store_true', help="init lips region")
    parser.add_argument('--finetune_lips', action='store_true', help="use LPIPS and landmarks to fine tune lips region")
    parser.add_argument('--smooth_lips', action='store_true', help="smooth the enc_a in a exponential decay way...")

    parser.add_argument('--torso', action='store_true', help="fix head and train torso")
    parser.add_argument('--head_ckpt', type=str, default='', help="head model")

    ### else
    parser.add_argument('--att', type=int, default=2, help="audio attention mode (0 = turn off, 1 = left-direction, 2 = bi-direction)")
    parser.add_argument('--aud', type=str, default='', help="audio source (empty will load the default, else should be a path to a npy file)")
    parser.add_argument('--emb', action='store_true', help="use audio class + embedding instead of logits")

    parser.add_argument('--ind_dim', type=int, default=4, help="individual code dim, 0 to turn off")
    parser.add_argument('--ind_num', type=int, default=10000, help="number of individual codes, should be larger than training dataset size")

    parser.add_argument('--ind_dim_torso', type=int, default=8, help="individual code dim, 0 to turn off")

    parser.add_argument('--amb_dim', type=int, default=2, help="ambient dimension")
    parser.add_argument('--part', action='store_true', help="use partial training data (1/10)")
    parser.add_argument('--part2', action='store_true', help="use partial training data (first 15s)")

    parser.add_argument('--train_camera', action='store_true', help="optimize camera pose")
    parser.add_argument('--smooth_path', action='store_true', help="brute-force smooth camera pose trajectory with a window size")
    parser.add_argument('--smooth_path_window', type=int, default=7, help="smoothing window size")

    # asr
    parser.add_argument('--asr', action='store_true', help="load asr for real-time app")
    parser.add_argument('--asr_wav', type=str, default='', help="load the wav and use as input")
    parser.add_argument('--asr_play', action='store_true', help="play out the audio")

    #parser.add_argument('--asr_model', type=str, default='deepspeech')
    parser.add_argument('--asr_model', type=str, default='cpierse/wav2vec2-large-xlsr-53-esperanto') #
    # parser.add_argument('--asr_model', type=str, default='facebook/wav2vec2-large-960h-lv60-self')
    # parser.add_argument('--asr_model', type=str, default='facebook/hubert-large-ls960-ft')

    parser.add_argument('--asr_save_feats', action='store_true')
    # audio FPS
    parser.add_argument('--fps', type=int, default=50)
    # sliding window left-middle-right length (unit: 20ms)
    parser.add_argument('-l', type=int, default=10)
    parser.add_argument('-m', type=int, default=8)
    parser.add_argument('-r', type=int, default=10)

    parser.add_argument('--fullbody', action='store_true', help="fullbody human")
    parser.add_argument('--fullbody_img', type=str, default='data/fullbody/img')
    parser.add_argument('--fullbody_width', type=int, default=580)
    parser.add_argument('--fullbody_height', type=int, default=1080)
    parser.add_argument('--fullbody_offset_x', type=int, default=0)
    parser.add_argument('--fullbody_offset_y', type=int, default=0)

    #musetalk opt
    parser.add_argument('--avatar_id', type=str, default='avator_1')
    parser.add_argument('--bbox_shift', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)

    # parser.add_argument('--customvideo', action='store_true', help="custom video")
    # parser.add_argument('--customvideo_img', type=str, default='data/customvideo/img')
    # parser.add_argument('--customvideo_imgnum', type=int, default=1)

    parser.add_argument('--customvideo_config', type=str, default='')

    parser.add_argument('--model', type=str, default='ernerf') #musetalk wav2lip

    parser.add_argument('--transport', type=str, default='rtcpush') #rtmp webrtc rtcpush
    parser.add_argument('--push_url', type=str, default='http://106.39.219.223:1985/rtc/v1/whip/?app=live&stream=livestream') #rtmp://localhost/live/livestream
    parser.add_argument('--pull_url', type=str, default='http://106.39.219.223:1985/rtc/v1/whep/?app=live&stream=livestream1')

    parser.add_argument('--max_session', type=int, default=1)  #multi session count
    parser.add_argument('--listenport', type=int, default=8010)

    opt = parser.parse_args()
    #app.config.from_object(opt)
    #print(app.config)
    opt.customopt = []
    if opt.customvideo_config!='':
        with open(opt.customvideo_config,'r') as file:
            opt.customopt = json.load(file)

    if opt.model == 'musetalk':
        from musereal import MuseReal,load_model,load_avatar,warm_up
        print(opt)
        model = load_model()
        avatar = load_avatar(opt.avatar_id) 
        warm_up(opt.batch_size,model)      
        # for k in range(opt.max_session):
        #     opt.sessionid=k
        #     nerfreal = MuseReal(opt,audio_processor,vae, unet, pe,timesteps)
        #     nerfreals.append(nerfreal)

    if opt.transport=='rtmp':
        thread_quit = Event()
        nerfreals[0] = build_nerfreal(0)
        rendthrd = Thread(target=nerfreals[0].render,args=(thread_quit,))
        rendthrd.start()
    
    #############################################################################
    appasync = web.Application()
    appasync.on_shutdown.append(on_shutdown)
    appasync.router.add_post("/offer", offer)
    appasync.router.add_post("/start", start)
    appasync.router.add_post("/humanaudio", humanaudio)
    appasync.router.add_static('/',path='web')

    # Configure default CORS settings.
    cors = aiohttp_cors.setup(appasync, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
        })
    # Configure CORS on all routes.
    for route in list(appasync.router.routes()):
        cors.add(route)

    pagename='webrtcapi.html'
    if opt.transport=='rtmp':
        pagename='echoapi.html'
    elif opt.transport=='rtcpush':
        pagename='rtcpushapi.html'
    print('start http server; http://<serverip>:'+str(opt.listenport)+'/'+pagename)

    def run_server(runner):
        global message_queue, loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        message_queue = asyncio.Queue()
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, '0.0.0.0', opt.listenport)
        loop.run_until_complete(site.start())
        if opt.transport=='rtcpush':
            for k in range(opt.max_session):
                push_url = opt.push_url
                pull_url = opt.pull_url
                if k!=0:
                    push_url = opt.push_url+str(k)
                loop.run_until_complete(run(push_url,k))
        loop.run_forever()   
    
    #Thread(target=run_server, args=(web.AppRunner(appasync),)).start()
    run_server(web.AppRunner(appasync))

    #app.on_shutdown.append(on_shutdown)
    #app.router.add_post("/offer", offer)

    # print('start websocket server')
    # server = pywsgi.WSGIServer(('0.0.0.0', 8000), app, handler_class=WebSocketHandler)
    # server.serve_forever()
    
    

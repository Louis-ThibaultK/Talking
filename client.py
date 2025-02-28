import time
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder
import aiohttp
import argparse
import logging
from typing import Optional, Set, Union
import threading
from av.frame import Frame
from av.packet import Packet
import asyncio

logging.basicConfig()
logger = logging.getLogger(__name__)

pcs = set()
push_url = "http://10.176.205.11:8010/offer"

class AudioTrack(MediaStreamTrack):
    def __init__(self, kind, sample_rate=48000, channels=1):
        super().__init__()
        self.kind = kind
        self.sample_rate = sample_rate
        self.channels = channels
        self._queue: asyncio.Queue[Union[Frame, Packet]] = asyncio.Queue()
        self._start: Optional[float] = None

    async def recv(self)  -> Union[Frame, Packet]:

        data = await self._queue.get()
        return data
    
class StreamBuffer:
    """
    将接收到的音频数据保存到缓冲区。
    """
    def __init__(self):
        self.audio_buffer: asyncio.Queue[Union[Frame, Packet]] = asyncio.Queue()
        self.video_buffer: asyncio.Queue[Union[Frame, Packet]] = asyncio.Queue()
  


    def get_audio(self):
        return self.audio_buffer
    
    def get_video(self):
        return self.video_buffer
    
    def clear_buffer(self):
        """清理 buffer 内容"""
        self.audio_buffer = asyncio.Queue()
        self.video_buffer = asyncio.Queue()


async def post(url,data):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as response:
                return await response.json()
    except aiohttp.ClientError as e:
        print(f'Error: {e}')

async def run(push_url, player, buffer):
    push_pc = RTCPeerConnection()
    pull_pc = RTCPeerConnection() 
    pcs.add(push_pc)
    pcs.add(pull_pc)

    @push_pc.on("connectionstatechange")
    async def on_connectionstatechange():

        print("push Connection state is %s" % push_pc.connectionState)
        if push_pc.connectionState == "failed":
            await push_pc.close()
            pcs.discard(push_pc)

    @pull_pc.on("connectionstatechange")
    async def on_connectionstatechange():

        print("pull Connection state is %s" % pull_pc.connectionState)
        if pull_pc.connectionState == "failed":
            await pull_pc.close()
            pcs.discard(pull_pc)
     
    async def on_track(track):
        if track.kind == "audio":
            print("Audio track received")
            while(True):
                frame = await track.recv()
                buffer.audio_buffer.put(frame)
            #         # 这里可以处理音频数据，例如进行转换、保存等
            #     print(f"Received audio frame with timestamp {frame.time}")

        elif track.kind == "video":
            print("Video track received")
            while(True):
                frame = await track.recv()
                buffer.video_buffer.put(frame) 
            #         # 这里可以处理音频数据，例如进行转换、保存等
            #     print(f"Received video frame with timestamp {frame.time}")

    pull_pc.on("track", on_track)

    #推流
    audio_transceiver = push_pc.addTransceiver(player.audio, direction="sendonly")

    #拉流
    pull_pc.addTransceiver("audio", direction="recvonly")
    pull_pc.addTransceiver("video", direction="recvonly")
    # 创建 SDP Offer
    push_offer = await push_pc.createOffer()
    
    await push_pc.setLocalDescription(push_offer)

    pull_offer = await pull_pc.createOffer()

    await pull_pc.setLocalDescription(pull_offer)

    # 向 WHEP 服务端发送 SDP Offer
    sdp_data = {
        "push_sdp": push_pc.localDescription.sdp,
        "pull_sdp": pull_pc.localDescription.sdp,
        "type": "offer"
    }
    print("push offer:", push_offer)
    print("pull offer:", pull_offer)
    response = await post(push_url, sdp_data)
    
    push_answer = RTCSessionDescription(sdp=response["pull_sdp"], type=response["type"])
    await push_pc.setRemoteDescription(push_answer)

    pull_answer = RTCSessionDescription(sdp=response["push_sdp"], type=response["type"])
    await pull_pc.setRemoteDescription(pull_answer)
    print("push answer:", push_answer)
    print("pull answer:", pull_answer)

def add_signaling_arguments(parser):
    """
    Add signaling method arguments to an argparse.ArgumentParser.
    """
    parser.add_argument(
        "--signaling",
        "-s",
        choices=["copy-and-paste", "tcp-socket", "unix-socket"],
    )
    parser.add_argument(
        "--signaling-host", default="127.0.0.1", help="Signaling host (tcp-socket only)"
    )
    parser.add_argument(
        "--signaling-port", default=1234, help="Signaling port (tcp-socket only)"
    )
    parser.add_argument(
        "--signaling-path",
        default="aiortc.socket",
        help="Signaling socket path (unix-socket only)",
    )

##使用示例如下
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video stream from the command line")
    parser.add_argument("--push_url", help = "Choose push stream url")

    add_signaling_arguments(parser)
    args = parser.parse_args()

    audio = AudioTrack(kind="audio")
    stream_buffer = StreamBuffer()
    
    if args.push_url:
        push_url = args.push_url

    # asyncio.run(run(
    #     push_url,
    #     player=player,
    #     recorder=recorder
    # ))
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run(
        push_url,
        audio=audio,
        buffer=stream_buffer
    ))
    loop.run_forever()

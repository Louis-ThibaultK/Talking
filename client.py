from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder
import aiohttp
import argparse
import logging
import asyncio

pcs = set()
push_url = ""

async def post(url,data):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as response:
                return await response.text()
    except aiohttp.ClientError as e:
        print(f'Error: {e}')

async def run(player, recorder):
    pc = RTCPeerConnection()
    pcs.add(pc)
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():

        print("pull Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)
     
    async def on_track(track):
        if track.kind == "audio":
            print("Audio track received")

        elif track.kind == "video":
            print("Video track received")
        recorder.addTrack(track)

    pc.on("track", on_track)

    pc.addTrack(player.audio)
    # 创建 SDP Offer
    offer = await pc.createOffer()
    print("offer:", offer)
    await pc.setLocalDescription(offer)

    # 向 WHEP 服务端发送 SDP Offer
    sdp_data = {
        "sdp": offer,
        "type": "offer"
    }
    response = await post(push_url, sdp_data)
    print("answer:", answer)
    answer = RTCSessionDescription(sdp=response["sdp"], type=response["type"])
    await pc.setRemoteDescription(answer)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video stream from the command line")
    parser.add_argument("--push_url", help = "Choose push stream url")
    parser.add_argument("--play-from", help="Read the media from a file and sent it.")
    parser.add_argument("--record-to", help="Write received media to a file.")
    parser.add_argument("--verbose", "-v", action="count")

    add_signaling_arguments(parser)
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    # create signaling and peer connection

    # create media source
    if args.play_from:
        player = MediaPlayer(args.play_from)
    else:
        player = None

    # create media sink
    if args.record_to:
        recorder = MediaRecorder(args.record_to)
    else:
        recorder = MediaBlackhole()
    
    if args.push_url:
        push_url = args.push_url

    loop = asyncio.get_event_loop(run(
        player=player,
        recorder=recorder
    ))


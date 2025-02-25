# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import dataclasses
import datetime
import io
import os
import wave
from dataclasses import dataclass

import httpx

import openai
import requests

from .models import WhisperModels, GroqAudioModels


@dataclass
class _STTOptions:
    language: str
    detect_language: bool
    model: WhisperModels


class STT():
    def __init__(
        self,
        *,
        language: str = "en",
        detect_language: bool = False,
        model: WhisperModels = "whisper-1",
        base_url: str | None = None,
        api_key: str | None = None,
        client: openai.AsyncClient | None = None,
        ## TODO
        # noise_filter: tuple[nn.Module, DF, str] | None = None,
    ):
        """
        Create a new instance of OpenAI STT.

        ``api_key`` must be set to your OpenAI API key, either using the argument or by setting the
        ``OPENAI_API_KEY`` environmental variable.
        """

        # super().__init__(
        #     capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        # )
        if detect_language:
            language = ""

        self._opts = _STTOptions(
            language=language,
            detect_language=detect_language,
            model=model,
        )

        self._client = client or openai.AsyncClient(
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.AsyncClient(
                timeout=httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0),
                follow_redirects=True,
                limits=httpx.Limits(
                    max_connections=50,
                    max_keepalive_connections=50,
                    keepalive_expiry=120,
                ),
            ),
        )

        ## TODO
        # self._noise_filter = noise_filter

    
    @staticmethod
    def with_groq(
        *,
        model: GroqAudioModels = "whisper-large-v3-turbo",
        api_key: str | None = None,
        base_url: str | None = "https://api.groq.com/openai/v1",
        client: openai.AsyncClient | None = None,
        language: str = "en",
        detect_language: bool = False,
    ) -> STT:
        """
        Create a new instance of Groq STT.

        ``api_key`` must be set to your Groq API key, either using the argument or by setting
        the ``GROQ_API_KEY`` environmental variable.
        """
        
        # Use environment variable if API key is not provided
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if api_key is None:
            raise ValueError("Groq API key is required")

        # Instantiate and return a configured STT instance
        return STT(
            model=model,
            api_key=api_key,
            base_url=base_url,
            client=client,
            language=language,
            detect_language=detect_language,
        )

    def _sanitize_options(self, *, language: str | None = None) -> _STTOptions:
        config = dataclasses.replace(self._opts)
        config.language = language or config.language
        return config

    async def recognize(
        self, buffer, language: str | None = None
    ) -> any:
        config = self._sanitize_options(language=language)

        ## TODO log
        # md = "openai"
        # if config.language == "zn":
        #     md = "qwen2"
        # print("STT(", md, ") input timestamp", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))

        ## TODO - before
        # buffer = agents.utils.merge_frames(buffer)
        io_buffer = io.BytesIO()
        with wave.open(io_buffer, "wb") as wav:
            wav.setnchannels(buffer.num_channels)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(buffer.sample_rate)
            wav.writeframes(buffer.get_data())
        

        url = "http://10.218.127.29:3000/transcribe"
        payload = {'language': 'zn'}
        
        resp = await self._client.audio.transcriptions.create(
            file=("my_file.wav", io_buffer.getvalue(), "audio/wav"),
            model=self._opts.model,
            language=config.language,
            response_format="json",
        )

        # ## TODO - after
        # buffer = agents.utils.merge_frames(buffer)
        # io_buffer = io.BytesIO()
        # with wave.open(io_buffer, "wb") as wav:
        #     wav.setnchannels(buffer.num_channels)
        #     wav.setsampwidth(2)  # 16-bit
        #     wav.setframerate(buffer.sample_rate)
        #     wav.writeframes(buffer.data)
        #
        # # 从 BytesIO 中获取原始音频数据
        # array_data = np.frombuffer(io_buffer.getvalue(), dtype=np.int16)  # Use int16 for 16-bit audio
        # num_frames = len(array_data) // buffer.num_channels  # Calculate the total number of frames
        # array_data = array_data.reshape(
        #     (num_frames, buffer.num_channels))  # Reshape the array to (num_frames, num_channels)
        #
        # src_tmp_fn = "file_" + str(int(time.time())) + ".wav"
        # soundfile.write(src_tmp_fn, array_data, buffer.sample_rate, format='WAV')
        # try:
        #     # deepFilterNet - 噪声过滤
        #     model = self._noise_filter[0]
        #     df_state = self._noise_filter[1]
        #     audio_src, _ = load_audio(src_tmp_fn, sr=df_state.sr())  # 加载原始数据
        #     audio_des = enhance(model, df_state, audio_src)
        #
        #     # 保存噪声过滤后文件
        #     # save_audio("enhanced_"+src_tmp_fn, audio_des, df_state.sr())
        #
        #     # 使用 wave 将处理后的音频数据写入新的 BytesIO 对象
        #     final_io_buffer = io.BytesIO()
        #     # 假设 audio_des 是增强后的浮点型音频数据
        #     audio_des_int16 = np.int16(audio_des * 32767)
        #     # 写入 BytesIO，确保使用 16-bit 采样
        #     with wave.open(final_io_buffer, "wb") as wav:
        #         wav.setnchannels(buffer.num_channels)
        #         wav.setsampwidth(2)  # 16-bit
        #         wav.setframerate(buffer.sample_rate)
        #         wav.writeframes(audio_des_int16.tobytes())  # 将去噪后的音频数据写入
        #     final_io_buffer.seek(0)
        #
        #     resp = await self._client.audio.transcriptions.create(
        #         file=("enhanced_my_file.wav", final_io_buffer.getvalue(), "audio/wav"),
        #         model=self._opts.model,
        #         language=config.language,
        #         response_format="json",
        #     )
        # finally:
        #     if os.path.exists(src_tmp_fn):
        #         os.remove(src_tmp_fn)

        ## TODO
        ## qwen2
        if resp.data is not None:
            resp.text = resp.data
        # print("STT(", md, ") output timestamp", resp.text, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))

        return resp.text

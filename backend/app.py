import argparse
import asyncio
import json
import websockets
import uuid
import os
import time
from os import remove
import wave
from transformers import pipeline
from faster_whisper import WhisperModel
import whisper
import argostranslate.package
import argostranslate.translate
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import pyttsx3
import base64

from pyannote.core import Segment
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection

# Download and install Argos Translate package
argostranslate.package.update_package_index()
available_packages = argostranslate.package.get_available_packages()

language_codes = {
    "afrikaans": "af",
    "amharic": "am",
    "arabic": "ar",
    "assamese": "as",
    "azerbaijani": "az",
    "bashkir": "ba",
    "belarusian": "be",
    "bulgarian": "bg",
    "bengali": "bn",
    "tibetan": "bo",
    "breton": "br",
    "bosnian": "bs",
    "catalan": "ca",
    "czech": "cs",
    "welsh": "cy",
    "danish": "da",
    "german": "de",
    "greek": "el",
    "english": "en",
    "spanish": "es",
    "estonian": "et",
    "basque": "eu",
    "persian": "fa",
    "finnish": "fi",
    "faroese": "fo",
    "french": "fr",
    "galician": "gl",
    "gujarati": "gu",
    "hausa": "ha",
    "hawaiian": "haw",
    "hebrew": "he",
    "hindi": "hi",
    "croatian": "hr",
    "haitian": "ht",
    "hungarian": "hu",
    "armenian": "hy",
    "indonesian": "id",
    "icelandic": "is",
    "italian": "it",
    "japanese": "ja",
    "javanese": "jw",
    "georgian": "ka",
    "kazakh": "kk",
    "khmer": "km",
    "kannada": "kn",
    "korean": "ko",
    "latin": "la",
    "luxembourgish": "lb",
    "lingala": "ln",
    "lao": "lo",
    "lithuanian": "lt",
    "latvian": "lv",
    "malagasy": "mg",
    "maori": "mi",
    "macedonian": "mk",
    "malayalam": "ml",
    "mongolian": "mn",
    "marathi": "mr",
    "malay": "ms",
    "maltese": "mt",
    "burmese": "my",
    "nepali": "ne",
    "dutch": "nl",
    "norwegian nynorsk": "nn",
    "norwegian": "no",
    "occitan": "oc",
    "punjabi": "pa",
    "persian": "fa",
    "polish": "pl",
    "pashto": "ps",
    "portuguese": "pt",
    "romanian": "ro",
    "russian": "ru",
    "sanskrit": "sa",
    "sindhi": "sd",
    "sinhalese": "si",
    "slovak": "sk",
    "slovenian": "sl",
    "shona": "sn",
    "somali": "so",
    "albanian": "sq",
    "serbian": "sr",
    "sundanese": "su",
    "swedish": "sv",
    "swahili": "sw",
    "tamil": "ta",
    "telugu": "te",
    "tajik": "tg",
    "thai": "th",
    "turkmen": "tk",
    "tagalog": "tl",
    "turkish": "tr",
    "tatar": "tt",
    "ukrainian": "uk",
    "urdu": "ur",
    "uzbek": "uz",
    "vietnamese": "vi",
    "yiddish": "yi",
    "yoruba": "yo",
    "chinese": "zh",
    "cantonese": "yue",
}


class WhisperASR:
    def __init__(self, **kwargs):
        model_name = kwargs.get("model_name", "openai/whisper-base")
        # self.asr_pipeline = pipeline("automatic-speech-recognition", model=model_name)
        self.model = whisper.load_model("base")

    async def transcribe(self, client):
        file_path = await save_audio_to_file(
            client.scratch_buffer, client.get_file_name()
        )

        print(client.config)
        if client.config["language"] is not None and client.config["translate"] == True:
            # to_return = self.asr_pipeline(
            #     file_path,
            #     generate_kwargs={
            #         "language": client.config["language"],
            #         "task": "translate",
            #     },
            # )["text"]
            to_return = self.model.transcribe(
                file_path,
                language=language_codes[client.config["outputLanguage"]],
                task="translate",
            )["text"]

            from_code = language_codes[client.config["language"]]
            to_code = language_codes[client.config["outputLanguage"]]

            package_to_install = next(
                filter(
                    lambda x: x.from_code == from_code and x.to_code == to_code,
                    available_packages,
                )
            )
            argostranslate.package.install_from_path(package_to_install.download())

            to_return = argostranslate.translate.translate(
                to_return, from_code, to_code
            )
        elif (
            client.config["language"] is not None
            and client.config["translate"] == False
        ):
            # to_return = self.asr_pipeline(
            #     file_path, generate_kwargs={"language": client.config["language"]}
            # )["text"]
            to_return = self.model.transcribe(
                file_path, language=language_codes[client.config["language"]]
            )["text"]
        else:
            # to_return = self.asr_pipeline(file_path)["text"]
            to_return = self.model.transcribe(file_path)["text"]

        os.remove(file_path)

        to_return = {
            "text": to_return.strip(),
        }
        return to_return


# class FasterWhisperASR:
#     def __init__(self, **kwargs):
#         model_size = kwargs.get("model_size", "base")
#         # Run on GPU with FP16
#         self.asr_pipeline = WhisperModel(
#             model_size, device="cpu", compute_type="float32"
#         )

#     async def transcribe(self, client):
#         file_path = await save_audio_to_file(
#             client.scratch_buffer, client.get_file_name()
#         )

#         language = (
#             None
#             if client.config["language"] is None
#             else language_codes.get(client.config["language"].lower())
#         )
#         segments, info = self.asr_pipeline.transcribe(
#             file_path, word_timestamps=True, language=language
#         )

#         segments = list(segments)  # The transcription will actually run here.
#         os.remove(file_path)

#         flattened_words = [word for segment in segments for word in segment.words]

#         to_return = {
#             "text": " ".join([s.text.strip() for s in segments]),
#         }
#         return to_return


async def save_audio_to_file(
    audio_data, file_name, audio_dir="audio_files", audio_format="wav"
):

    os.makedirs(audio_dir, exist_ok=True)

    file_path = os.path.join(audio_dir, file_name)

    with wave.open(file_path, "wb") as wav_file:
        wav_file.setnchannels(1)  # Assuming mono audio
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(audio_data)

    return file_path


def synthesize_speech(text):
    engine = pyttsx3.init()
    engine.save_to_file(text, "output.wav")
    engine.runAndWait()
    with open("output.wav", "rb") as f:
        audio_data = f.read()
    return base64.b64encode(audio_data).decode("latin1")


class PyannoteVAD:

    def __init__(self, **kwargs):

        model_name = "pyannote/segmentation"

        auth_token = "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"

        if auth_token is None:
            raise ValueError("Missing required 'auth_token'")

        pyannote_args = kwargs.get(
            "pyannote_args",
            {
                "onset": 0.5,
                "offset": 0.5,
                "min_duration_on": 0.3,
                "min_duration_off": 0.3,
            },
        )

        self.model = Model.from_pretrained(model_name, use_auth_token=auth_token)
        self.vad_pipeline = VoiceActivityDetection(segmentation=self.model)
        self.vad_pipeline.instantiate(pyannote_args)

    async def detect_activity(self, client):
        audio_file_path = await save_audio_to_file(
            client.scratch_buffer, client.get_file_name()
        )
        vad_results = self.vad_pipeline(audio_file_path)
        remove(audio_file_path)
        vad_segments = []
        if len(vad_results) > 0:
            vad_segments = [
                {"start": segment.start, "end": segment.end, "confidence": 1.0}
                for segment in vad_results.itersegments()
            ]
        return vad_segments


class SilenceAtEndOfChunk:

    def __init__(self, client, **kwargs):
        self.client = client

        self.chunk_length_seconds = kwargs.get("chunk_length_seconds")
        self.chunk_length_seconds = float(self.chunk_length_seconds)

        self.chunk_offset_seconds = kwargs.get("chunk_offset_seconds")
        self.chunk_offset_seconds = float(self.chunk_offset_seconds)

        self.error_if_not_realtime = kwargs.get("error_if_not_realtime", False)

        self.processing_flag = False

    def process_audio(self, websocket, vad_pipeline, asr_pipeline):

        chunk_length_in_bytes = (
            self.chunk_length_seconds
            * self.client.sampling_rate
            * self.client.samples_width
        )
        if len(self.client.buffer) > chunk_length_in_bytes:
            if self.processing_flag:
                exit("Error: Chunk interference")

            self.client.scratch_buffer += self.client.buffer
            self.client.buffer.clear()
            self.processing_flag = True
            # Schedule the processing in a separate task
            asyncio.create_task(
                self.process_audio_async(websocket, vad_pipeline, asr_pipeline)
            )

    async def process_audio_async(self, websocket, vad_pipeline, asr_pipeline):

        start = time.time()
        vad_results = await vad_pipeline.detect_activity(self.client)

        if len(vad_results) == 0:
            self.client.scratch_buffer.clear()
            self.client.buffer.clear()
            self.processing_flag = False
            return

        last_segment_should_end_before = (
            len(self.client.scratch_buffer)
            / (self.client.sampling_rate * self.client.samples_width)
        ) - self.chunk_offset_seconds
        if vad_results[-1]["end"] < last_segment_should_end_before:
            transcription = await asr_pipeline.transcribe(self.client)
            if transcription["text"] != "":
                end = time.time()
                transcription["processing_time"] = end - start
                json_transcription = json.dumps(transcription)
                await websocket.send(json_transcription)
                # if (
                #     "speechTranslation" in self.client.config
                #     and self.client.config["speechTranslation"] == True
                # ):
                #     speech_audio_data = synthesize_speech(transcription["text"])
                #     await websocket.send(
                #         json.dumps({"type": "audio", "data": speech_audio_data})
                #     )
            self.client.scratch_buffer.clear()
            self.client.increment_file_counter()

        self.processing_flag = False


class Client:

    def __init__(self, client_id, sampling_rate, samples_width):
        self.client_id = client_id
        self.buffer = bytearray()
        self.scratch_buffer = bytearray()
        self.config = {
            "language": None,
            "processing_strategy": "silence_at_end_of_chunk",
            "processing_args": {"chunk_length_seconds": 3, "chunk_offset_seconds": 0.1},
        }
        self.file_counter = 0
        self.total_samples = 0
        self.sampling_rate = sampling_rate
        self.samples_width = samples_width
        self.buffering_strategy = SilenceAtEndOfChunk(
            self, **self.config["processing_args"]
        )

    def update_config(self, config_data):
        self.config.update(config_data)
        self.buffering_strategy = SilenceAtEndOfChunk(
            self, **self.config["processing_args"]
        )

    def append_audio_data(self, audio_data):
        self.buffer.extend(audio_data)
        self.total_samples += len(audio_data) / self.samples_width

    def clear_buffer(self):
        self.buffer.clear()

    def increment_file_counter(self):
        self.file_counter += 1

    def get_file_name(self):
        return f"{self.client_id}_{self.file_counter}.wav"

    def process_audio(self, websocket, vad_pipeline, asr_pipeline):
        self.buffering_strategy.process_audio(websocket, vad_pipeline, asr_pipeline)


def text_summarizer(text):
    # Tokenizing the text
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text)

    freqTable = dict()
    for word in words:
        word = word.lower()
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    sentences = sent_tokenize(text)
    sentenceValue = dict()

    for sentence in sentences:
        for word, freq in freqTable.items():
            if word in sentence.lower():
                if sentence in sentenceValue:
                    sentenceValue[sentence] += freq
                else:
                    sentenceValue[sentence] = freq

    sumValues = 0
    for sentence in sentenceValue:
        sumValues += sentenceValue[sentence]

    # Average value of a sentence from the original text

    average = int(sumValues / len(sentenceValue))

    # Storing sentences into our summary.
    summary = ""
    for sentence in sentences:
        if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
            summary += " " + sentence
    return summary


class Server:

    def __init__(
        self,
        vad_pipeline,
        asr_pipeline,
        host="localhost",
        port=8765,
        sampling_rate=16000,
        samples_width=2,
    ):
        self.vad_pipeline = vad_pipeline
        self.asr_pipeline = asr_pipeline
        self.host = host
        self.port = port
        self.sampling_rate = sampling_rate
        self.samples_width = samples_width
        self.connected_clients = {}

    async def handle_audio(self, client, websocket):
        while True:
            message = await websocket.recv()
            if isinstance(message, bytes):
                client.append_audio_data(message)
            elif isinstance(message, str):
                config = json.loads(message)
                if config.get("type") == "config":
                    client.update_config(config["data"])
                    continue
                if config.get("type") == "summarize":
                    text = config["data"]["text"]
                    summary = text_summarizer(text)
                    await websocket.send(
                        json.dumps({"type": "summary", "text": summary})
                    )
            else:
                print(f"Unexpected message type from {client.client_id}")

            # this is synchronous, any async operation is in BufferingStrategy
            client.process_audio(websocket, self.vad_pipeline, self.asr_pipeline)

    async def handle_websocket(self, websocket, path):
        client_id = str(uuid.uuid4())
        client = Client(client_id, self.sampling_rate, self.samples_width)
        self.connected_clients[client_id] = client

        print(f"Client {client_id} connected")

        try:
            await self.handle_audio(client, websocket)
        except websockets.ConnectionClosed as e:
            print(f"Connection with {client_id} closed: {e}")
        finally:
            del self.connected_clients[client_id]

    def start(self):
        return websockets.serve(self.handle_websocket, self.host, self.port)


def main(host, port):

    try:
        # vad_args = json.loads('{"auth_token": "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"}')
        asr_args = json.loads('{"model_size": "base"}')
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON arguments: {e}")
        return

    vad_pipeline = PyannoteVAD()
    asr_pipeline = WhisperASR(**asr_args)

    server = Server(
        vad_pipeline,
        asr_pipeline,
        host=host,
        port=port,
        sampling_rate=16000,
        samples_width=2,
    )

    asyncio.get_event_loop().run_until_complete(server.start())
    asyncio.get_event_loop().run_forever()


if __name__ == "__main__":
    main(host="127.0.0.1", port=8765)

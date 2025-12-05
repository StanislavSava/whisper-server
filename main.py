# server.py -- Buffered Mode (transcribe only at end)

import asyncio
import json
import logging

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from pywhispercpp.model import Model

log = logging.getLogger("uvicorn.error")


TARGET_SR = 16000
BYTES_PER_SAMPLE = 2

WHISPER_MODEL = "large-v3"


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = Model(WHISPER_MODEL)

inference_lock = asyncio.Lock()


def pcm16_bytes_to_float32_array(b: bytes) -> np.ndarray:
    if not b:
        return np.zeros(0, dtype=np.float32)
    return (np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32768.0)

def normalize_audio(float_audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    if float_audio.size == 0:
        return float_audio
    mx = np.max(np.abs(float_audio))
    if mx < 1e-6:
        return float_audio
    return (float_audio * (peak / mx)).astype(np.float32)

def resample_linear(x: np.ndarray, in_sr: int, out_sr: int) -> np.ndarray:
    if in_sr == out_sr:
        return x
    ratio = in_sr / out_sr
    new_len = int(len(x) / ratio)
    idx = np.linspace(0, len(x) - 1, new_len)
    return np.interp(idx, np.arange(len(x)), x).astype(np.float32)

def segments_to_text(segments):
    try:
        return "".join(getattr(s, "text", str(s)) for s in segments)
    except:
        return ""


@app.websocket("/ws/transcribe")
async def websocket_transcribe(ws: WebSocket):
    await ws.accept()
    log.info("Client connected")

    audio_buffer = bytearray()
    client_sr = TARGET_SR

    try:
        while True:
            msg = await ws.receive()

          
            if msg.get("bytes") is not None:
                # Just buffer the audio, don't transcribe yet
                audio_buffer += msg["bytes"]
                log.debug(f"Buffered {len(msg['bytes'])} bytes, total: {len(audio_buffer)} bytes")

            elif msg.get("text") is not None:
                try:
                    packet = json.loads(msg["text"])
                except:
                    packet = {}

                # client sample rate
                if "sr" in packet:
                    try:
                        client_sr = int(packet["sr"])
                        log.info(f"Client sample rate set to {client_sr}")
                    except:
                        pass

                # FINALIZATION - transcribe all buffered audio
                if packet.get("end"):
                    if len(audio_buffer) == 0:
                        log.warning("Received end packet but buffer is empty")
                        await ws.send_json({"type": "final", "text": ""})
                    else:
                        log.info(f"End packet received, transcribing {len(audio_buffer)} bytes")
                        
                        # Convert all buffered audio
                        all_audio = bytes(audio_buffer)
                        float_audio = pcm16_bytes_to_float32_array(all_audio)

                        # Resample if necessary
                        if client_sr != TARGET_SR:
                            log.info(f"Resampling from {client_sr}Hz to {TARGET_SR}Hz")
                            float_audio = resample_linear(float_audio, client_sr, TARGET_SR)

                        # Normalize
                        float_audio = normalize_audio(float_audio)

                        log.info(f"Running transcription on {len(float_audio)} samples ({len(float_audio)/TARGET_SR:.2f} seconds)")
                        
                        # Transcribe entire buffer
                        async with inference_lock:
                            try:
                                segs = model.transcribe(float_audio)
                                text = segments_to_text(segs)
                            except Exception as e:
                                log.exception("Transcription error")
                                text = ""

                        if text.strip():
                            await ws.send_json({"type": "final", "text": text})
                            log.info(f"Transcription complete: {text[:100]}...")
                        else:
                            await ws.send_json({"type": "final", "text": ""})
                            log.warning("Transcription produced no text")

                    # Reset buffer for next utterance
                    audio_buffer = bytearray()
                    continue

            # --------------------------------
            # DISCONNECT
            # --------------------------------
            if msg.get("type") == "websocket.disconnect":
                return

    except WebSocketDisconnect:
        log.info("Client disconnected")
        return
    except Exception as e:
        log.exception("Server error")
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except:
            pass

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8765, log_level="info")
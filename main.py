# server.py (VAD REMOVED)
import asyncio
import json
import logging
import time

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from pywhispercpp.model import Model

log = logging.getLogger("uvicorn.error")

# ----------------------------
# CONFIG
# ----------------------------
TARGET_SR = 16000
BYTES_PER_SAMPLE = 2

FRAME_SECONDS = 1.5
HOP_SECONDS = 0.5
FRAME_BYTES = int(TARGET_SR * FRAME_SECONDS) * BYTES_PER_SAMPLE
HOP_BYTES = int(TARGET_SR * HOP_SECONDS) * BYTES_PER_SAMPLE

KEEP_CONTEXT_SECONDS = 10.0
KEEP_BYTES = int(TARGET_SR * KEEP_CONTEXT_SECONDS) * BYTES_PER_SAMPLE

PARTIAL_MIN_INTERVAL = 0.5
POST_SPEECH_PAD_SECONDS = 0.25

WHISPER_MODEL = "large-v3"
TRANSCRIBE_OPTIONS = {}

# ----------------------------
# App & Model
# ----------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

log.info(f"Loading whisper model {WHISPER_MODEL} ...")
model = Model(WHISPER_MODEL)
log.info("Model loaded.")

inference_lock = asyncio.Lock()

# ----------------------------
# Helpers
# ----------------------------
def pcm16_bytes_to_float32_array(b: bytes) -> np.ndarray:
    if not b:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32768.0

def normalize_audio(x: np.ndarray, peak: float = 0.95) -> np.ndarray:
    if x.size == 0:
        return x
    m = np.max(np.abs(x))
    if m < 1e-6:
        return x
    return (x * (peak / m)).astype(np.float32)

def resample_linear(x: np.ndarray, in_sr: int, out_sr: int) -> np.ndarray:
    if in_sr == out_sr or x.size == 0:
        return x.astype(np.float32)
    ratio = in_sr / out_sr
    new_len = int(len(x) / ratio)
    idx = np.linspace(0, len(x) - 1, new_len)
    return np.interp(idx, np.arange(len(x)), x).astype(np.float32)

def segments_to_text(segments):
    try:
        return "".join(getattr(s, "text", str(s)) for s in segments)
    except:
        return ""

def longest_common_prefix(a: str, b: str) -> int:
    i = 0
    L = min(len(a), len(b))
    while i < L and a[i] == b[i]:
        i += 1
    return i

def collapse_repeated_endings(text: str, min_repeat_chars=6) -> str:
    s = text.strip()
    L = len(s)
    for size in range(min_repeat_chars, L // 2 + 1):
        if s[-2*size:-size] == s[-size:]:
            return s[:-size]
    return s

# ----------------------------
# WebSocket endpoint
# ----------------------------
@app.websocket("/ws/transcribe")
async def websocket_transcribe(ws: WebSocket):
    await ws.accept()
    log.info("Client connected")

    audio_buffer = bytearray()
    client_sr = TARGET_SR

    last_partial_full = ""
    last_partial_emit_time = 0.0

    try:
        while True:
            msg = await ws.receive()

            # -------------------------
            # BINARY AUDIO FROM CLIENT
            # -------------------------
            if msg.get("bytes") is not None:
                chunk = msg["bytes"]
                if not chunk:
                    continue

                audio_buffer += chunk

                # keep context bounded
                if len(audio_buffer) > KEEP_BYTES:
                    audio_buffer = audio_buffer[-KEEP_BYTES:]

                # -------------------------
                # PARTIAL TRANSCRIPTION
                # -------------------------
                now = time.time()
                if (now - last_partial_emit_time) >= PARTIAL_MIN_INTERVAL:
                    samples_needed = int(TARGET_SR * FRAME_SECONDS)
                    bytes_needed = samples_needed * BYTES_PER_SAMPLE
                    tail_bytes = audio_buffer[-bytes_needed:] if len(audio_buffer) >= bytes_needed else audio_buffer

                    float_audio = pcm16_bytes_to_float32_array(bytes(tail_bytes))

                    if client_sr != TARGET_SR:
                        float_audio = resample_linear(float_audio, client_sr, TARGET_SR)

                    float_audio = normalize_audio(float_audio)

                    async with inference_lock:
                        try:
                            segs = model.transcribe(float_audio, **TRANSCRIBE_OPTIONS)
                            full_text = segments_to_text(segs).strip()
                        except:
                            full_text = ""

                    full_text = collapse_repeated_endings(full_text)

                    # compute delta via LCP
                    if full_text and full_text != last_partial_full:
                        lcp = longest_common_prefix(last_partial_full, full_text)
                        delta = full_text[lcp:].lstrip()

                        if delta:
                            await ws.send_json({"type": "partial", "text": delta})
                            last_partial_full = full_text
                            last_partial_emit_time = now

            # -------------------------
            # TEXT CONTROL MESSAGES
            # -------------------------
            elif msg.get("text") is not None:
                try:
                    packet = json.loads(msg["text"])
                except:
                    packet = {}

                if "sr" in packet:
                    try:
                        client_sr = int(packet["sr"])
                    except:
                        pass

                if packet.get("end") is True:
                    await asyncio.sleep(POST_SPEECH_PAD_SECONDS)

                    float_audio = pcm16_bytes_to_float32_array(bytes(audio_buffer))
                    if client_sr != TARGET_SR:
                        float_audio = resample_linear(float_audio, client_sr, TARGET_SR)
                    float_audio = normalize_audio(float_audio)

                    async with inference_lock:
                        try:
                            segs = model.transcribe(float_audio, **TRANSCRIBE_OPTIONS)
                            final_text = segments_to_text(segs).strip()
                        except:
                            final_text = ""

                    final_text = collapse_repeated_endings(final_text)
                    await ws.send_json({"type": "final", "text": final_text})

                    # reset
                    audio_buffer = bytearray()
                    last_partial_full = ""
                    last_partial_emit_time = 0.0
                    continue

            # -------------------------
            # DISCONNECT
            # -------------------------
            elif msg.get("type") == "websocket.disconnect":
                return

    except WebSocketDisconnect:
        return
    except Exception:
        log.exception("Server error")
        try:
            await ws.send_json({"type": "error", "message": "Server crashed"})
        except:
            pass

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8765, log_level="info")
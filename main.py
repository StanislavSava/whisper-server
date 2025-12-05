# server.py  -- High-Accuracy Mode with overlap + VAD + normalization
import asyncio
import json
import logging

import numpy as np
import webrtcvad
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from pywhispercpp.model import Model

log = logging.getLogger("uvicorn.error")

# ----------------------------
# Config (HIGH-ACCURACY)
# ----------------------------
TARGET_SR = 32000                # model/sample rate
BYTES_PER_SAMPLE = 2             # PCM16 little-endian
FRAME_SECONDS = 2              # analysis window (0.5s)
HOP_SECONDS = FRAME_SECONDS / 2  # 50% overlap
FRAME_BYTES = int(TARGET_SR * FRAME_SECONDS) * BYTES_PER_SAMPLE
HOP_BYTES = int(TARGET_SR * HOP_SECONDS) * BYTES_PER_SAMPLE

KEEP_CONTEXT_SECONDS = 3.0       # keep last N seconds for context
KEEP_BYTES = int(TARGET_SR * KEEP_CONTEXT_SECONDS) * BYTES_PER_SAMPLE

VAD_MODE = 2                     # 0-3 (aggressiveness), 2 is a good default
VAD_ENABLED = True
VAD_FRAME_MS = 30                # webrtcvad supports 10,20,30 ms frames
VAD_SILENCE_HANG = int((0.6) / HOP_SECONDS)  # number of hops of silence to finalize (0.6s default)
POST_SPEECH_PAD_SECONDS = 0.35   # pad after final detection before final inference

WHISPER_MODEL = "large-v3"       # high-accuracy model (use large-v3-turbo if large-v3 not available)

# ----------------------------
# App setup
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
    arr = np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32768.0
    return arr

def float32_to_pcm16_bytes(x: np.ndarray) -> bytes:
    clipped = np.clip(x, -1.0, 1.0)
    ints = (clipped * 32767.0).astype(np.int16)
    return ints.tobytes()

def segments_to_text(segments):
    if not segments:
        return ""
    try:
        return "".join(getattr(s, "text", str(s)) for s in segments)
    except Exception:
        return "".join((s.get("text","") if isinstance(s, dict) else str(s)) for s in segments)

def rms_energy(float32_audio: np.ndarray) -> float:
    if float32_audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(float32_audio**2)))

def normalize_audio(float_audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    max_abs = np.max(np.abs(float_audio)) if float_audio.size else 0.0
    if max_abs < 1e-6:
        return float_audio
    gain = target_peak / max_abs
    return (float_audio * gain).astype(np.float32)

def resample_linear(x: np.ndarray, in_sr: int, out_sr: int) -> np.ndarray:
    if in_sr == out_sr:
        return x
    ratio = in_sr / out_sr
    new_len = int(np.round(len(x) / ratio))
    if new_len <= 0:
        return np.zeros(0, dtype=np.float32)
    idx = (np.arange(new_len) * ratio)
    idx0 = np.floor(idx).astype(int)
    idx1 = np.minimum(idx0 + 1, len(x) - 1)
    w = idx - idx0
    y = (1.0 - w) * x[idx0] + w * x[idx1]
    return y.astype(np.float32)

# WebRTC VAD needs 10/20/30ms frames of raw PCM16
def frame_generator(frame_ms: int, audio_bytes: bytes, sample_rate: int):
    n = int(sample_rate * (frame_ms / 1000.0)) * BYTES_PER_SAMPLE
    offset = 0
    while offset + n <= len(audio_bytes):
        yield audio_bytes[offset: offset + n]
        offset += n

# ----------------------------
# WebSocket endpoint
# ----------------------------
@app.websocket("/ws/transcribe")
async def websocket_transcribe(ws: WebSocket):
    await ws.accept()
    log.info("Client connected")
    audio_buffer = bytearray()
    client_sr = TARGET_SR
    vad = webrtcvad.Vad(VAD_MODE) if VAD_ENABLED else None
    consecutive_silence_hops = 0
    last_partial_text = ""
    pending_final_pad = 0.0  # seconds to wait after VAD stop for post-speech pad

    try:
        while True:
            msg = await ws.receive()

            # Binary audio frame
            if msg.get("bytes") is not None:
                audio_buffer += msg["bytes"]

                # process while we have at least one hop available.
                # We process in HOP_BYTES steps (overlap handled by keeping leftover)
                while len(audio_buffer) >= FRAME_BYTES:
                    window = bytes(audio_buffer[:FRAME_BYTES])  # current analysis window (0.5s)
                    # optionally resample and convert to float32
                    float_audio = pcm16_bytes_to_float32_array(window)
                    if client_sr != TARGET_SR:
                        float_audio = resample_linear(float_audio, client_sr, TARGET_SR)

                    # normalize to improve VAD & model accuracy
                    float_audio = normalize_audio(float_audio)

                    # VAD: analyze using smaller frames (10/20/30ms).
                    is_speech_window = True
                    if VAD_ENABLED:
                        is_speech_window = False
                        for f in frame_generator(VAD_FRAME_MS, window, TARGET_SR):
                            if len(f) < int(TARGET_SR * (VAD_FRAME_MS/1000.0)) * BYTES_PER_SAMPLE:
                                continue
                            if vad.is_speech(f, TARGET_SR):
                                is_speech_window = True
                                break

                    # If speech detected in this 0.5s window, reset silence counter; else increment
                    if is_speech_window:
                        consecutive_silence_hops = 0
                    else:
                        consecutive_silence_hops += 1

                    # Do inference when speech present (for partials) OR occasionally on silence to provide feedback
                    do_infer = is_speech_window

                    if do_infer:
                        # Run inference (synchronously inside lock)
                        async with inference_lock:
                            try:
                                segments = model.transcribe(float_audio)  # pass float32 array
                                text = segments_to_text(segments)
                            except Exception:
                                log.exception("Inference failed")
                                text = ""

                        # send partial if changed
                        if text and text.strip() and text != last_partial_text:
                            last_partial_text = text
                            await ws.send_json({"type": "partial", "text": text})

                    # advance buffer by HOP_BYTES (50% overlap)
                    audio_buffer = audio_buffer[HOP_BYTES:]
                    # ensure we keep KEEP_BYTES most recent for context
                    if len(audio_buffer) > KEEP_BYTES:
                        audio_buffer = audio_buffer[-KEEP_BYTES:]

                    # If sustained silence => schedule finalization (add a small post-speech pad)
                    if consecutive_silence_hops >= VAD_SILENCE_HANG:
                        # wait a small pad to catch trailing phonemes
                        pending_final_pad = POST_SPEECH_PAD_SECONDS
                        # break to outer loop to allow receiving more bytes or time to pass
                        break

                # if pending_final_pad > 0 we don't immediately finalize; finalization handled below using text frames or next iterations

            # Text control frame
            elif msg.get("text") is not None:
                try:
                    packet = json.loads(msg["text"])
                except Exception:
                    packet = {}

                if "sr" in packet:
                    try:
                        client_sr = int(packet["sr"])
                        log.info(f"Client sample rate set to {client_sr}")
                    except Exception:
                        pass
                # end-of-utterance requested explicitly by client
                if packet.get("end") is True:
                    # optionally wait a small post-speech pad to gather trailing audio
                    await asyncio.sleep(POST_SPEECH_PAD_SECONDS)
                    remaining = bytes(audio_buffer)
                    float_remaining = pcm16_bytes_to_float32_array(remaining)
                    if client_sr != TARGET_SR:
                        float_remaining = resample_linear(float_remaining, client_sr, TARGET_SR)
                    float_remaining = normalize_audio(float_remaining)
                    async with inference_lock:
                        segments = model.transcribe(float_remaining)
                        final_text = segments_to_text(segments)
                    if final_text and final_text.strip():
                        await ws.send_json({"type": "final", "text": final_text})
                    # reset buffer & counters
                    audio_buffer = bytearray()
                    consecutive_silence_hops = 0
                    last_partial_text = ""
                    continue

            # If we have pending_final_pad scheduled due to VAD silence, implement it here:
            if pending_final_pad > 0:
                # reduce pad using a short async sleep (non-blocking)
                await asyncio.sleep(min(pending_final_pad, HOP_SECONDS))
                pending_final_pad -= min(pending_final_pad, HOP_SECONDS)

                # After pad elapsed, finalize using current buffer
                if pending_final_pad <= 0:
                    remaining = bytes(audio_buffer)
                    float_remaining = pcm16_bytes_to_float32_array(remaining)
                    if client_sr != TARGET_SR:
                        float_remaining = resample_linear(float_remaining, client_sr, TARGET_SR)
                    float_remaining = normalize_audio(float_remaining)
                    async with inference_lock:
                        segments = model.transcribe(float_remaining)
                        final_text = segments_to_text(segments)
                    if final_text and final_text.strip():
                        await ws.send_json({"type": "final", "text": final_text})
                    # reset buffer & counters
                    audio_buffer = bytearray()
                    consecutive_silence_hops = 0
                    last_partial_text = ""
                    pending_final_pad = 0.0

            # Handle disconnect message from ASGI (rare)
            if msg.get("type") == "websocket.disconnect":
                log.info("Client disconnected")
                if audio_buffer:
                    remaining = bytes(audio_buffer)
                    float_remaining = pcm16_bytes_to_float32_array(remaining)
                    if client_sr != TARGET_SR:
                        float_remaining = resample_linear(float_remaining, client_sr, TARGET_SR)
                    float_remaining = normalize_audio(float_remaining)
                    async with inference_lock:
                        segments = model.transcribe(float_remaining)
                        final_text = segments_to_text(segments)
                    if final_text and final_text.strip():
                        # socket is closed, cannot send; log instead
                        log.info("Final after disconnect: %s", final_text)
                return

    except WebSocketDisconnect:
        log.info("Client disconnected (exception)")
        if audio_buffer:
            remaining = bytes(audio_buffer)
            float_remaining = pcm16_bytes_to_float32_array(remaining)
            if client_sr != TARGET_SR:
                float_remaining = resample_linear(float_remaining, client_sr, TARGET_SR)
            float_remaining = normalize_audio(float_remaining)
            async with inference_lock:
                segments = model.transcribe(float_remaining)
                final_text = segments_to_text(segments)
            if final_text and final_text.strip():
                log.info("Final after disconnect: %s", final_text)
    except Exception as e:
        log.exception("Unexpected server error")
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8765, log_level="info")
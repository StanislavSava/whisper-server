import ctypes
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WHISPER_PATH = ROOT.parent / "whisper.cpp"
LIB_PATH = WHISPER_PATH / "build" / "src" / "libwhisper.dylib"
MODEL_PATH = WHISPER_PATH / "models" / "ggml-base.en.bin"


# ------------------------------
# Whisper C struct definitions
# ------------------------------
class WhisperFullParams(ctypes.Structure):
    _fields_ = [
        ("strategy", ctypes.c_int),
        ("n_threads", ctypes.c_int),
        ("n_max_text_ctx", ctypes.c_int),
        ("offset_ms", ctypes.c_int),
        ("duration_ms", ctypes.c_int),
        ("translate", ctypes.c_bool),
        ("no_context", ctypes.c_bool),
        ("single_segment", ctypes.c_bool),
        ("print_special", ctypes.c_bool),
        ("print_progress", ctypes.c_bool),
        ("print_realtime", ctypes.c_bool),
        ("print_timestamps", ctypes.c_bool),
        ("token_timestamps", ctypes.c_bool),
        ("thold_pt", ctypes.c_float),
        ("thold_ptsum", ctypes.c_float),
        ("max_len", ctypes.c_int),
        ("split_on_word", ctypes.c_bool),

        # VAD settings
        ("vad_filter", ctypes.c_bool),
        ("vad_thold", ctypes.c_float),
        ("vad_prob_min", ctypes.c_float),
    ]

# ------------------------------
# Whisper.cpp Python wrapper
# ------------------------------
class WhisperCpp:
    def __init__(self):
        self.lib = ctypes.cdll.LoadLibrary(str(LIB_PATH))

        # return values & argument types
        self.lib.whisper_init_from_file.argtypes = [ctypes.c_char_p]
        self.lib.whisper_init_from_file.restype = ctypes.c_void_p

        self.ctx = self.lib.whisper_init_from_file(
            str(MODEL_PATH).encode("utf-8")
        )

        # default params
        self.lib.whisper_full_default_params.restype = WhisperFullParams
        self.lib.whisper_full_default_params.argtypes = [ctypes.c_int]

        self.params = self.lib.whisper_full_default_params(0)

        # enable built-in VAD
        self.params.vad_filter = True
        self.params.vad_thold = 0.6
        self.params.vad_prob_min = 0.1

        # transcription
        self.lib.whisper_full.restype = ctypes.c_int
        self.lib.whisper_full.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(WhisperFullParams),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]

        # extract results
        self.lib.whisper_full_n_segments.argtypes = [ctypes.c_void_p]
        self.lib.whisper_full_n_segments.restype = ctypes.c_int

        self.lib.whisper_full_get_segment_text.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        self.lib.whisper_full_get_segment_text.restype = ctypes.c_char_p

    def transcribe_pcm16(self, audio_pcm16: bytes):
        pcm = np.frombuffer(audio_pcm16, np.int16).astype(np.float32) / 32768.0

        ret = self.lib.whisper_full(
            self.ctx,
            ctypes.byref(self.params),
            pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(pcm),
        )

        if ret != 0:
            return None

        num = self.lib.whisper_full_n_segments(self.ctx)
        segments = []
        for i in range(num):
            text = self.lib.whisper_full_get_segment_text(self.ctx, i)
            segments.append(text.decode())

        return " ".join(segments) if segments else ""
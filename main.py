import logging
import os
import subprocess
import tempfile
import time
import uuid
import asyncio

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from transformers import AutoModel, pipeline

# Load environment variables from .env file if it exists
if os.path.exists(".env"):
    with open(".env", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                # Clean key and value
                key = key.strip()
                val = val.strip().strip("'\"")
                os.environ[key] = val



# ============================================================
# CONFIGURATION
# ============================================================

INDIC_MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
WHISPER_MODEL_ID = "openai/whisper-tiny"

ENGLISH_CODE = "en"

# "ctc" = fast, single forward pass, non-autoregressive.
# "rnnt" = slightly higher accuracy, autoregressive, slower.
# Default to ctc for low latency; can be overridden per-request.
# (Only applies to the Indic model — Whisper ignores this.)
DEFAULT_DECODING = "ctc"

UPLOAD_FOLDER = "uploads"

ALLOWED_EXTENSIONS = {
    "wav",
    "mp3",
    "m4a",
    "flac",
    "ogg",
    "webm",
    "mp4",
}

# Reject absurdly large uploads before they're fully buffered in memory.
MAX_FILE_SIZE_BYTES = 200 * 1024 * 1024  # 200 MB

# Kill ffmpeg if it hangs on a malformed/corrupt file.
FFMPEG_TIMEOUT_SECONDS = 120


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Multilingual STT API (IndicConformer + Whisper)",
    description="Local Speech-to-Text API",
    version="3.0.0",
)


# ============================================================
# UPLOAD DIRECTORY
# ============================================================

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    DEVICE = "cuda:0"

    # Autocast (mixed precision at inference time) is used instead
    # of casting the model's weights to fp16/bf16 directly, since
    # this is a custom trust_remote_code model and we can't be
    # sure every internal op handles low-precision weights safely
    # (we hit exactly this kind of silent-empty-output bug with
    # Whisper in plain fp16). Autocast keeps master weights in
    # fp32 and only runs eligible ops in lower precision, which is
    # much safer, while still giving most of the speed benefit.
    USE_AUTOCAST = torch.cuda.is_bf16_supported()
    AUTOCAST_DTYPE = torch.bfloat16 if USE_AUTOCAST else torch.float16

    logger.info("========================================")
    logger.info("CUDA AVAILABLE")
    logger.info("GPU: %s", torch.cuda.get_device_name(0))
    logger.info("CUDA version: %s", torch.version.cuda)
    logger.info("Autocast dtype: %s", AUTOCAST_DTYPE)
    logger.info("========================================")

else:

    DEVICE = "cpu"
    USE_AUTOCAST = False
    AUTOCAST_DTYPE = torch.float32

    logger.warning("CUDA NOT AVAILABLE")
    logger.warning("Using CPU")


# ============================================================
# LOAD MODELS (eager: both loaded once at startup)
# ============================================================

logger.info("Loading IndicConformer model (Indic languages)...")

indic_load_start = time.perf_counter()

indic_transcriber = AutoModel.from_pretrained(
    INDIC_MODEL_ID,
    trust_remote_code=True,
    token=os.environ.get("HF_TOKEN"),
)
indic_transcriber = indic_transcriber.to(DEVICE)
indic_transcriber.eval()

indic_model_load_time = time.perf_counter() - indic_load_start

logger.info(
    "IndicConformer model loaded in %.2f seconds",
    indic_model_load_time,
)

logger.info("Loading Whisper model (English)...")

whisper_load_start = time.perf_counter()

# Weights are kept in float32; speed comes from the autocast
# context applied around the actual inference call below, not
# from casting the model's weights to fp16 (that caused the
# earlier empty-transcript bug on this model).
whisper_transcriber = pipeline(
    task="automatic-speech-recognition",
    model=WHISPER_MODEL_ID,
    torch_dtype=torch.float32,
    device=DEVICE,
)

whisper_model_load_time = time.perf_counter() - whisper_load_start

logger.info(
    "Whisper model loaded in %.2f seconds",
    whisper_model_load_time,
)

# Combined figure kept around for the response payload's timing
# block, so existing API consumers don't lose the field.
model_load_time = indic_model_load_time + whisper_model_load_time


# ============================================================
# LANGUAGE MAP
# ============================================================

LANGUAGE_MAP = {
    "english": "en", "en": "en",

    "hindi": "hi", "hi": "hi",
    "bengali": "bn", "bangla": "bn", "bn": "bn",
    "tamil": "ta", "ta": "ta",
    "telugu": "te", "te": "te",
    "kannada": "kn", "kn": "kn",
    "malayalam": "ml", "ml": "ml",
    "marathi": "mr", "mr": "mr",
    "gujarati": "gu", "gu": "gu",
    "punjabi": "pa", "pa": "pa",
    "odia": "or", "oriya": "or", "or": "or",
    "assamese": "as", "as": "as",
    "urdu": "ur", "ur": "ur",
    "nepali": "ne", "ne": "ne",
    "sanskrit": "sa", "sa": "sa",
    "konkani": "kok", "kok": "kok",
    "maithili": "mai", "mai": "mai",
    "sindhi": "sd", "sd": "sd",
    "dogri": "doi", "doi": "doi",
    "bodo": "brx", "brx": "brx",
    "santali": "sat", "sat": "sat",
    # "en" now routes to Whisper (see run_transcription_pipeline);
    # every other code here routes to IndicConformer.
}

VALID_DECODING_STRATEGIES = {"ctc", "rnnt"}


# ============================================================
# FILE VALIDATION
# ============================================================

def is_allowed_file(filename: str) -> bool:

    if not filename:
        return False

    if "." not in filename:
        return False

    extension = filename.rsplit(".", 1)[1].lower()

    return extension in ALLOWED_EXTENSIONS


# ============================================================
# CONVERT AUDIO TO WAV
# ============================================================

def convert_to_wav(input_file: str, output_file: str):
    """
    Convert any supported media file into:

    PCM signed 16-bit
    16 kHz
    Mono WAV
    """

    command = [
        "ffmpeg",

        "-y",

        "-i",
        input_file,

        # Mono
        "-ac",
        "1",

        # 16 kHz
        "-ar",
        "16000",

        # PCM 16-bit
        "-c:a",
        "pcm_s16le",

        output_file,
    ]

    logger.info("Running FFmpeg conversion")

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=FFMPEG_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        logger.error("FFmpeg timed out after %s seconds", FFMPEG_TIMEOUT_SECONDS)
        raise RuntimeError(
            "FFmpeg timed out while decoding the uploaded audio."
        ) from exc

    if result.returncode != 0:

        logger.error(
            "FFmpeg failed:\n%s",
            result.stderr,
        )

        raise RuntimeError(
            "FFmpeg failed to decode the uploaded audio."
        )

    if not os.path.exists(output_file):

        raise RuntimeError(
            "FFmpeg did not create the output WAV file."
        )

    logger.info(
        "Audio converted successfully: %s",
        output_file,
    )


# ============================================================
# LOAD WAV INTO NUMPY
# ============================================================

def load_wav_audio(wav_path: str):
    """
    Load 16 kHz mono WAV using Python's wave module.
    Returns float32 numpy array.
    """

    import wave

    with wave.open(wav_path, "rb") as wav:

        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frames = wav.getnframes()

        logger.info(
            "WAV information: sample_rate=%s channels=%s "
            "sample_width=%s frames=%s",
            sample_rate,
            channels,
            sample_width,
            frames,
        )

        audio_bytes = wav.readframes(frames)

    if sample_rate != 16000:

        raise RuntimeError(
            f"Unexpected sample rate: {sample_rate}"
        )

    if channels != 1:

        raise RuntimeError(
            f"Expected mono audio, got {channels} channels."
        )

    if sample_width != 2:

        raise RuntimeError(
            f"Expected 16-bit audio, got {sample_width * 8}-bit."
        )

    # Convert int16 -> float32
    audio = np.frombuffer(
        audio_bytes,
        dtype=np.int16,
    ).astype(np.float32)

    audio /= 32768.0

    return audio


# ============================================================
# AUDIO VALIDATION
# ============================================================

def validate_audio(audio: np.ndarray):

    if audio is None:

        raise RuntimeError(
            "Audio data is None."
        )

    if len(audio) == 0:

        raise RuntimeError(
            "Audio contains zero samples."
        )

    duration = len(audio) / 16000

    min_value = float(np.min(audio))
    max_value = float(np.max(audio))
    rms = float(np.sqrt(np.mean(audio ** 2)))

    logger.info(
        "Audio duration: %.2f seconds",
        duration,
    )

    logger.info(
        "Audio min: %.6f max: %.6f RMS: %.6f",
        min_value,
        max_value,
        rms,
    )

    # Detect completely silent audio
    if rms < 0.0001:

        raise RuntimeError(
            "The uploaded audio appears to be silent."
        )

    return duration


# ============================================================
# BLOCKING INFERENCE CALL (run off the event loop)
# ============================================================

def run_indic_inference(
    audio_array: np.ndarray,
    language_code: str,
    decoding: str,
) -> str:
    """
    Blocking call to the IndicConformer model. Expects a torch
    tensor shaped (1, num_samples) at 16 kHz, plus an explicit
    language code and decoding strategy ("ctc" is fast/
    non-autoregressive, "rnnt" is slower but sometimes slightly
    more accurate).
    """

    wav_tensor = torch.from_numpy(audio_array).unsqueeze(0).to(DEVICE)

    # inference_mode disables autograd bookkeeping entirely (faster
    # and lower memory than the older no_grad()).
    with torch.inference_mode():

        if USE_AUTOCAST:
            with torch.autocast(device_type="cuda", dtype=AUTOCAST_DTYPE):
                transcription = indic_transcriber(wav_tensor, language_code, decoding)
        else:
            transcription = indic_transcriber(wav_tensor, language_code, decoding)

    return str(transcription).strip()


def run_whisper_inference(audio_array: np.ndarray) -> str:
    """
    Blocking call to the Whisper pipeline, used for English only.
    Same autocast approach as IndicConformer: weights stay in
    fp32, autocast handles the mixed-precision speedup so we don't
    reintroduce the empty-transcript issue plain fp16 caused
    earlier.
    """

    with torch.inference_mode():

        if USE_AUTOCAST:
            with torch.autocast(device_type="cuda", dtype=AUTOCAST_DTYPE):
                result = whisper_transcriber(
                    {"raw": audio_array, "sampling_rate": 16000},
                    generate_kwargs={"task": "transcribe", "language": "en"},
                    return_timestamps=True,
                    chunk_length_s=30,
                    stride_length_s=(5, 5),
                )
        else:
            result = whisper_transcriber(
                {"raw": audio_array, "sampling_rate": 16000},
                generate_kwargs={"task": "transcribe", "language": "en"},
                return_timestamps=True,
                chunk_length_s=30,
                stride_length_s=(5, 5),
            )

    return result.get("text", "").strip()


def run_transcription_pipeline(
    audio_array: np.ndarray,
    language_code: str,
    decoding: str,
) -> str:
    """
    Dispatcher: routes English to Whisper, everything else to
    IndicConformer. This is the only place that needs to change
    if a third engine is added later for some other language.
    """

    if language_code == ENGLISH_CODE:
        return run_whisper_inference(audio_array)

    return run_indic_inference(audio_array, language_code, decoding)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health():

    return {
        "success": True,
        "status": "healthy",
        "models": {
            "indic": INDIC_MODEL_ID,
            "english": WHISPER_MODEL_ID,
        },
        "device": DEVICE,
        "cpu": {
            "available": True
        },
        "gpu": {
            "available": torch.cuda.is_available(),
            "name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        },
    }


# ============================================================
# TRANSCRIPTION
# ============================================================

@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    language: str = Form(...),
    decoding: str = Form(default=DEFAULT_DECODING),
):

    request_start = time.perf_counter()

    original_path = None
    wav_path = None
    conversion_time = 0.0
    inference_time = 0.0

    try:

        # ====================================================
        # VALIDATE FILE
        # ====================================================

        if not audio.filename:

            raise HTTPException(
                status_code=400,
                detail="Audio filename is missing.",
            )

        if not is_allowed_file(audio.filename):

            raise HTTPException(
                status_code=400,
                detail=(
                    "Unsupported audio format. "
                    "Allowed formats: wav, mp3, m4a, flac, "
                    "ogg, webm, mp4."
                ),
            )

        # ====================================================
        # LANGUAGE
        # ====================================================

        language_key = language.strip().lower()

        resolved_language = LANGUAGE_MAP.get(language_key)

        if resolved_language is None:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Unsupported language. Supported codes: "
                    + ", ".join(sorted(set(LANGUAGE_MAP.values())))
                ),
            )

        is_english = resolved_language == ENGLISH_CODE

        # ====================================================
        # DECODING STRATEGY
        # ====================================================
        # Only meaningful for IndicConformer — Whisper (English)
        # ignores it entirely, so skip validation for English
        # requests rather than forcing the client to send a value
        # that has no effect.

        decoding_key = decoding.strip().lower()

        if not is_english and decoding_key not in VALID_DECODING_STRATEGIES:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Unsupported decoding strategy. "
                    "Use: ctc (fast) or rnnt (slower, sometimes "
                    "more accurate)."
                ),
            )

        # ====================================================
        # SAVE ORIGINAL FILE
        # ====================================================

        extension = audio.filename.rsplit(
            ".",
            1
        )[1].lower()

        # UUID avoids filename collisions between concurrent
        # requests that a millisecond timestamp would not.
        request_id = uuid.uuid4().hex

        original_filename = (
            f"audio_{request_id}.{extension}"
        )

        original_path = os.path.join(
            UPLOAD_FOLDER,
            original_filename,
        )

        file_data = await audio.read()

        if not file_data:

            raise HTTPException(
                status_code=400,
                detail="Uploaded audio file is empty.",
            )

        if len(file_data) > MAX_FILE_SIZE_BYTES:

            raise HTTPException(
                status_code=413,
                detail=(
                    "Uploaded audio file is too large "
                    f"(max {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB)."
                ),
            )

        with open(
            original_path,
            "wb"
        ) as file:

            file.write(file_data)

        logger.info(
            "Received file: %s",
            audio.filename,
        )

        logger.info(
            "File size: %.2f MB",
            len(file_data) / (1024 * 1024),
        )

        # ====================================================
        # CREATE TEMP WAV
        # ====================================================

        wav_filename = (
            f"audio_{request_id}_converted.wav"
        )

        wav_path = os.path.join(
            UPLOAD_FOLDER,
            wav_filename,
        )

        # ====================================================
        # FFMPEG CONVERSION (offloaded: blocking subprocess call)
        # ====================================================

        conversion_start = time.perf_counter()

        await asyncio.to_thread(
            convert_to_wav,
            original_path,
            wav_path,
        )

        conversion_time = (
            time.perf_counter()
            - conversion_start
        )

        logger.info(
            "FFmpeg conversion time: %.3f seconds",
            conversion_time,
        )

        # ====================================================
        # LOAD AUDIO
        # ====================================================

        audio_array = load_wav_audio(
            wav_path
        )

        # ====================================================
        # VALIDATE AUDIO
        # ====================================================

        duration = validate_audio(
            audio_array
        )

        # ====================================================
        # TRANSCRIPTION (offloaded: blocking model inference)
        # ====================================================

        engine_name = "whisper" if is_english else "indic_conformer"

        logger.info(
            "Starting %s inference (lang=%s, decoding=%s)...",
            engine_name,
            resolved_language,
            "n/a" if is_english else decoding_key,
        )

        inference_start = time.perf_counter()

        result = await asyncio.to_thread(
            run_transcription_pipeline,
            audio_array,
            resolved_language,
            decoding_key,
        )

        logger.info(
            "Transcription result: %s",
            result,
        )

        inference_time = (
            time.perf_counter()
            - inference_start
        )

        # ====================================================
        # EXTRACT TEXT
        # ====================================================

        # IndicConformer returns a plain string directly (unlike
        # the Whisper pipeline, which returned a dict).
        transcript = str(result).strip()

        # ====================================================
        # TOTAL TIME
        # ====================================================

        total_time = (
            time.perf_counter()
            - request_start
        )

        logger.info(
            "========================================"
        )

        logger.info(
            "TRANSCRIPTION RESULT"
        )

        logger.info(
            "Transcript: %s",
            transcript,
        )

        logger.info(
            "Audio duration: %.2f seconds",
            duration,
        )

        logger.info(
            "Conversion: %.3f seconds",
            conversion_time,
        )

        logger.info(
            "Inference: %.3f seconds",
            inference_time,
        )

        logger.info(
            "Total: %.3f seconds",
            total_time,
        )

        logger.info(
            "========================================"
        )

        # ====================================================
        # EMPTY TRANSCRIPT
        # ====================================================

        if not transcript:

            return {
                "success": False,
                "model": engine_name,
                "device": DEVICE,
                "language": resolved_language,
                "decoding": None if is_english else decoding_key,
                "transcript": "",
                "message": (
                    "No speech was detected in the audio."
                ),
                "timing": {
                    "model_load_seconds": round(
                        model_load_time,
                        3,
                    ),
                    "conversion_seconds": round(
                        conversion_time,
                        3,
                    ),
                    "inference_seconds": round(
                        inference_time,
                        3,
                    ),
                    "total_request_seconds": round(
                        total_time,
                        3,
                    ),
                },
            }

        # ====================================================
        # RESPONSE
        # ====================================================

        return {
            "success": True,
            "model": engine_name,
            "device": DEVICE,
            "language": resolved_language,
            "decoding": None if is_english else decoding_key,
            "transcript": transcript,
            "audio": {
                "duration_seconds": round(
                    duration,
                    2,
                ),
                "sample_rate": 16000,
                "channels": 1,
            },
            "timing": {
                "model_load_seconds": round(
                    model_load_time,
                    3,
                ),
                "conversion_seconds": round(
                    conversion_time,
                    3,
                ),
                "inference_seconds": round(
                    inference_time,
                    3,
                ),
                "total_request_seconds": round(
                    total_time,
                    3,
                ),
            },
        }

    except HTTPException:

        raise

    except Exception as exc:

        logger.exception(
            "Transcription failed"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    finally:

        # ====================================================
        # DELETE ORIGINAL FILE
        # ====================================================

        if (
            original_path
            and os.path.exists(original_path)
        ):

            try:

                os.remove(original_path)

            except OSError:

                logger.warning(
                    "Unable to delete: %s",
                    original_path,
                )

        # ====================================================
        # DELETE WAV FILE
        # ====================================================

        if (
            wav_path
            and os.path.exists(wav_path)
        ):

            try:

                os.remove(wav_path)

            except OSError:

                logger.warning(
                    "Unable to delete: %s",
                    wav_path,
                )


# ============================================================
# APPLICATION START
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
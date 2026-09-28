from pathlib import Path

import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as audio_functional
from torch import nn

CLASS_NAMES = ("clap", "knock", "sing", "spoon", "whistle")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}

SAMPLE_RATE = 16_000
DURATION_SECONDS = 1.5
NUM_SAMPLES = int(SAMPLE_RATE * DURATION_SECONDS)
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
N_MELS = 64


def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    return waveform if waveform.shape[0] == 1 else waveform.mean(dim=0, keepdim=True)


def fix_length(waveform: torch.Tensor, target_samples: int = NUM_SAMPLES) -> torch.Tensor:
    current = waveform.shape[-1]
    if current > target_samples:
        start = (current - target_samples) // 2
        return waveform[..., start : start + target_samples]
    if current < target_samples:
        return torch.nn.functional.pad(waveform, (0, target_samples - current))
    return waveform


def load_waveform(path: str | Path) -> torch.Tensor:
    audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = to_mono(torch.from_numpy(audio.T))
    if source_rate != SAMPLE_RATE:
        waveform = audio_functional.resample(waveform, source_rate, SAMPLE_RATE)
    return fix_length(waveform)


class LogMelExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            win_length=WIN_LENGTH,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            power=2.0,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return torch.log(self.mel(waveform) + 1e-6)


def preprocess_audio(path: str | Path) -> torch.Tensor:
    """Подготавливает один вход модели размером [1, 64, 151]."""
    return LogMelExtractor()(load_waveform(path))

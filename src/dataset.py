import random
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocessing import CLASS_TO_INDEX, LogMelExtractor, load_waveform

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_SESSIONS = {
    "s02_phone_near",
    "s03_pc_far",
    "s04_phone_far",
    "s06_pc_other_room",
    "s07_phone_ambient",
    "s08_pc_ambient",
}
VALIDATION_SESSIONS = {"s01_pc_near"}
TEST_SESSIONS = {"s05_phone_r2_near"}


def shift_with_zeros(waveform: torch.Tensor, shift: int) -> torch.Tensor:
    if shift == 0:
        return waveform
    result = torch.zeros_like(waveform)
    if shift > 0:
        result[..., shift:] = waveform[..., :-shift]
    else:
        result[..., :shift] = waveform[..., -shift:]
    return result


class TrainWaveformAugment:
    """Итоговый набор аугментаций, проверенный в эксперименте."""

    def __init__(self, shift_probability: float = 0.5, gain_probability: float = 0.5) -> None:
        self.shift_probability = shift_probability
        self.gain_probability = gain_probability

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if random.random() < self.shift_probability:
            max_shift = int(waveform.shape[-1] * 0.20)
            waveform = shift_with_zeros(waveform, random.randint(-max_shift, max_shift))
        if random.random() < self.gain_probability:
            gain_db = random.uniform(-9.0, 9.0)
            waveform = waveform * (10.0 ** (gain_db / 20.0))
        peak = waveform.abs().max()
        return waveform / peak if peak > 1.0 else waveform


def load_metadata(path: str | Path = PROJECT_ROOT / "metadata.csv") -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"path", "class", "session"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")
    unknown_classes = set(frame["class"]) - set(CLASS_TO_INDEX)
    if unknown_classes:
        raise ValueError(f"Unknown classes: {sorted(unknown_classes)}")
    return frame


def split_metadata(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    known_sessions = TRAIN_SESSIONS | VALIDATION_SESSIONS | TEST_SESSIONS
    unknown = set(frame["session"]) - known_sessions
    if unknown:
        raise ValueError(f"Unknown sessions: {sorted(unknown)}")
    return {
        "train": frame[frame["session"].isin(TRAIN_SESSIONS)].reset_index(drop=True),
        "validation": frame[frame["session"].isin(VALIDATION_SESSIONS)].reset_index(drop=True),
        "test": frame[frame["session"].isin(TEST_SESSIONS)].reset_index(drop=True),
    }


class AudioDataset(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        data_root: str | Path = PROJECT_ROOT / "data/raw",
        training: bool = False,
    ) -> None:
        self.metadata = metadata.reset_index(drop=True).copy()
        self.data_root = Path(data_root)
        self.augment = TrainWaveformAugment() if training else None
        self.extract_features = LogMelExtractor()

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.metadata.iloc[index]
        path = Path(row["path"])
        waveform = load_waveform(path if path.is_absolute() else self.data_root / path)
        if self.augment is not None:
            waveform = self.augment(waveform)
        features = self.extract_features(waveform)
        target = torch.tensor(CLASS_TO_INDEX[row["class"]], dtype=torch.long)
        return features, target

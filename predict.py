import argparse
from pathlib import Path

import torch
from src.models import build_model
from src.preprocessing import CLASS_NAMES, preprocess_audio

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINTS = {
    "cnn": PROJECT_ROOT / "checkpoints/cnn_width8.pt",
    "crnn": PROJECT_ROOT / "checkpoints/crnn_micro.pt",
}


def predict(path: Path, model_name: str, device: torch.device) -> tuple[str, float]:
    model = build_model(model_name).to(device)
    model.load_state_dict(torch.load(CHECKPOINTS[model_name], map_location=device, weights_only=True))
    model.eval()
    features = preprocess_audio(path).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model(features).softmax(dim=1)[0]
    confidence, index = probabilities.max(dim=0)
    return CLASS_NAMES[index.item()], confidence.item()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify a WAV file with CNN, CRNN, or both models.")
    parser.add_argument("audio", type=Path, help="Path to an audio file")
    parser.add_argument("--model", choices=("cnn", "crnn", "both"), default="both")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.audio.is_file():
        raise SystemExit(f"Audio file not found: {args.audio}")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    selected = ("cnn", "crnn") if args.model == "both" else (args.model,)
    for model_name in selected:
        label, confidence = predict(args.audio, model_name, device)
        print(f"{model_name.upper()}: {label} (confidence={confidence:.3f})")


if __name__ == "__main__":
    main()

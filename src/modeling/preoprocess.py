from pathlib import Path
from typing import Iterator

import librosa
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch import nn
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
)


class TextVectorizer(nn.Module):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, input):
        encoded_input = self.tokenizer(
            input, padding=True, truncation=True, return_tensors="pt"
        )  # type: ignore

        with torch.no_grad():
            out = self.model(**encoded_input)

        # Mean pool over non-padding tokens → (batch, hidden_dim)
        mask = (
            encoded_input["attention_mask"].unsqueeze(-1).expand(out[0].size()).float()
        )
        mean_pooled = torch.sum(out[0] * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        # normalize
        return F.normalize(mean_pooled, p=2, dim=1)


class AudioVectorizer(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base"):
        super().__init__()
        self.model = AutoModel.from_pretrained("facebook/wav2vec2-base")

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            out = self.model(input_values=input_values)

        hidden = out.last_hidden_state
        return hidden.mean(dim=1)


class ImageVectorizer(nn.Module):
    def __init__(self, model_name="facebook/dinov3-vitl16-pretrain-lvd1689m"):
        super().__init__()

        self.model = AutoModel.from_pretrained(
            model_name,
            device_map="auto",
        )
        self.processor = AutoImageProcessor.from_pretrained(model_name)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.pooler_output


def extract_filename_from_id(id: str) -> str:
    return id.split("/")[-1].split(".")[0]


def load_audio_from_id(id: str, audio_dir: Path, sr: int = 16000) -> torch.Tensor:
    audio_path = (
        audio_dir / id.rpartition("/")[0] / f"{extract_filename_from_id(id)}.flac"
    )
    waveform, _ = librosa.load(audio_path, sr=sr, mono=True)
    # add any alterations youd like to make to the audio here
    tensor = torch.from_numpy(waveform)
    return tensor.unsqueeze(0)  # [1, length]


def load_annotations_from_id(id: str, annotations_dir: Path) -> dict:
    annotation_path = annotations_dir / id / f"{extract_filename_from_id(id)}.yml"

    def _construct_python_tuple(loader, node):
        return tuple(loader.construct_sequence(node))

    yaml.SafeLoader.add_constructor(
        "tag:yaml.org,2002:python/tuple",
        _construct_python_tuple,
    )

    with open(annotation_path, "r") as f:
        return yaml.safe_load(f)


def iter_frames_from_id(id: str, frames_dir: str | Path) -> Iterator[Image.Image]:
    frames_dir = Path(frames_dir) / id
    n = 0
    while True:
        path = frames_dir / f"frame-{n}.jpg"
        if not path.exists():
            return
        yield Image.open(path).convert("RGB")
        n += 1


def main():
    audio_model = AudioVectorizer()
    text_model = TextVectorizer()
    # image_model = ImageVectorizer()

    audio_dir = Path("data/interim/Audio")
    annotations_dir = Path("data/raw/data/transcription")
    # frames_dir = Path("data/raw/data/cropped-aligned-faces")

    video_df = pl.read_csv("data/raw/data/bah-video.csv")
    video_df = video_df.filter(pl.col("video-path").str.contains("82553|82554|82555"))

    schema = pa.schema(
        [
            ("id", pa.string()),
            ("audio_emb", pa.list_(pa.float32())),
            ("text_emb", pa.list_(pa.float32())),
            ("label", pa.int32()),
        ]
    )
    writer = pq.ParquetWriter("data/processed/text_audio.parquet", schema)

    for video in video_df.iter_rows():
        video_id = video[0]
        label = video[1]

        audio_tensor = load_audio_from_id(video_id, audio_dir)
        transcription_dict = load_annotations_from_id(video_id, annotations_dir)
        # frames = iter_frames_from_id(video_id, frames_dir)

        audio_emb = audio_model.forward(audio_tensor)
        transcription_emb = text_model.forward(transcription_dict.get("text"))

        batch = pa.record_batch(
            {
                "id": [video_id],
                "audio_emb": [audio_emb.squeeze().cpu().numpy().tolist()],
                "text_emb": [transcription_emb.squeeze().cpu().numpy().tolist()],
                "label": [label],
            },
            schema=schema,
        )

        writer.write_batch(batch)

    writer.close()


if __name__ == "__main__":
    main()

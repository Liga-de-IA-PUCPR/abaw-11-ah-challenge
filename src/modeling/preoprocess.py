import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor, AutoFeatureExtractor
from transformers.image_utils import load_image




class TextVectorizer(nn.Module):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, input, attention_mask):
        encoded_input = self.tokenizer(input, padding=True, truncation=True, return_tensors='pt')

        with torch.no_grad():
            out = self.model(**encoded_input)

        # Mean pool over non-padding tokens → (batch, hidden_dim)
        mask = attention_mask.unsqueeze(-1).expand(out[0].size()).float()
        mean_pooled = torch.sum(out[0] * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        # normalize
        return F.normalize(mean_pooled, p=2, dim=1)


class AudioVectorizer(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        out = self.model(input_values=input_values, attention_mask=attention_mask)
        # Mean pool over time → (batch, hidden_dim)
        hidden = out.last_hidden_state          # (batch, T, hidden_dim)
        if attention_mask is not None:
            # Build a mask aligned to the encoder's output length
            mask = self.model._get_feature_vector_attention_mask(hidden.shape[1], attention_mask)
            mask = mask.unsqueeze(-1).float()
            return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
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
    return id.split('/')[-1].split('.')[0]

def load_audio_from_id(id: str, audio_dir: Path, sr: int = 16000) -> torch.Tensor:
    audio_path = audio_dir / id / f"{extract_filename_from_id(id)}.flac"
    waveform, sr = torchaudio.load(audio_path)
    waveform = waveform.mean(dim=0)  # stereo → mono
    return waveform


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

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


def main():

    waveforms = load_waveforms()
    audio_feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

    # waveforms: list of 1-D numpy arrays at 16 kHz
    inputs = audio_feature_extractor(
        waveforms,
        sampling_rate=16_000,
        return_tensors="pt",
        padding=True,      
    )

    audio_model = AudioVectorizer()
    vectors = audio_model(inputs)

    imgs:list[str] = load_images()
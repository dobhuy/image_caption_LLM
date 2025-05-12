from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    CLIPModel,
    CLIPProcessor,
    PretrainedConfig,
)
import gc
import torch.nn as nn
import torch
from datasets import Dataset, Image as HuggingFaceImage
import evaluate
import json
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import nltk
nltk.download("wordnet")

class CLIPPhoGPTImageCaptioningConfig(PretrainedConfig):
    model_type = "image_captioning"

    def __init__(
        self,
        clip_model_name="openai/clip-vit-base-patch32",
        phogpt_model_name="vinai/PhoGPT-4B",
        max_caption_length=140,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.clip_model_name = clip_model_name
        self.phogpt_model_name = phogpt_model_name
        self.max_caption_length = max_caption_length


class CLIPPhoGPTImageCaptioningModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(config.clip_model_name)
        self.phogpt = AutoModelForCausalLM.from_pretrained(config.phogpt_model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(config.phogpt_model_name, trust_remote_code=True)

        clip_output_dim = self.clip.visual_projection.weight.size(0)
        phogpt_input_dim = self.phogpt.get_input_embeddings().embedding_dim
        self.projection = nn.Linear(clip_output_dim, phogpt_input_dim)

        # Freeze CLIP model
        for param in self.clip.parameters():
            param.requires_grad = False

    def forward(self, images, captions=None):
        # Encode images using CLIP
        image_features = self.clip.get_image_features(images)
        image_embeddings = self.projection(image_features)

        if captions is not None:
            # Training mode
            input_ids = captions
            decoder_inputs_embeds = self.phogpt.transformer.wte(input_ids) + image_embeddings.unsqueeze(1)
            outputs = self.phogpt(
                inputs_embeds=decoder_inputs_embeds,
                labels=input_ids,
            )
            return outputs.loss
        else:
            raise NotImplementedError("For generation, use the `generate` method.")

    def generate(self, images, max_length=140):
        # Encode images using CLIP
        image_features = self.clip.get_image_features(images)
        image_embeddings = self.projection(image_features)

        # Generate captions using PhoGPT
        input_ids = self.tokenizer("<start>", return_tensors="pt").input_ids.to(images.device)
        generated_ids = self.phogpt.generate(
            inputs_embeds=image_embeddings.unsqueeze(1),
            max_length=max_length,
        )
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
config = CLIPPhoGPTImageCaptioningConfig(
    clip_model_name="openai/clip-vit-base-patch32",
    phogpt_model_name="vinai/PhoGPT-4B",
    max_caption_length=140,
)
model = CLIPPhoGPTImageCaptioningModel(config)
torch.cuda.empty_cache()
gc.collect()
model.load_state_dict(torch.load("/kaggle/input/phogpt_frozen_clip/pytorch/default/1/PhoGPT_frozen_Clip.pth"),strict=False)
torch.cuda.empty_cache()
gc.collect()
model = model.to(device)
model.eval()
torch.cuda.empty_cache()
gc.collect()
processor = CLIPProcessor.from_pretrained(config.clip_model_name)

with open(f"/kaggle/input/ktvic-dataset/ktvic_dataset/test_data.json", encoding="utf8") as f:
    test_data = json.load(f)

image_metadata = test_data["images"]
annotations = test_data["annotations"]

image_index = {
    image["id"]: (f"/kaggle/input/ktvic-dataset/ktvic_dataset/public-test-images/{image['filename']}", [])
    for image in image_metadata
}

for x in annotations:
    image_index[x["image_id"]][1].append(x["caption"])

image_paths = []
captions = []

for image_path, caption_list in image_index.values():
    image_paths.append(image_path)
    captions.append(caption_list)

test_dict = {
    "images": image_paths,
    "captions": captions,
}

test_df = pd.DataFrame(test_dict)
test_dataset = Dataset.from_pandas(test_df).cast_column("images", HuggingFaceImage())

def test_transforms(example_batch):
    preprocess_image = processor(
        images=example_batch["images"], padding=True, return_tensors="pt"
    )
    return {
        "images": preprocess_image["pixel_values"],
        "captions": example_batch["captions"],
    }

test_preprocess_dataset = test_dataset.map(test_transforms, batched=True, batch_size=16)
test_preprocess_dataset.set_format("torch")
test_captions = test_preprocess_dataset["captions"]
test_dataloader = DataLoader(test_preprocess_dataset, batch_size=4, shuffle=False)
predicted_output = []
for batch in tqdm(test_dataloader):
    # Lấy batch ảnh và chuyển qua thiết bị
    test_images = batch["images"].to(device)

    with torch.no_grad():
        output = model.generate(test_images)
        predicted_output.extend(output)
        
    del test_images
    torch.cuda.empty_cache()
bleu = evaluate.load("bleu")
results_bleu_1 = bleu.compute(predictions=predicted_output, references=test_captions, max_order=1)
print(results_bleu_1)

results_bleu_4 = bleu.compute(predictions=predicted_output, references=test_captions)
print(results_bleu_4)

cider = evaluate.load("Kamichanw/CIDEr")
results_cider = cider.compute(predictions=predicted_output, references=test_captions)
print(results_cider)

meteor = evaluate.load("meteor")
results_meteor = meteor.compute(predictions=predicted_output, references=test_captions)
print(results_meteor)

rouge = evaluate.load("rouge")
results_rouge = rouge.compute(predictions=predicted_output, references=test_captions)
print(results_rouge)

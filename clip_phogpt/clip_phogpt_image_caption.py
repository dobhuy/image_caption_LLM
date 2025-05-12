import torch
from transformers import CLIPModel, CLIPProcessor, AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import json
from datasets import Dataset
from torch.utils.data import Dataset, DataLoader, random_split
import os
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import gc

# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from nltk.translate.meteor_score import meteor_score
# from rouge_score import rouge_scorer
# from pycocoevalcap.cider.cider import Cider
import nltk
nltk.download('wordnet')

torch.cuda.empty_cache()
gc.collect()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
clip_preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

tokenizer = AutoTokenizer.from_pretrained("vinai/PhoGPT-4B", trust_remote_code=True)
phogpt = AutoModelForCausalLM.from_pretrained("vinai/PhoGPT-4B", trust_remote_code=True)
torch.cuda.empty_cache()
gc.collect()
def load_dataset(dataset_root, json_type, type):
    with open(os.path.join(dataset_root, json_type), encoding='utf-8') as f:
        data_json = json.load(f)

    images_json = data_json['images']
    caption_json = data_json['annotations']
    images_path = []
    captions = []
    images_dict = {i['id']: i['filename'] for i in images_json}
    for l in caption_json:
        img_id = l['image_id']
        if img_id in images_dict:
            images_path.append(f"{dataset_root}/{type}/{images_dict[img_id]}")
        captions.append(l['caption'])
    return images_path, captions

dataset_root = '/data/cndt_thangcpd/HuyDB/ktvic_dataset'
images_train, captions_train = load_dataset(dataset_root, 'train_data.json', 'train-images')
images_test, captions_test = load_dataset(dataset_root, 'test_data.json', 'public-test-images')

class CaptionDataset(Dataset):
    def __init__(self, image_paths, captions, clip_processor):
        self.image_paths = image_paths
        self.captions = captions
        self.processor = clip_processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        caption = self.captions[idx]
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0), caption

train_dataset = CaptionDataset(image_paths=images_train, captions=captions_train, clip_processor=clip_preprocess)
test_dataset = CaptionDataset(image_paths=images_test, captions=captions_test, clip_processor=clip_preprocess)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True, num_workers=2)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True, pin_memory=True)

# def calculate_metrics(references, predictions):
#     tokenized_references = [[ref.split()] for ref in references]
#     tokenized_predictions = [pred.split() for pred in predictions]

#     smooth_fn = SmoothingFunction().method1
#     bleu1_scores = [
#         sentence_bleu(ref, pred, weights=(1, 0, 0, 0), smoothing_function=smooth_fn)
#         for ref, pred in zip(tokenized_references, tokenized_predictions)
#     ]
#     bleu4_scores = [
#         sentence_bleu(ref, pred, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_fn)
#         for ref, pred in zip(tokenized_references, tokenized_predictions)
#     ]
#     meteor_scores = [
#         meteor_score(ref, pred)
#         for ref, pred in zip(tokenized_references, tokenized_predictions)
#     ]
#     scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
#     rouge_scores = [
#         scorer.score(" ".join(ref[0]), " ".join(pred))['rougeL'].fmeasure
#         for ref, pred in zip(tokenized_references, tokenized_predictions)
#     ]
#     cider = Cider()
#     gts = {i: [" ".join(ref[0])] for i, ref in enumerate(tokenized_references)}
#     res = {i: [" ".join(pred)] for i, pred in enumerate(tokenized_predictions)}
#     cider_score, _ = cider.compute_score(gts, res)

#     metrics = {
#         'BLEU-1': sum(bleu1_scores) / len(bleu1_scores),
#         'BLEU-4': sum(bleu4_scores) / len(bleu4_scores),
#         'METEOR': sum(meteor_scores) / len(meteor_scores),
#         'ROUGE-L': sum(rouge_scores) / len(rouge_scores),
#         'CIDEr': cider_score,
#     }
#     return metrics

class ImageCaptioningModel(nn.Module):
    def __init__(self, clip_model, phogpt_model, tokenizer):
        super(ImageCaptioningModel, self).__init__()
        # Load CLIP and PhoGPT
        self.clip = clip_model
        self.phogpt = phogpt_model
        self.tokenizer = tokenizer
        
        # Projection layer to map CLIP image embeddings to PhoGPT's input space
        self.projection = nn.Linear(self.clip.visual_projection.weight.size(0), 
                                    self.phogpt.get_input_embeddings().embedding_dim)

    def forward(self, images, captions=None):
        # Encode images
        image_features = self.clip.get_image_features(images)
        image_embeddings = self.projection(image_features)  # Shape: [batch_size, hidden_dim]
        
        # Prepare inputs for PhoGPT
        decoder_inputs_embeds = image_embeddings.unsqueeze(1)  # Add sequence dimension (Shape: [batch_size, 1, hidden_dim])
    
        if captions is not None:
            input_ids = captions  # Captions are already tokenized in the training loop
    
            # Generate embeddings for input_ids (token labels)
            token_embeddings = self.phogpt.get_input_embeddings()(input_ids)  # Shape: [batch_size, seq_len, hidden_dim]
    
            # Concatenate image embeddings and token embeddings
            decoder_inputs_embeds = torch.cat((decoder_inputs_embeds, token_embeddings[:, :-1, :]), dim=1)
    
            # Generate logits from PhoGPT
            outputs = self.phogpt(inputs_embeds=decoder_inputs_embeds, labels=input_ids)
    
            logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
            _labels = input_ids.clone()  # Shape: [batch_size, seq_len]
            
            # Set the first token of each sequence in the batch to -100
            _labels[:, 0] = -100
    
            # Align sequence lengths
            seq_len = min(logits.size(1), _labels.size(1))
            logits = logits[:, :seq_len, :]  # Align sequence length of logits
            _labels = _labels[:, :seq_len]  # Align sequence length of labels
    
            # Flatten tensors for cross-entropy loss
            logits = logits.view(-1, logits.size(-1))  # [batch_size * seq_len, vocab_size]
            _labels = _labels.view(-1)  # [batch_size * seq_len]
    
            # Compute loss
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits, _labels)
            return loss
    
        else:
            # Generate captions during inference
            decoder_input_ids = self.tokenizer("<start>", return_tensors="pt").input_ids.to(images.device)
            generated_ids = self.phogpt.generate(
                inputs_embeds=decoder_inputs_embeds,
                max_length=140,
            )
            return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)


for param in clip_model.parameters():
    param.requires_grad = False
model = ImageCaptioningModel(clip_model=clip_model, phogpt_model=phogpt, tokenizer=tokenizer)
model = model.to(device)
torch.cuda.empty_cache()
gc.collect()
optimizer = Adam(model.parameters(), lr=2e-5)

def train(model, num_epochs, optimizer, train_dataloader, device):
    train_losses = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        print('Epoch: ', epoch + 1)
        for imgs, caps in tqdm(train_dataloader):
            imgs = imgs.to(device)
            caps_tokenized = tokenizer(caps, return_tensors="pt", padding='max_length', truncation=True, max_length=140).input_ids
            caps_tokenized = caps_tokenized.to(device)

            optimizer.zero_grad()
            loss = model(imgs, captions=caps_tokenized)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss}")

train(
    model=model,
    num_epochs=5,
    optimizer=optimizer,
    train_dataloader=train_dataloader,
    device=device
)

model_filepath = f'/data/cndt_thangcpd/HuyDB/PhoGPT_5e.pth'
torch.save(model.state_dict(), model_filepath)
print('Model saved to ', model_filepath)

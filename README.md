# 🖼️ Image Captioning with LLM/Transformer and CLIP
This project focuses on generating **descriptive captions for images** by combining **vision-language models** and **language models**. The core idea is to use **CLIP** to extract semantic visual features from images, and then pass those features to **language models** like **PhoGPT, mT5,** or **mBART** to generate natural language captions — especially tailored for Vietnamese.

## 🎯 Objective
To build an **image captioning system** capable of generating fluent and contextually relevant descriptions in **Vietnamese**, leveraging both powerful **vision encoders** and **large language models (LLMs).**

## 🧠 Model Components
- 🖼️ **CLIP (Contrastive Language-Image Pretraining):**
  - Used to encode images into rich semantic feature vectors.
  - Acts as the vision encoder.
-  **Language Models:**
  - PhoGPT (Vietnamese LLM): For generating Vietnamese captions using prompt-based inference.
  - mT5 / mBART (multilingual transformers): Used in fine-tuning settings to generate captions from encoded features

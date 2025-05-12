# Image Captioning with CLIP and PhoGPT

## Train model

1. Tải KTVIC dataset
2. Cài đặt Java (nếu muốn evaluate CIDEr)
3. Cài đặt pytorch 2.5.1 theo hướng dẫn trên website
4. Cài đặt các thư viện cần thiết

```bash
pip install -r requirements.txt
```

5. Search `EDIT` trong file `clip_phogpt_image_caption.py` và chỉnh sửa cho phù hợp
6. Chạy file

```bash
python clip_phogpt_image_caption.py
```

## Evaluate model

1. Tải KTVIC dataset
2. Cài đặt Java (nếu muốn evaluate CIDEr)
3. Cài đặt pytorch 2.5.1 theo hướng dẫn trên website
4. Cài đặt các thư viện cần thiết

```bash
pip install -r requirements.txt
```

5. Search `EDIT` trong file `clip_phogpt_image_caption_eval.py` và chỉnh sửa cho phù hợp
6. Chạy file

```bash
python clip_phogpt_image_caption_eval.py
```

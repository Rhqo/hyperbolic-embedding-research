# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import torch
from tqdm import tqdm

# MERU-specific imports
from meru.config import LazyConfig, LazyFactory
from meru.models import MERU
from meru.tokenizer import Tokenizer
from meru.utils.checkpointing import CheckpointManager

# --- Configuration ---
# 중요: 아래 경로들을 실제 파일 위치에 맞게 수정하세요.
TRAIN_CONFIG_PATH = './configs/train_meru_vit_b.py'  # 학습 설정 파일 (.py) 경로
CHECKPOINT_PATH = './checkpoints/meru_vit_b.pth'    # 모델 체크포인트 파일 (.pth) 경로
PEXELS_TEXT_PATH = './assets/pexels_text.json'      # 텍스트 데이터 파일 (.json) 경로
OUTPUT_JSON_PATH = './embeddings/text_embeddings.json' # 최종 JSON 파일을 저장할 경로

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _C_TRAIN = LazyConfig.load(TRAIN_CONFIG_PATH)
    model = LazyFactory.build_model(_C_TRAIN, device).eval()
    CheckpointManager(model=model).load(CHECKPOINT_PATH)

    try:
        with open(PEXELS_TEXT_PATH, 'r') as f:
            pexels_text = json.load(f)
        print(f"Successfully loaded text data from '{PEXELS_TEXT_PATH}'.")
    except FileNotFoundError:
        print(f"Error: Text data file not found at '{PEXELS_TEXT_PATH}'")
        return

    text_embeddings = {}
    tokenizer = Tokenizer()

    with torch.no_grad():
        # 1. Captions
        captions = pexels_text.get("captions", [])
        if captions:
            for text in tqdm(captions, desc="Generating Caption Embeddings"):
                tokens = tokenizer([text])
                embedding = model.encode_text(tokens, project=True)
                text_embeddings[text] = embedding.cpu().numpy().tolist()[0]
        
        # 2. Nouns
        nouns = pexels_text.get("nouns", [])
        NOUN_PROMPT = "a photo of a {}."
        if nouns:
            for text in tqdm(nouns, desc="Generating Noun Embeddings   "):
                prompted_text = NOUN_PROMPT.format(text)
                tokens = tokenizer([prompted_text])
                embedding = model.encode_text(tokens, project=True)
                # 원본 명사를 key로 사용합니다.
                text_embeddings[text] = embedding.cpu().numpy().tolist()[0]

        # 3. Adjectives
        adjectives = pexels_text.get("adjectives", [])
        ADJ_PROMPT = "this photo is {}."
        if adjectives:
            for text in tqdm(adjectives, desc="Generating Adjective Embeddings"):
                prompted_text = ADJ_PROMPT.format(text)
                tokens = tokenizer([prompted_text])
                embedding = model.encode_text(tokens, project=True)
                # 원본 형용사를 key로 사용합니다.
                text_embeddings[text] = embedding.cpu().numpy().tolist()[0]

        # 4. [ROOT] 임베딩 추가
        # MERU 모델의 경우 [ROOT]는 쌍곡 공간의 원점(0벡터)입니다.
        if isinstance(model, MERU):
            embed_dim = _C_TRAIN.model.embed_dim
            root_feat = torch.zeros(embed_dim, device=device)
        else:
            # CLIP의 경우 체크포인트에 'root' 임베딩이 저장되어 있을 수 있습니다.
            root_feat = torch.load(CHECKPOINT_PATH).get("root", torch.zeros(_C_TRAIN.model.embed_dim, device=device))
        
        text_embeddings["[ROOT]"] = root_feat.cpu().numpy().tolist()
        print("\n[ROOT] embedding generated.")

    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)

    # 임베딩 딕셔너리를 JSON 파일로 저장합니다.
    with open(OUTPUT_JSON_PATH, 'w') as json_file:
        json.dump(text_embeddings, json_file, indent=4)

    print(f"\nProcessing complete. Embeddings saved to '{OUTPUT_JSON_PATH}'")

if __name__ == "__main__":
    main()
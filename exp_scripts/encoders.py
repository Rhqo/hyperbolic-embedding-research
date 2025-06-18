# 'encoder.py' 파일로 저장하여 사용하세요.

import torch
from PIL import Image
from torchvision import transforms as T
import numpy as np

# MERU-specific imports
from meru.config import LazyConfig, LazyFactory
from meru.models import MERU, CLIPBaseline
from meru.tokenizer import Tokenizer
from meru.utils.checkpointing import CheckpointManager

class Encoder:
    def __init__(self, train_config_path: str, checkpoint_path: str):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 모델 로딩
        _C_TRAIN = LazyConfig.load(train_config_path)
        self.model = LazyFactory.build_model(_C_TRAIN, self.device).eval()
        CheckpointManager(model=self.model).load(checkpoint_path)
        
        # 토크나이저 초기화
        self.tokenizer = Tokenizer()
        
        # 이미지 전처리기 정의
        self.preprocess = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor()
        ])

    def image_encoder(self, image_path: str) -> np.ndarray:
        try:
            with torch.no_grad():
                # 이미지 열기 및 전처리
                image = Image.open(image_path).convert("RGB")
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                
                # 모델을 사용하여 이미지 임베딩 생성
                embedding = self.model.encode_image(image_input, project=True)
                
                # L2 정규화
                embedding /= embedding.norm(dim=-1, keepdim=True)
                
                # CPU로 이동 후 numpy 배열로 변환하여 반환
                return embedding.cpu().numpy()

        except Exception as e:
            print(f"Could not process image {image_path}. Error: {e}")
            return None

    def text_encoder(self, text: str) -> np.ndarray:
        try:
            with torch.no_grad():
                # 텍스트 토큰화
                text_token = self.tokenizer([text])
                
                # 모델을 사용하여 텍스트 특성 생성
                text_features = self.model.encode_text(text_token, project=True)
                
                # CPU로 이동 후 numpy 배열로 변환하여 반환
                return text_features.cpu().numpy()

        except Exception as e:
            print(f"Could not process text. Error: {e}")
            return None

"""
Perform multimodal traversals between an image and a specific text embedding,
finding related images and texts.
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from PIL import Image
from torchvision import transforms as T

from meru import lorentz as L
from meru.config import LazyConfig, LazyFactory
from meru.models import MERU, CLIPBaseline
from meru.tokenizer import Tokenizer
from meru.utils.checkpointing import CheckpointManager


parser = argparse.ArgumentParser(description=__doc__)
_AA = parser.add_argument
_AA("--checkpoint-path", default='./checkpoints/meru_vit_b.pth', help="Path to checkpoint of a trained MERU/CLIP model.")
_AA("--train-config", default='./configs/train_meru_vit_b.py', help="Path to train config (.yaml/py) for given checkpoint.")
_AA("--image-path", default='./datasets/fashion200k/casual_and_day_dresses/51727804/51727804_0.jpeg', help="Path to an image (.jpg) for perfoming traversal.")
_AA("--target-text", default='a photo of a black dress', help="Target text for multimodal traversal.")
_AA("--steps", type=int, default=50, help="Number of traversal steps.")
_AA("--image-pool-path", default='./datasets/fashion200k/', help="Path to a directory of images for traversal.")
_AA("--max-pool-images", type=int, default=1000, help="Maximum number of images to use from the image pool.")


def interpolate(model, feats: torch.Tensor, target_feat: torch.Tensor, steps: int):
    """
    Interpolate between given feature vector and target feature depending on model type.
    """

    # Linear interpolation between root and image features. For MERU, this happens
    # in the tangent space of the origin.
    if isinstance(model, MERU):
        feats = L.log_map0(feats, model.curv.exp())
        target_feat = L.log_map0(target_feat, model.curv.exp())

    interp_feats = [
        torch.lerp(feats, target_feat, weight.item())
        for weight in torch.linspace(0.0, 1.0, steps=steps)
    ]
    interp_feats = torch.stack(interp_feats)

    # Lift on the Hyperboloid (for MERU), or L2 normalize (for CLIP).
    if isinstance(model, MERU):
        interp_feats = L.exp_map0(interp_feats, model.curv.exp())
    else:
        interp_feats = torch.nn.functional.normalize(interp_feats, dim=-1)

    return interp_feats


def calc_scores(
    model, image_feats: torch.Tensor, text_feats: torch.Tensor, has_root: bool
):
    """
    Calculate similarity scores between the given image and text features depending
    on model type.

    Args:
        has_root: Flag to indicate whether the last text embedding (at dim=0)
            is the `[ROOT]` embedding.
    """

    if isinstance(model, MERU):
        scores = L.pairwise_inner(image_feats, text_feats, model.curv.exp())

        # For MERU, exclude text embeddings that do not entail the given image.
        _aper = L.half_aperture(text_feats, model.curv.exp())
        _oxy_angle = L.oxy_angle(
            text_feats[:, None, :], image_feats[None, :, :], model.curv.exp()
        )
        entailment_energy = _oxy_angle - _aper[..., None]

        # Root entails everything.
        if has_root:
            entailment_energy[-1, ...] = 0

        # Set a large negative score if text does not entail image.
        scores[entailment_energy.T > 0] = -1e12
        return scores
    else:
        # model is not needed here.
        return image_feats @ text_feats.T


@torch.inference_mode()
def get_image_feats(
    model: MERU | CLIPBaseline,
    device: torch.device,
    image_folder: str,
    max_images: int,
) -> tuple[list[str], torch.Tensor]:
    """
    Get features for all images in a folder.
    """
    all_image_paths = []
    for root, _, files in os.walk(image_folder):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                all_image_paths.append(os.path.join(root, file))

    if len(all_image_paths) > max_images:
        all_image_paths.sort()
        all_image_paths = all_image_paths[:max_images]

    image_transform = T.Compose(
        [T.Resize(224, T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor()]
    )

    all_image_feats = []
    processed_paths = []
    batch_size = 32
    for i in range(0, len(all_image_paths), batch_size):
        batch_paths = all_image_paths[i : i + batch_size]
        batch_images = []
        valid_paths_in_batch = []
        for image_path in batch_paths:
            try:
                image = Image.open(image_path).convert("RGB")
                batch_images.append(image_transform(image))
                valid_paths_in_batch.append(image_path)
            except Exception as e:
                print(f"Warning: Skipping image {image_path} due to error: {e}")

        if not batch_images:
            continue

        images_tensor = torch.stack(batch_images).to(device)
        image_feats = model.encode_image(images_tensor, project=True)
        all_image_feats.append(image_feats.cpu())
        processed_paths.extend(valid_paths_in_batch)

    if not all_image_feats:
        return [], torch.empty(0)

    all_image_feats = torch.cat(all_image_feats, dim=0).to(device)
    return processed_paths, all_image_feats


@torch.inference_mode()
def get_text_feats(model: MERU | CLIPBaseline) -> tuple[list[str], torch.Tensor]:
    # Get all captions, nouns, and ajectives collected from pexels.com website
    pexels_text = json.load(open("assets/pexels_text.json"))

    # Use very simple prompts for noun and adjective tags.
    tokenizer = Tokenizer()
    NOUN_PROMPT = "a photo of a {}."
    ADJ_PROMPT = "this photo is {}."

    all_text_feats = []

    # Tokenize and encode captions.
    caption_tokens = tokenizer(pexels_text["captions"])
    all_text_feats.append(model.encode_text(caption_tokens, project=True))

    # Tokenize and encode prompts filled with tags.
    # Extract features of all captions and tags.
    noun_prompt_tokens = tokenizer(
        [NOUN_PROMPT.format(tag) for tag in pexels_text["nouns"]]
    )
    all_text_feats.append(model.encode_text(noun_prompt_tokens, project=True))

    adj_prompt_tokens = tokenizer(
        [ADJ_PROMPT.format(tag) for tag in pexels_text["adjectives"]]
    )
    all_text_feats.append(model.encode_text(adj_prompt_tokens, project=True))

    all_text_feats = torch.cat(all_text_feats, dim=0)
    all_pexels_text = [
        *pexels_text["captions"],
        *pexels_text["nouns"],
        *pexels_text["adjectives"],
    ]
    return all_pexels_text, all_text_feats


@torch.inference_mode()
def main(_A: argparse.Namespace):
    # Get the current device (this will be `cuda:0` here by default) or use CPU.
    device = (
        torch.cuda.current_device()
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # Create the model using training config and load pre-trained weights.
    _C_TRAIN = LazyConfig.load(_A.train_config)
    model = LazyFactory.build_model(_C_TRAIN, device).eval()

    CheckpointManager(model=model).load(_A.checkpoint_path)

    # If no external text features are provided, use captions/tags from pexels.
    text_pool, text_feats_pool = get_text_feats(model)

    # Get image features from the specified folder.
    image_pool, image_feats_pool = get_image_feats(
        model, device, _A.image_pool_path, _A.max_pool_images
    )
    if not image_pool:
        print(f"No images found in {_A.image_pool_path}. Skipping image traversal.")

    # Encode the target text
    tokenizer = Tokenizer()
    target_text_tokens = tokenizer([_A.target_text])
    target_text_feat = model.encode_text(target_text_tokens, project=True)[0]

    # ------------------------------------------------------------------------
    print(f"\nPerforming multimodal traversals from image: {_A.image_path} to text: '{_A.target_text}'...")
    # ------------------------------------------------------------------------
    image = Image.open(_A.image_path).convert("RGB")

    image_transform = T.Compose(
        [T.Resize(224, T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor()]
    )
    image = image_transform(image).to(device)
    image_feats = model.encode_image(image[None, ...], project=True)[0]

    interp_feats = interpolate(model, image_feats, target_text_feat, _A.steps)
    nn1_scores = calc_scores(model, interp_feats, text_feats_pool, has_root=False) # target text is not root

    nn1_scores, _nn1_idxs = nn1_scores.max(dim=-1)
    nn1_texts = [text_pool[_idx.item()] for _idx in _nn1_idxs]

    # De-duplicate retrieved texts (multiple points may have same NN) and print.
    print(f"Texts retrieved from [IMAGE] -> [TARGET TEXT] traversal:")
    unique_nn1_texts = []
    for _text in nn1_texts:
        if _text not in unique_nn1_texts:
            unique_nn1_texts.append(_text)
            print(f"  - {_text}")

    # Now find nearest images along the traversal.
    if image_pool:
        nn2_scores = calc_scores(model, interp_feats, image_feats_pool, has_root=False)
        nn2_scores, _nn2_idxs = nn2_scores.max(dim=-1)
        nn2_images = [image_pool[_idx.item()] for _idx in _nn2_idxs]

        # De-duplicate retrieved images and print.
        print(f"\nImages retrieved from [IMAGE] -> [TARGET TEXT] traversal:")
        unique_nn2_images = []
        for _image in nn2_images:
            if _image not in unique_nn2_images:
                unique_nn2_images.append(_image)
                print(f"  - {_image}")


if __name__ == "__main__":
    _A = parser.parse_args()
    main(_A)
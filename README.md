# Hyperbolic Embedding Research

This repository focuses on hyperbolic embeddings for vision-language models, featuring MERU, a framework for training and evaluating models like CLIP using Vision Transformers (ViT). It includes tools for training, evaluation (zero-shot classification, retrieval), and experimental scripts for analyzing embedding spaces.

`exp_scripts/`: Contains Python scripts for running various experiments.
`src/`: Core implementation of MERU and related utilities.

To run 'custom' experiment scripts, use the following command:

```bash
uv run exp_scripts/*.py
```

This project, "hyperbolic-embedding-research," is dedicated to exploring and visualizing hyperbolic embeddings, primarily for multimodal (image and text) data. It utilizes the MERU model for encoding and analysis.

The `exp_scripts` directory contains a suite of experimental scripts for:
*   **Embedding Comparison:** Visualizing dimension-wise comparisons of image and text embeddings (`compare_img.py`, `compare_img_text.py`).
*   **Embedding Encoding:** Providing a utility class to encode images and text using pre-trained MERU models (`encoders.py`).
*   **Multimodal Traversal:** Exploring semantic paths between images and target text embeddings (`multimodal_traversal.py`).
*   **Dimensionality Reduction & Visualization:** Reducing and visualizing embeddings (image, text, root) in 2D and 3D using PCA (`pca_2d.py`, `pca_3d.py`) and UMAP (`umap_2d.py`, `umap_3d.py`).
*   **Traversal Visualization:** Specifically visualizing the semantic interpolation paths between image and text embeddings in 2D and 3D using both PCA and UMAP (`visualize_traversal_pca.py`, `visualize_traversal_pca_3d.py`, `visualize_traversal_umap.py`, `visualize_traversal_umap_3d.py`).
*   **Embedding Analysis:** Generating heatmaps and histograms of embeddings to analyze their distribution and properties (`spectrum_all.py`, `spectrum_statistics.py`).

In essence, `exp_scripts` serves as a toolkit for in-depth analysis, visualization, and understanding of hyperbolic embeddings and their relationships within multimodal data.
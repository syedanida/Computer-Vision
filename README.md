# Computer-Vision

This repository contains implementations of various deep learning techniques focusing on supervised contrastive learning, transfer learning across different modalities, zero-shot learning, and vision classifiers.

## Table of Contents
- [Part 1: Supervised Contrastive Learning](https://github.com/syedanida/Computer-Vision/blob/main/Part_1_Supervised_Contrastive_Learning.ipynb)
- [Part 2: Transfer Learning on Various Modalities](https://github.com/syedanida/Computer-Vision/blob/main/Part_2_Transfer_Learning_on_Various_Modalities.ipynb)
- [Part 3: Zero-Shot Transfer Learning with CLIP](https://github.com/syedanida/Computer-Vision/blob/main/Part_3_Zero_Shot_Transfer_Learning_with_CLIP.ipynb)
- [Part 4: Vision Classifiers](https://github.com/syedanida/Computer-Vision/blob/main/Part_4_Medical_Imaging.ipynb)
- [Video Walkthrough](#video-walkthrough)

## Part 1: Supervised Contrastive Learning

In this section, we implemented a complete comparison between traditional classification and supervised contrastive learning approaches:

- Traditional softmax-based classification with cross-entropy loss
- Two-phase supervised contrastive learning with contrastive pre-training
- Visualizations showing the difference in feature embeddings
- Performance and training time comparisons

Our implementation demonstrates how supervised contrastive learning can create more discriminative feature spaces compared to traditional approaches.

**Colab Link:** [Supervised Contrastive Learning](https://github.com/syedanida/Computer-Vision/blob/main/Part_1_Supervised_Contrastive_Learning.ipynb)

## Part 2: Transfer Learning on Various Modalities

We implemented transfer learning across four different data types, demonstrating both feature extraction and fine-tuning approaches for each modality.

### 2.1 Image Transfer Learning

- Feature extraction and fine-tuning with pre-trained models
- Dogs vs Cats classification
- Performance comparisons and visualizations
- Analysis of feature representations

### 2.2 Audio Transfer Learning

- YAMNet-based transfer learning
- Feature extraction and simplified fine-tuning
- Audio classification with visualization
- Analysis of audio feature representations

### 2.3 Video Transfer Learning

- I3D-based transfer learning for action recognition
- Feature extraction and fine-tuning approaches
- Animation and visualization of results
- Temporal feature analysis

### 2.4 NLP Transfer Learning

- Universal Sentence Encoder for feature extraction
- BERT for both feature extraction and fine-tuning
- Sentiment analysis with performance comparisons
- Analysis of contextual embeddings

**Colab Link:** [Image Transfer Learning](https://github.com/syedanida/Computer-Vision/blob/main/Part_2_Transfer_Learning_on_Various_Modalities.ipynb)

## Part 3: Zero-Shot Transfer Learning with CLIP

We implemented zero-shot classification capabilities using OpenAI's CLIP model:

- Simulated CLIP functionality for zero-shot classification
- Comparison of different prompt templates
- Testing with out-of-distribution classes
- Visualization of zero-shot predictions

Our implementation demonstrates how CLIP can classify images into categories it has never explicitly been trained on.

**Colab Link:** [Zero-Shot Transfer Learning with CLIP](https://github.com/syedanida/Computer-Vision/blob/main/Part_3_Zero_Shot_Transfer_Learning_with_CLIP.ipynb)

## Part 4: Vision Classifiers

We implemented various vision classifiers across multiple datasets, comparing traditional approaches with state-of-the-art models.

### 4.1 Standard Vision Datasets

- MNIST classification with multiple models
- Fashion MNIST classification with multiple models
- CIFAR-10 classification with multiple models
- Comparisons between EfficientNet, BiT, MLP-Mixer, and ConvNeXt

**Colabs:**
- [MNIST Classifiers]
- [Fashion MNIST Classifiers]
- [CIFAR-10 Classifiers]

  **Colab Link:** [Standard Vision Datasets](https://github.com/syedanida/Computer-Vision/blob/main/Part_4_Medical_Imaging.ipynb)

### 4.2 Medical Imaging

- X-ray pneumonia classification using ConvNets and transfer learning
- 3D CT scan classification with 3D CNNs
- Visualization of medical imaging results with simulated activation maps

**Colabs:**
- [X-ray Pneumonia Classification]
- [3D CT Scan Classification]

  **Colab Link:** [Medical Imaging](https://github.com/syedanida/Computer-Vision/blob/main/Part_4_Standard_Vision.ipynb)

## Requirements

- TensorFlow 2.x
- TensorFlow Hub
- Matplotlib
- NumPy
- Pandas
- Scikit-learn
- Additional libraries as specified in individual notebooks

## Video Walkthrough

A complete video walkthrough of all implementations is available here:

[Video Walkthrough Link](https://youtu.be/mh9meGrpbgQ)

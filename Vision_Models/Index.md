# Vision Models — Interview Preparation Notes

> Built for Amazon interview depth. Tier 1 topics (CNN, Transfer Learning, CLIP, Contrastive Learning, Embeddings, VLMs) are covered fully. Tier 2 topics (Detection, Segmentation, VAE, GAN, ViT, Normalization) are covered at the concept + interview-answer level.
> Same format as Sequence Models: **Problem → Concept → Diagram → Example → Trade-offs → Interview Callouts → Summary.**

---

## Part 1 — CNN Foundations

| Lesson | Topic |
|---|---|
| [Lesson 1.1](1_CNN_Foundations/Lesson_1_1.md) | How CNNs See: Convolution, Filters, Feature Maps |
| [Lesson 1.2](1_CNN_Foundations/Lesson_1_2.md) | Pooling, Receptive Field, and CNN Depth |
| [Lesson 1.3](1_CNN_Foundations/Lesson_1_3.md) | CNN Architectures: From AlexNet to ResNet |

## Part 2 — Transfer Learning

| Lesson | Topic |
|---|---|
| [Lesson 2.1](2_Transfer_Learning/Lesson_2_1.md) | Transfer Learning: Why It Works and When to Use It |
| [Lesson 2.2](2_Transfer_Learning/Lesson_2_2.md) | Fine-Tuning Strategies: Feature Extraction vs Full Fine-Tuning |

## Part 3 — Contrastive Learning & CLIP

| Lesson | Topic |
|---|---|
| [Lesson 3.1](3_CLIP_Contrastive/Lesson_3_1.md) | Contrastive Learning: Learning Representations by Comparison |
| [Lesson 3.2](3_CLIP_Contrastive/Lesson_3_2.md) | CLIP: Aligning Images and Text at Scale |

## Part 4 — Image Embeddings & Similarity Search

| Lesson | Topic |
|---|---|
| [Lesson 4.1](4_Embeddings_Search/Lesson_4_1.md) | Image Embeddings and the Vector Space Concept |
| [Lesson 4.2](4_Embeddings_Search/Lesson_4_2.md) | Similarity Search: ANN, FAISS, and Visual Search Systems |

## Part 5 — Vision-Language Models

| Lesson | Topic |
|---|---|
| [Lesson 5.1](5_VLMs/Lesson_5_1.md) | Vision-Language Models: Connecting Vision and Language |

## Part 6 — Detection & Segmentation

| Lesson | Topic |
|---|---|
| [Lesson 6.1](6_Detection_Segmentation/Lesson_6_1.md) | Object Detection: YOLO vs R-CNN |
| [Lesson 6.2](6_Detection_Segmentation/Lesson_6_2.md) | Segmentation: Semantic vs Instance vs Panoptic |

## Part 7 — Generative Models

| Lesson | Topic |
|---|---|
| [Lesson 7.1](7_Generative_Models/Lesson_7_1.md) | VAE: Learning Structured Latent Spaces |
| [Lesson 7.2](7_Generative_Models/Lesson_7_2.md) | GAN: Adversarial Training and Its Problems |

## Part 8 — Vision Transformers & Normalization

| Lesson | Topic |
|---|---|
| [Lesson 8.1](8_ViT_Normalization/Lesson_8_1.md) | ViT: Images as Token Sequences |
| [Lesson 8.2](8_ViT_Normalization/Lesson_8_2.md) | Batch Normalization and Layer Normalization |

---

*Amazon context: Amazon Go uses object detection + segmentation. Amazon Search uses image embeddings + CLIP. Amazon Nova is a VLM. Alexa multimodal uses VLM-style fusion. These notes are built with those systems in mind.*

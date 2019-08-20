# YelpMeKnow
**YelpMeKnow** is a text classifier model, which leverages the power of [**Google's BERT**](https://arxiv.org/pdf/1810.04805.pdf) pretrained models through the [**Hugging Face**](https://github.com/huggingface/pytorch-transformers) Pytorch implementation. 
Specifically the **BERT-Base-Uncased**: 12-layer, 768-hidden, 12-heads, 110M parameters, is used.

The model performs a Sequence Classification analysis of the customer satisfaction, (positive vs. negative reviews), contained in [**Yelp Review Polarity Dataset**](https://course.fast.ai/datasets).
The dataset containes 560,000 training samples and 38,000 testing samples, but due to limited resources and time I'm training/validating and testing on a really small subset of it.

### Project data
Train data size: 20000 ~ 3.6% of training samples

Test data size: 2000 ~ 5.3% of testing samples

Epochs: 1

Matthew's correlation coefficient: ~ 0.86

The accuracy of predictions is evaluated using [**Matthew’s correlation coefficient**](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html), which is in essence a correlation coefficient value between -1 and +1. A coefficient of +1 represents a perfect prediction, 0 an average random prediction and -1 an inverse prediction.

**N.B.**
[WORK IN PROGRESS] Data preparation and model require improvements and further training.

This project is part of [**SPAIC Project Showcase Challenge**](https://sites.google.com/udacity.com/secureprivateai-challenge/community/project-showcase-challenge)

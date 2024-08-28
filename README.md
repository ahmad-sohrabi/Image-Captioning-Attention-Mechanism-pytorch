# Image-Captioning-PyTorch
This repository contains codes to preprocess, train and evaluate sequence models on Flickr8k Image dataset in pytorch.

**Models Experimented with**:
- Pretrained Resnet-18 & LSTM with Attention Mechanism


**Pre-requisites**:
 - Datasets:
    - Flickr8k Dataset: [images](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip) and [annotations](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip)
 - Pre-trained word embeddings:
    - [Glove Embeddings of 6B words](http://nlp.stanford.edu/data/glove.6B.zip)

**Data Folder Structure for training using [`train_torch.py`](train_torch.py) or [`train_attntn.py`](train_attntn.py):**
```
data/
    flickr8k/
        Flicker8k_Dataset/
            *.jpg
        Flickr8k_text/
            Flickr8k.token.txt
            Flickr_8k.devImages.txt
            Flickr_8k.testImages.txt
            Flickr_8k.trainImages.txt
    glove.6B/
        glove.6B.50d.txt
        glove.6B.100d.txt
        glove.6B.200d.txt
        glove.6B.300d.txt
```


**Bleu 4 score of trained model with Attention Mechanism is about 0.147**

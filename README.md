# Embed Images in 🤗 Dataset
Generates image embeddings for a 🤗 Dataset then uploads it to the 🤗 Hub

## Installation
```bash
pip install -r requirements.txt
```

## Usage

Import module
```python
from Embed import create_dataset_embeddings
```

Generates image embeddings and upload to 🤗

```python
create_dataset_embeddings(input_dataset='tonyassi/fashion-decade-images-1',
                          output_dataset='tonyassi/fashion-decade-images-1-embeddings',
                          token='YOUR_TOKEN')

```

```python
create_dataset_embeddings(input_dataset='tonyassi/fashion-decade-images-1',
                          output_dataset='tonyassi/fashion-decade-images-1-embeddings',
                          token='YOUR_TOKEN',
                          model_ckpt='tonyassi/fashion-clothing-decade')

```

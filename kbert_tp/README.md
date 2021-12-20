## Requirements

Software:
```
Python3
Pytorch >= 1.0
argparse == 1.1
```


## Prepare

* Download the ``pytorch_model.bin`` from https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract/tree/main, and rename it to 
 'biomed_BLP_pubmed_model.bin' and save it to the ``models/`` directory.
* Download the ``.spo`` files from (https://drive.google.com/drive/folders/1og1PT14opUv6W89_kqBETnCZq9KF_P2t?usp=sharing), and save it to the ``brain/kgs/`` directory.

The directory tree of K-BERT:
```
K-BERT
├── brain
│   ├── config.py
│   ├── __init__.py
│   ├── kgs
│   │   ├── *.spo 
│   └── knowgraph.py
├── datasets
├── models
│   ├── biomed_BLP_pubmed_config.json
│   ├── biomed_BLP_pubmed_model.bin
│   └── biomed_BLP_pubmed_vocab.txt
│   └── tokenizer_config.json
├── outputs
├── uer
├── README.md
├── requirements.txt
└── run_kbert_ner.py
```

## K-BERT for named entity recognization (NER)

### NER example

Run an example on the msra_ner dataset with CnDbpedia:

```
CUDA_VISIBLE_DEVICES='0' python3 -u run_kbert_ner.py \
    --pretrained_model_path ./models/biomed_BLP_pubmed_model.bin \
    --config_path ./models/biomed_BLP_pubmed_config.json \
    --vocab_path ./models/biomed_BLP_pubmed_vocab.txt \
    --train_path train \
    --dev_path validation \
    --test_path test \
    --epochs_num 5 \
    --batch_size 32 \
    --kg_name Medical \
    --output_model_path ./outputs/kbert_med.bin \
```

useage: [--pretrained_model_path] - Path to the pre-trained model parameters.
        [--config_path] - Path to the model configuration file.
        [--vocab_path] - Path to the vocabulary file.
        --train_path - Path to the training dataset.
        --dev_path - Path to the validating dataset.
        --test_path - Path to the testing dataset.
        [--epochs_num] - The number of training epoches.
        [--batch_size] - Batch size of the training process.
        [--kg_name] - The name of knowledge graph.
        [--output_model_path] - Path to the output model.
```


## Acknowledgement
```
@inproceedings{weijie2019kbert,
  title={{K-BERT}: Enabling Language Representation with Knowledge Graph},
  author={Weijie Liu, Peng Zhou, Zhe Zhao, Zhiruo Wang, Qi Ju, Haotang Deng, Ping Wang},
  booktitle={Proceedings of AAAI 2020},
  year={2020}
}
```

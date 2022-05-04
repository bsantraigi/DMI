---
layout: post
title: Introducing DMI
date:   2022-05-01 11:02:34 +0530
categories: nlp dialog
---

![DMI_model](DMI/assets/dmi_model.png)

This repository hosts the PyTorch-based implementation for the DMI model proposed in our NAACL 2022 paper - [**Representation Learning for Conversational Data using Discourse Mutual Information Maximization**](https://arxiv.org/abs/2112.05787).

<!-- ![DMI_Cover](DMI/assets/DMI-coverimage.png) -->

## Getting Access to the Source Code or Pretrained Models

To get access to the source-code or pretrained-model checkpoints, please send a request to [AcadGrants@service.microsoft.com](mailto:AcadGrants@service.microsoft.com) and cc to *pawang [_at_] iitkgp.ac.in* and *bsantraigi [_at_] gmail.com*.

### Note

The requesting third party
1. **Can download and use these deliverables for research as well as commercial use,**
2. **Modify it as they like but should include citation to our work and include this readme**, and
3. **Cannot redistribute strictly to any other organization.**

**Cite As**

```bibtex
@inproceedings{santra2022representation,
  title={Representation Learning for Conversational Data using Discourse Mutual Information Maximization},
  author={Santra, Bishal and Roychowdhury, Sumegh and Mandal, Aishik and Gurram, Vasu and Naik, Atharva and Gupta, Manish and Goyal, Pawan},
  booktitle={Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  year={2022}
}
```

## Requirements

- wandb
- transformers
- datasets
- torch 1.8.2 (lts)

## How to run?

### Loading and Finetuning the model for a task

For finetuning the model on the tasks mentioned in the paper, or on a new task, use the `run_finetune.py` script or modify it according to your requirements. Example commands for launching finetuning based on some DMI checkpoints can be found in the `auto_eval` directory.

### Pretraining Dataset

There are two types of dataset structure that are available for model pretraining.

In case of smaller or **"Normal"** datasets, a single train_dialog file contains all the training data and is consumed fully during each epoch.

In case of **"Large"** datasets, the files are split into smaller shards and saved as .json files.

1. **Normal Datasets**: For example of this, check the `data/dailydialog` or `data/reddit_1M` directories.
```sh
data/reddit_1M
├── test_dialogues.txt
├── train_dialogues.txt
└── val_dialogues.txt
```
2. **Large Datasets**: This mode can be activated by setting the `--dataset` argument to `rMax`, i.e., `--dataset rMax` or `-dd rMax`. This also require you to provide the `-rmp` argument for the directory path containing the json files. For validation during pretraining, this model uses the DailyDialog validation set by default.
```sh
data/rMax-subset
├── test-00000-of-01000.json
├── test-00001-of-01000.json
├── test-00002-of-01000.json
├── test-00003-of-01000.json
├── ...
├── train-00000-of-01000.json
├── train-00001-of-01000.json
├── train-00002-of-01000.json
├── train-00003-of-01000.json
└── ...
```

### For training a model

To train a new model, it can be started using the pretrain.py script.

**Example:**

1. For training from scratch:
```bash
python pretrain.py \
  -dd rMax -voc roberta \
  --roberta_init \
  -sym \
  -bs 64 -ep 1000 -vi 400 -li 50 -lr 5e-5 -scdl \
  --data_path ./data \
  -rmp /disk2/infonce-dialog/data/r727m/ \
  -t 1 \
  -ddp --world_size 6 \
  -ntq
```
2. To resume training from an existing checkpoint: This example shows resuming training from a checkpoint saved under `checkpoints/DMI-Small_BERT-26Jan/`. Also note how we specify a name an existing BERT/RoBERTa model which defines the architecture and the original initialization of the model weights.
```
python pretrain.py \
  -dd rMax -voc bert \
  --roberta_init \
  -robname google/bert_uncased_L-8_H-768_A-12 \
  -sym -bs 130 -lr 1e-5 -scdl -ep 1000 -vi 400 -li 50 \
  --data_path ./data \
  -rmp /disk2/infonce-dialog/data/r727m/ \
  -ddp --world_size 4 \
  -ntq -t 1 \
  -re -rept checkpoints/DMI-Small_BERT-26Jan/model_current.pth
```

**It accepts the following arguments.**

```
  -h, --help            show this help message and exit
  -dd {dd,r5k,r100k,r1M,r1M/cc,rMax,rMax++,paa,WoW}, --dataset {dd,r5k,r100k,r1M,r1M/cc,rMax,rMax++,paa,WoW}
                        which dataset to use for pretraining.
  -rf, --reddit_filter_enabled
                        Enable reddit data filter for removing low quality dialogs.
  -rmp RMAX_PATH, --rmax_path RMAX_PATH
                        path to dir for r727m (.json) data files.
  -dp DATA_PATH, --data_path DATA_PATH
                        path to the root data folder.
  -op OUTPUT_PATH, --output_path OUTPUT_PATH
                        Path to store the output ``model.pth'' files
  -voc {bert,blender,roberta,dgpt-m}, --vocab {bert,blender,roberta,dgpt-m}
                        mention which tokenizer was used for pretraining? bert or blender
  -rob, --roberta_init  Initialize transformer-encoder with roberta weights?
  -robname ROBERTA_NAME, --roberta_name ROBERTA_NAME
                        name of checkpoint from huggingface
  -d D_MODEL, --d_model D_MODEL
                        size of transformer encoders' hidden representation
  -d_ff DIM_FEEDFORWARD, --dim_feedforward DIM_FEEDFORWARD
                        dim_feedforward for transformer encoder.
  -p PROJECTION, --projection PROJECTION
                        size of projection layer output
  -el ENCODER_LAYERS, --encoder_layers ENCODER_LAYERS
                        number of layers in transformer encoder
  -eh ENCODER_HEADS, --encoder_heads ENCODER_HEADS
                        number of heads in tformer enc
  -sym, --symmetric_loss
                        whether to train using symmetric infonce
  -udrl, --unsupervised_discourse_losses
                        Additional unsupervised discourse-relation loss components
  -sdrl, --supervised_discourse_losses
                        Additional supervised discourse-relation loss components
  -es {infonce,jsd,nwj,tuba,dv,smile,infonce/td}, --estimator {infonce,jsd,nwj,tuba,dv,smile,infonce/td}
                        which MI estimator is used as the loss function.
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        batch size during pretraining
  -ep EPOCHS, --epochs EPOCHS
                        epochs for pretraining
  -vi VAL_INTERVAL, --val_interval VAL_INTERVAL
                        validation interval during training
  -li LOG_INTERVAL, --log_interval LOG_INTERVAL
                        logging interval during training
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        set learning rate
  -lrc, --learning_rate_control
                        LRC: outer layer and projection layer will have faster LR and rest will be LR/10
  -t {0,1}, --tracking {0,1}
                        whether to track training+validation loss wandb
  -scdl, --use_scheduler
                        whether to use a warmup+decay schedule for LR
  -ntq, --no_tqdm       disable tqdm to create concise log files!
  -ddp, --distdp        Should it use pytorch Distributed dataparallel?
  -ws WORLD_SIZE, --world_size WORLD_SIZE
                        world size when using DDP with pytorch.
  -re, --resume         2-stage pretrain: Resume training from a previous checkpoint?
  -rept RESUME_MODEL_PATH, --resume_model_path RESUME_MODEL_PATH
                        If ``Resuming'', path to ckpt file.
```

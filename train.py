import ast
import copy
import gc
import itertools
import joblib
import json
import math
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import pandas as pd
import pickle
import random
import re
import scipy as sp
import string
import sys
import time
import warnings
import wandb
from model import CustomModel

from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
# import tokenizers
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import confusion_matrix
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--exp_name', type=str, default='test')

args = parser.parse_args()

# ======= OPTIONS =========
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Current device is: {device}")
warnings.filterwarnings("ignore")
os.makedirs('output', exist_ok=True)

class config:
    APEX = True # Automatic Precision Enabled
    BATCH_SCHEDULER = True
    BATCH_SIZE_TRAIN = 32
    BATCH_SIZE_VALID = 16
    BETAS = (0.9, 0.999)
    DEBUG = False
    DECODER_LR = 2e-5
    ENCODER_LR = 2e-5
    EPOCHS = 5
    EPS = 1e-6
    FOLDS = 4
    GRADIENT_ACCUMULATION_STEPS = 1
    GRADIENT_CHECKPOINTING = True
    MAX_GRAD_NORM=1000
    MAX_LEN = 512
    MIN_LR = 1e-6
    MODEL = "microsoft/deberta-v3-base"
    NUM_CYCLES = 0.5
    NUM_WARMUP_STEPS = 0
    NUM_WORKERS = multiprocessing.cpu_count()
    PRINT_FREQ = 20
    SCHEDULER = 'cosine' # ['linear', 'cosine']
    SEED = 27
    TRAIN = True
    TRAIN_FOLDS = [0, 1, 2, 3]
    WANDB = False
    WEIGHT_DECAY = 0.01

    
class kaggle_paths:
    OUTPUT_DIR = "/kaggle/working/output"
    EXTERNAL_DATA = "/kaggle/input/daigt-external-dataset/daigt_external_dataset.csv"
    TRAIN_PROMPTS = "/kaggle/input/llm-detect-ai-generated-text/train_prompts.csv"
    TRAIN_ESSAYS = "/kaggle/input/llm-detect-ai-generated-text/train_essays.csv"
    TEST_ESSAYS = "/kaggle/input/llm-detect-ai-generated-text/test_essays.csv"
    
class paths:
    OUTPUT_DIR = "output"
    EXTERNAL_DATA = "data/external-dataset/daigt_external_dataset.csv"
    TRAIN_PROMPTS = "data/LLM-DetectAI/train_prompts.csv"
    TRAIN_ESSAYS = "data/LLM-DetectAI/train_essays.csv"
    TEST_ESSAYS = "data/LLM-DetectAI/test_essays.csv"
    PROCESSED_DATA = "train_essays_3.0.csv"

    MODEL_PATH = "output"
    BEST_MODEL_PATH = "output/microsoft_deberta-v3-base_fold_0_best.pth"
    SUBMISSION_CSV = "data/LLM-DetectAI/sample_submission.csv"

if config.DEBUG:
    config.EPOCHS = 2
    config.TRAIN_FOLDS = [0]

def get_config_dict(config):
    """
    Return the config, which is originally a class, as a Python dictionary.
    """
    config_dict = dict((key, value) for key, value in config.__dict__.items() 
    if not callable(value) and not key.startswith('__'))
    return config_dict


def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if "model" not in n],
         'lr': decoder_lr, 'weight_decay': 0.0}
    ]
    return optimizer_parameters


def get_logger(filename=paths.OUTPUT_DIR):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def get_scheduler(cfg, optimizer, num_train_steps):
    if cfg.SCHEDULER == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.NUM_WARMUP_STEPS,
            num_training_steps=num_train_steps
        )
    elif cfg.SCHEDULER == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.NUM_WARMUP_STEPS,
            num_training_steps=num_train_steps, num_cycles=cfg.NUM_CYCLES
        )
    return scheduler
    

def get_score(y_trues, y_preds):
    score = roc_auc_score(y_trues, y_preds)
    return score


def seed_everything(seed=20):
    """Seed everything to ensure reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    

def sep():
    print("-"*100)
    
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))  
    
LOGGER = get_logger()
seed_everything(seed=config.SEED)


notes = ""

if config.WANDB:
    try:
        # from kaggle_secrets import UserSecretsClient
        # user_secrets = UserSecretsClient()
        # secret_value_0 = user_secrets.get_secret("wandb_api")
        # wandb.login(key=secret_value_0)
        anony = None
        wandb.login(key='a3cae2e24b9f52f5007c896738f0fb473b2f3ad5')
    except:
        anony = "must"
        print('If you want to use your W&B account, go to Add-ons -> Secrets and provide your W&B access token. Use the Label name as wandb_api. \nGet your W&B access token from here: https://wandb.ai/authorize')

    run = wandb.init(project='kaggle-daigt', 
                     name=args.exp_name,
                     config=get_config_dict(config),
                    #  group="anti_overfit",
                     job_type="train",
                     notes=notes,
                     anonymous=anony)


def prepare_input(cfg, text, tokenizer):
    """
    This function tokenizes the input text with the configured padding and truncation. Then,
    returns the input dictionary, which contains the following keys: "input_ids",
    "token_type_ids" and "attention_mask". Each value is a torch.tensor.
    :param cfg: configuration class with a TOKENIZER attribute.
    :param text: a numpy array where each value is a text as string.
    :return inputs: python dictionary where values are torch tensors.
    """
    inputs = tokenizer.encode_plus(
        text, 
        return_tensors=None, 
        add_special_tokens=True, 
        max_length=cfg.MAX_LEN,
        padding='max_length', # TODO: check padding to max sequence in batch
        truncation=True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long) # TODO: check dtypes
    return inputs


def collate(inputs):
    """
    It truncates the inputs to the maximum sequence length in the batch. 
    """
    mask_len = int(inputs["attention_mask"].sum(axis=1).max()) # Get batch's max sequence length
    for k, v in inputs.items():
        inputs[k] = inputs[k][:,:mask_len]
    return inputs


class CustomDataset(Dataset):
    def __init__(self, cfg, df, tokenizer):
        self.cfg = cfg
        self.texts = df['text'].values
        self.labels = df['generated'].values
        self.tokenizer = tokenizer
        self.text_ids = df['id'].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        output = {}
        output["inputs"] = prepare_input(self.cfg, self.texts[item], self.tokenizer)
        output["labels"] = torch.tensor(self.labels[item], dtype=torch.float) # TODO: check dtypes
        output["ids"] = self.text_ids[item]
        return output



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def train_epoch(train_loader, model, criterion, optimizer, epoch, scheduler, device):
    """One epoch training pass."""
    model.train() # set model in train mode
    scaler = torch.cuda.amp.GradScaler(enabled=config.APEX) # Automatic Mixed Precision tries to match each op to its appropriate datatype.
    losses = AverageMeter() # initiate AverageMeter to track the loss.
    start = end = time.time() # track the execution time.
    global_step = 0
    
    # ========== ITERATE OVER TRAIN BATCHES ============
    with tqdm(train_loader, unit="train_batch", desc='Train') as tqdm_train_loader:
        for step, batch in enumerate(tqdm_train_loader):
            inputs = batch.pop("inputs")
            labels = batch.pop("labels")
            inputs = collate(inputs) # collate inputs
            for k, v in inputs.items(): # send each tensor value to `device`
                inputs[k] = v.to(device)
            labels = labels.to(device) # send labels to `device`
            batch_size = labels.size(0)
            with torch.cuda.amp.autocast(enabled=config.APEX):
                y_preds = model(inputs) # forward propagation pass
                loss = criterion(y_preds, labels.unsqueeze(1)) # get loss
            if config.GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / config.GRADIENT_ACCUMULATION_STEPS
            losses.update(loss.item(), batch_size) # update loss function tracking
            scaler.scale(loss).backward() # backward propagation pass
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)

            if (step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.step(optimizer) # update optimizer parameters
                scaler.update()
                optimizer.zero_grad() # zero out the gradients
                global_step += 1
                if config.BATCH_SCHEDULER:
                    scheduler.step() # update learning rate
            end = time.time() # get finish time

            # ========== LOG INFO ==========
            if step % config.PRINT_FREQ == 0 or step == (len(train_loader)-1):
                print('Epoch: [{0}][{1}/{2}] '
                      'Elapsed {remain:s} '
                      'Loss: {loss.avg:.4f} '
                      'Grad: {grad_norm:.4f}  '
                      'LR: {lr:.8f}  '
                      .format(epoch+1, step, len(train_loader), 
                              remain=timeSince(start, float(step+1)/len(train_loader)),
                              loss=losses,
                              grad_norm=grad_norm,
                              lr=scheduler.get_lr()[0]))
            if config.WANDB:
                wandb.log({f"[fold_{fold}] train loss": losses.val,
                           f"[fold_{fold}] lr": scheduler.get_lr()[0]})

    return losses.avg


def valid_epoch(valid_loader, model, criterion, device):
    model.eval() # set model in evaluation mode
    losses = AverageMeter() # initiate AverageMeter for tracking the loss.
    prediction_dict = {}
    preds = []
    start = end = time.time() # track the execution time.
    with tqdm(valid_loader, unit="valid_batch", desc='Validation') as tqdm_valid_loader:
        for step, batch in enumerate(tqdm_valid_loader):
            inputs = batch.pop("inputs")
            labels = batch.pop("labels")
            ids = batch.pop("ids")
            inputs = collate(inputs) # collate inputs
            for k, v in inputs.items():
                inputs[k] = v.to(device) # send inputs to device
            labels = labels.to(device)
            batch_size = labels.size(0)
            with torch.no_grad():
                y_preds = model(inputs) # forward propagation pass
                loss = criterion(y_preds, labels.unsqueeze(1)) # get loss
            if config.GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / config.GRADIENT_ACCUMULATION_STEPS
            losses.update(loss.item(), batch_size) # update loss function tracking
            preds.append(y_preds.to('cpu').numpy()) # save predictions
            end = time.time() # get finish time

            # ========== LOG INFO ==========
            if step % config.PRINT_FREQ == 0 or step == (len(valid_loader)-1):
                print('EVAL: [{0}/{1}] '
                      'Elapsed {remain:s} '
                      'Loss: {loss.avg:.4f} '
                      .format(step, len(valid_loader),
                              loss=losses,
                              remain=timeSince(start, float(step+1)/len(valid_loader))))
            if config.WANDB:
                wandb.log({f"[fold_{fold}] val loss": losses.val})

    prediction_dict["predictions"] = np.concatenate(preds) # np.array() of shape (fold_size, target_cols)
    prediction_dict["ids"] = ids
    return losses.avg, prediction_dict

def train_loop(folds, fold):
    
    LOGGER.info(f"========== Fold: {fold} training ==========")

    # ======== SPLIT ==========
    train_folds = folds[folds['fold'] != fold].reset_index(drop=True)
    valid_folds = folds[folds['fold'] == fold].reset_index(drop=True)

    valid_labels = valid_folds['generated'].values

    # ======== DATASETS ==========
    label0_dataset = train_folds[train_folds['generated']==0]
    label1_dataset = train_folds[train_folds['generated']==1]
    dataset = pd.concat([label0_dataset,
                    label1_dataset,
                    label1_dataset,
                    label1_dataset,
                    label1_dataset,
                    label1_dataset,
                    ], 
                    ignore_index=True)
    train_folds = dataset
    train_dataset = CustomDataset(config, train_folds, tokenizer)
    valid_dataset = CustomDataset(config, valid_folds, tokenizer)
    
    # ======== DATALOADERS ==========
    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE_TRAIN, # TODO: split into train and valid
                              shuffle=True,
                              pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=config.BATCH_SIZE_VALID,
                              shuffle=False,
                              pin_memory=True, drop_last=False)
    
    # ======== MODEL ==========
    model = CustomModel(config, config_path=None, pretrained=True)
    torch.save(model.config, paths.OUTPUT_DIR + '/config.pth')
    model.to(device)

    # freeze 6 first layers
    # freeze_layers = [model.model.embeddings, model.model.encoder.layer[:6]]
    # for module in freeze_layers:
    #     for param in module.parameters():
    #         param.requires_grad = False
    #---------------------------------

    optimizer_parameters = get_optimizer_params(model,
                                                encoder_lr=config.ENCODER_LR, 
                                                decoder_lr=config.DECODER_LR,
                                                weight_decay=config.WEIGHT_DECAY)
    optimizer = AdamW(optimizer_parameters,
                      lr=config.ENCODER_LR,
                      eps=config.EPS,
                      betas=config.BETAS)
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-5,
        epochs=config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy="cos",
        final_div_factor=100,
    )

    # ======= LOSS ==========
    criterion = nn.BCEWithLogitsLoss()
    
    best_score = -np.inf
    # ====== ITERATE EPOCHS ========
    for epoch in range(config.EPOCHS):

        start_time = time.time()

        # ======= TRAIN ==========
        avg_loss = train_epoch(train_loader, model, criterion, optimizer, epoch, scheduler, device)

        # ======= EVALUATION ==========
        avg_val_loss, prediction_dict = valid_epoch(valid_loader, model, criterion, device)
        predictions = prediction_dict["predictions"]
        # ======= SCORING ==========
        score = get_score(valid_labels, sigmoid(predictions))

        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Score: {score:.4f}')
        
        if config.WANDB:
            wandb.log({f"[fold_{fold}] epoch": epoch+1, 
                       f"[fold_{fold}] avg_train_loss": avg_loss, 
                       f"[fold_{fold}] avg_val_loss": avg_val_loss,
                       f"[fold_{fold}] score": score})
            
        if score > best_score:
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save(model.state_dict(),
                        paths.OUTPUT_DIR + f"/{config.MODEL.replace('/', '_')}_fold_{fold}_best.pth")
            best_model_predictions = predictions

    valid_folds["preds"] = best_model_predictions

    torch.cuda.empty_cache()
    gc.collect()
    
    return valid_folds

def inference_fn(config, test_df, tokenizer, device):
    # ======== DATASETS ==========
    test_dataset = CustomDataset(config, test_df, tokenizer)

    # ======== DATALOADERS ==========
    test_loader = DataLoader(test_dataset,
                             batch_size=config.BATCH_SIZE_TEST,
                             shuffle=False,
                             num_workers=0,
                             pin_memory=True, drop_last=False)
    
    # ======== MODEL ==========
    model = CustomModel(config, config_path=paths.MODEL_PATH + "/config.pth", pretrained=False)
    state = torch.load(paths.BEST_MODEL_PATH)
    model.load_state_dict(state)
    model.to(device)
    model.eval() # set model in evaluation mode
    prediction_dict = {}
    preds = []
    with tqdm(test_loader, unit="test_batch", desc='Inference') as tqdm_test_loader:
        for step, batch in enumerate(tqdm_test_loader):
            inputs = batch.pop("inputs")
            ids = batch.pop("ids")
            inputs = collate(inputs) # collate inputs
            for k, v in inputs.items():
                inputs[k] = v.to(device) # send inputs to device
            with torch.no_grad():
                y_preds = model(inputs) # forward propagation pass
            preds.append(y_preds.to('cpu').numpy()) # save predictions

    prediction_dict["predictions"] = np.concatenate(preds) # np.array() of shape (fold_size, target_cols)
    prediction_dict["ids"] = ids
    return prediction_dict

def get_result(oof_df):
    labels = oof_df["generated"].values
    preds = oof_df["preds"].values
    score = get_score(labels, preds)
    LOGGER.info(f'Score: {score:<.4f}')

if config.TRAIN:
    train_df = pd.read_csv(paths.PROCESSED_DATA)
    print(f"Train essays dataframe has shape: {train_df.shape}")

    '''
        Stratified K Fold
    '''
    skf = StratifiedKFold(n_splits=5)
    X = train_df.loc[:, train_df.columns != "generated"]
    y = train_df.loc[:, train_df.columns == "generated"]

    for i, (train_index, valid_index) in enumerate(skf.split(X, y)):
        train_df.loc[valid_index, "fold"] = i
        
    print(train_df.groupby("fold")["generated"].value_counts())
    train_df.head()

    #Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL)
    tokenizer.save_pretrained(paths.OUTPUT_DIR + '/tokenizer/')
    print(tokenizer)


    lengths = []
    tqdm_loader = tqdm(train_df['text'].fillna("").values, total=len(train_df))
    for text in tqdm_loader:
        length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
        lengths.append(length)
    oof_df = pd.DataFrame()
    for fold in range(config.FOLDS):
        if fold == 0:
            _oof_df = train_loop(train_df, fold)
            oof_df = pd.concat([oof_df, _oof_df])
            LOGGER.info(f"========== Fold: {fold} result ==========")
            get_result(_oof_df)
    oof_df = oof_df.reset_index(drop=True)
    LOGGER.info(f"========== CV ==========")
    get_result(oof_df)
    oof_df.to_csv(paths.OUTPUT_DIR + '/oof_df.csv', index=False)
    if config.WANDB:
        wandb.finish()

    oof_df["preds"] = oof_df["preds"].apply(lambda x: sigmoid(x))

    
    def binarize(x, threshold):
        if x > threshold:
            x = 1
        else:
            x = 0
        return x

    # Assuming df is your pandas DataFrame
    oof_df["binary"] = oof_df["preds"].apply(lambda x: binarize(x, 0.5))
    true_labels = oof_df["generated"].values
    predicted_labels = oof_df["binary"].values

    # Get the unique classes from both true and predicted labels
    classes = np.unique(np.concatenate((true_labels, predicted_labels)))

    # Compute the confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()
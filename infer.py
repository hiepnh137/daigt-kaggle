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
import tokenizers
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig


class config:
    BATCH_SIZE_TEST = 8
    DEBUG = False
    GRADIENT_CHECKPOINTING = True
    MAX_LEN = 1024
    MODEL = "microsoft/deberta-v3-base"
    NUM_WORKERS = 0 
    PRINT_FREQ = 20
    SEED = 42


class paths:
    MODEL_PATH = "output"
    BEST_MODEL_PATH = "output/microsoft_deberta-v3-base_fold_0_best.pth"
    TEST_ESSAYS = "data/LLM-DetectAI/test_essays.csv"
    SUBMISSION_CSV = "data/LLM-DetectAI/sample_submission.csv"

# def get_score(y_trues, y_preds):
#     mcrmse_score, scores = MCRMSE(y_trues, y_preds)
#     return mcrmse_score, scores


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
    
seed_everything(seed=config.SEED)
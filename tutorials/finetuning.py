from methylbert.utils import set_seed
from torch.utils.data import DataLoader
from methylbert.data.vocab import MethylVocab
from methylbert.data.dataset import MethylBertFinetuneDataset
from methylbert.trainer import MethylBertFinetuneTrainerWithClassifier
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import torch
warnings.filterwarnings("ignore") # Ignore warnings for a clear notebook
torch.multiprocessing.set_sharing_strategy('file_system')

set_seed(42)
seq_len=150
n_mers=3
batch_size=4
num_workers=2
output_path = "tmp/fine_tune/"

# Creat a look-up table
tokenizer = MethylVocab(n_mers)

# Read number of classes(labels) from dmrs.csv, and generate "label2id" & "id2label" for config
df_dmr = pd.read_csv("tmp/dmrs.csv")
if len(df_dmr["ctype"]) > 0:
    ctypes = pd.unique(df_dmr["ctype"])
    num_ctypes = len(ctypes)
    
    df_id2label = ctypes.copy()
    df_id2label.insert(1, "id2label", [i for i in range(num_ctypes)])
    dict_id2label = df_id2label.to_dict()
    
    df_label2id = ctypes.copy()
    df_label2id.insert(0, "label2id", [i for i in range(num_ctypes)])
    dict_label2id = df_label2id.to_dict()
    
    print(f"Found {num_ctypes} cell types from data.")
else
    raise ValueError('Can\'t find column "ctype" from dmrs.csv.')

# Load the data files int a data set object
train_dataset = 0("tmp/train_seq.csv", 
                                          tokenizer, 
                                          seq_len=seq_len)
test_dataset = MethylBertFinetuneDataset("tmp/test_seq.csv", 
                                         tokenizer,seq_len=seq_len) 

# Load the data into a data loader
train_data_loader = DataLoader(train_dataset, batch_size= batch_size, 
                               num_workers= num_workers, pin_memory=False,  
                               shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size= batch_size, 
                              num_workers= num_workers, pin_memory=True,  
                              shuffle=False)
trainer = MethylBertFinetuneTrainerWithClassifier(len(tokenizer), 
                      save_path=output_path, 
                      train_dataloader=train_data_loader, 
                      test_dataloader=test_data_loader,
                      lr=1e-4, with_cuda=True, 
                      log_freq=1,
                      #eval_freq=10, #activate this only when you want to evaluate the model with test_data_loader
                      warmup_step=3,
                      num_classes=num_ctypes,
                      id2label=dict_id2label,
                      label2id=dict_label2id,
                      ignore_mismatched_sizes=True)
trainer.load("hanyangii/methylbert_hg19_4l")
trainer.train(steps=10)
df_train  = pd.read_csv("tmp/fine_tune/train.csv", sep="\t")
df_train.head()
df_eval  = pd.read_csv("tmp/fine_tune/eval.csv", sep="\t")
df_eval.head()
sns.lineplot(data=df_train, x="step", y="loss", label="train loss")
sns.lineplot(data=df_eval, x="step", y="loss", label="eval loss")
plt.savefig("loss.jpg")
sns.lineplot(data=df_train, x="step", y="lr", label="learning rate", color="m")
plt.savefig("learning_rate.jpg")

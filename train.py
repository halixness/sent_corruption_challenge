#############################################
# Diego Calanzone
# Research Interest Demonstration
# University of Trento, Italy
#############################################

from dataset import *
from trainer import Trainer
from model import EnglishDistinguisher
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # Used to report errors on CUDA side

config = {
    "trainer":{
        "epochs": 5,
        "lr": 0.0005,
        "device": "cuda",
        "val_interval": 5,
        "batch_size": 128,
        "train_split": 0.8
    },
    "model": {
        "emsize": 384,
        "d_hid": 384,
        "nlayers": 4,
        "nhead": 4,
        "dropout": 0.3,
    }
}

train_file = os.path.join("data", "train.txt")
train, val, tokenizer = get_datasplits(
    path=train_file, 
    batch_size=config["trainer"]["batch_size"],
    train_split=config["trainer"]["train_split"]
)

ntokens = len(tokenizer)
model = EnglishDistinguisher(
    ntoken=ntokens, 
    d_model=config["model"]["emsize"], 
    nhead=config["model"]["nhead"], 
    d_hid=config["model"]["d_hid"], 
    nlayers=config["model"]["nlayers"], 
    dropout=config["model"]["dropout"]
).to(config["trainer"]["device"])


trainer = Trainer(model=model, config=config["trainer"])
trainer.train(train, val)
import json
from os import makedirs
from os.path import join

# CONSTANTS
MODEL_DIR = "/mnt/beegfs/home/laiking/miti_laiking/models"
ROOT_DIR = "/mnt/beegfs/home/laiking/miti_laiking/code/semantic_similarity"
DATA_DIR = f"{ROOT_DIR}/data"
CONFIG_DIR = f"{ROOT_DIR}/configs"
TEMPLATE_PATH = f"{CONFIG_DIR}/template.json"
TEST_OUT_DIR = f"{ROOT_DIR}/out/test"
TRAIN_OUT_DIR = f"{ROOT_DIR}/out/train"
config = {}

# VARIABLES
dataset = "ak-outcome-sim" 
random_seeds_nb = 5 # nb of seeds for each model (to mitigate effect of random initialization and small dataset eval)
train_paradigm = "cross-encoder"
models_list = [
    ("bert-base-uncased","hf"),
    ("biobert-base-cased-v1.2","hf"),
    ("scibert-scivocab-uncased","hf"),
    ("BiomedNLP-PubMedBERT-base-uncased-abstract","hf"),
    ("BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext","hf"),
    ("BioLinkBERT-base","hf"),
    ("all-mpnet-base-v2","st"),
]

# HYPERPARAMETERS
config["epochs"]=10
config["warmup_ratio"]=0.1
config["eval_steps_per_epoch"]=4
config["batch_size"]=128
config["learning_rate"]=1e-5
config["use_amp"]=True # enable or disable pytorch automatic mixed precision training

# start config creation
out_dir = join(CONFIG_DIR, dataset + "_" + train_paradigm)
makedirs(out_dir, exist_ok=True)
config["dataset_path"] = join(DATA_DIR, dataset)
config["train_paradigm"] = train_paradigm

for seed in range(random_seeds_nb):
    for model_name, model_type in models_list:
        config["model_path"] = join(MODEL_DIR, model_name)
        config["model_type"] = model_type
        config["seed"] = seed
        config["exp_name"] = f"{model_name}_seed{seed}_{dataset}"
        config["test_save_path"] = join(TEST_OUT_DIR, config["exp_name"])
        config["model_save_path"] = join(TRAIN_OUT_DIR, config["exp_name"])
        out_file = f"{model_name}_seed{seed}.json"
        with open(join(out_dir, out_file), "w") as f:
            json.dump(config, f, indent=4)

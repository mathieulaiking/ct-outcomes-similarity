from sentence_transformers import SentenceTransformer, InputExample, models
from sentence_transformers.losses import CosineSimilarityLoss
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from datasets import load_from_disk
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from math import ceil
import logging
import random
import torch
import numpy


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("dataset_disk_path", type=str)
    parser.add_argument("model_path", type=str)
    parser.add_argument("model_save_path", type=str)
    parser.add_argument("test_save_path", type=str)
    
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--eval_steps_per_epoch", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--classifier_dropout", type=float, default=0.1)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)

def get_train_dataloader(hf_dataset_path:str,
                         batch_size:int) -> DataLoader:
    training_set = load_from_disk(hf_dataset_path)["train"]
    train_examples = [InputExample(
                guid = str(example['sent_id']),
                texts = [example["sentence1"], example["sentence2"]],
                label = float(example["label"])
            ) for example in training_set]
    logging.info("Train samples: {}".format(len(train_examples)))
    return DataLoader(train_examples, shuffle=False, batch_size = batch_size)

def get_evaluator(hf_dataset_path:str,
                  split:str,
                  batch_size:int):
    dataset = load_from_disk(hf_dataset_path)[split]
    evaluator = BinaryClassificationEvaluator(
        sentences1=dataset["sentence1"],
        sentences2=dataset["sentence2"],
        labels=dataset["label"],
        batch_size=batch_size,
    )
    return evaluator

def main():

    args = parse_args()

    # Set random state
    set_seed(config["seed"])

 

    # Load train and validation sets
    train_dataloader = get_train_dataloader(args.dataset_disk_path, args.batch_size)
    validation_evaluator = get_evaluator(args.dataset_disk_path, "validation", args.batch_size, config["train_paradigm"])
    # Load model
    if model_type == "st":
        model = SentenceTransformer(model_name_or_path)
    elif model_type == "hf":
        word_embedding_model = models.Transformer(model_name_or_path)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    # Loss and steps
    train_loss = CosineSimilarityLoss(self.model)
    train_steps = len(train_dataloader)
    warmup_steps = ceil(train_steps * epochs * warmup_ratio)
    evaluation_steps=int(train_steps/eval_steps_per_epoch)
    # Train
    model.fit(
        train_objectives=[(train_dataloader,train_loss)],
        evaluator=evaluator,
        epochs = epochs,
        optimizer_params={'lr': learning_rate},
        evaluation_steps=evaluation_steps,
        warmup_steps = warmup_steps,
        output_path = model_save_path,
        show_progress_bar=False,
        save_best_model=True,
        use_amp=use_amp
    )
    # test
    test_evaluator = get_evaluator(args.dataset_disk_path, "test", args.batch_size, config["train_paradigm"])
    best_model = SentenceTransformer(config["model_save_path"])
    best_model.evaluate(test_evaluator, config["test_save_path"])


if __name__ == "__main__":
    main()
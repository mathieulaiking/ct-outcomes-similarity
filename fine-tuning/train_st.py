from sentence_transformers import SentenceTransformer, InputExample, models
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.losses import CosineSimilarityLoss
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from datasets import load_from_disk
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from math import ceil
import logging
import random
import torch
import numpy
import json


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
                  batch_size:int,
                  train_paradigm:str="bi-encoder"):
    dataset = load_from_disk(hf_dataset_path)[split]
    if train_paradigm == "bi-encoder":
        evaluator = BinaryClassificationEvaluator(
            sentences1=dataset["sentence1"],
            sentences2=dataset["sentence2"],
            labels=dataset["label"],
            batch_size=batch_size,
        )
    elif train_paradigm == "cross-encoder":
        evaluator = CEBinaryClassificationEvaluator(
            sentence_pairs = [[s1,s2] for s1,s2 in zip(dataset["sentence1"], dataset["sentence2"])],
            labels=dataset["label"],
        )
    return evaluator


class STOutcomeTrainer:
    """Sentence Transformers Outcome Similarity Trainer
    for a single dataset split"""
    
    def __init__(self, model_name_or_path:str="", model_type:str="st", train_paradigm:str="bi-encoder"):
       
        if train_paradigm == "bi-encoder":
            if model_type == "st":
                self.model = SentenceTransformer(model_name_or_path)
            elif model_type == "hf":
                word_embedding_model = models.Transformer(model_name_or_path)
                pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
                self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        elif train_paradigm == "cross-encoder":
            self.model = CrossEncoder(model_name_or_path, num_labels=1)
        # setting the train paradigm for fit arguments
        self.train_paradigm = train_paradigm
        
    def train(self, 
              train_dataloader:DataLoader,
              evaluator:BinaryClassificationEvaluator,
              epochs:int,
              warmup_ratio:float,
              model_save_path:str,
              eval_steps_per_epoch:int,
              learning_rate:float,
              use_amp:bool
            ):
        train_loss = CosineSimilarityLoss(self.model)
        train_steps = len(train_dataloader)
        warmup_steps = ceil(train_steps * epochs * warmup_ratio)
        evaluation_steps=int(train_steps/eval_steps_per_epoch)

        if self.train_paradigm == "bi-encoder" :
            self.model.fit(
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
        elif self.train_paradigm == "cross-encoder":
            self.model.fit(
                train_dataloader=train_dataloader,
                evaluator=evaluator,
                epochs=epochs,
                optimizer_params={'lr': learning_rate},
                evaluation_steps=evaluation_steps,
                warmup_steps=warmup_steps,
                output_path=model_save_path,
                show_progress_bar=False,
                save_best_model=True,
                use_amp=use_amp)
    

def main():
    parser = ArgumentParser()
    parser.add_argument("config_path", type=str)
    config = json.load(open(parser.parse_args().config_path, "r"))
    # Set random state
    set_seed(config["seed"])
    # Load train and validation sets
    train_dataloader = get_train_dataloader(config["dataset_path"], config["batch_size"])
    validation_evaluator = get_evaluator(config["dataset_path"], "validation", config["batch_size"], config["train_paradigm"])
    # Load model
    trainer = STOutcomeTrainer(config["model_path"], config["model_type"], config["train_paradigm"])
    # Train
    trainer.train(
        train_dataloader,
        validation_evaluator,
        epochs=config["epochs"],
        warmup_ratio=config["warmup_ratio"],
        model_save_path=config["model_save_path"],
        eval_steps_per_epoch=config["eval_steps_per_epoch"],
        learning_rate=config["learning_rate"],
        use_amp=config["use_amp"]
    )
    # test
    test_evaluator = get_evaluator(config["dataset_path"], "test", config["batch_size"], config["train_paradigm"])
    if config["train_paradigm"] == "bi-encoder":
        best_model = SentenceTransformer(config["model_save_path"])
        best_model.evaluate(test_evaluator, config["test_save_path"])
    elif config["train_paradigm"] == "cross-encoder":
        best_model = CrossEncoder(config["model_save_path"])
        test_evaluator(best_model, output_path=config["test_save_path"])

if __name__ == "__main__":
    main()
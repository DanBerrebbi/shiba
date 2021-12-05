import os
from typing import Dict

import torch
import torchmetrics
import transformers
from datasets import load_dataset
from transformers import HfArgumentParser, Trainer, EvalPrediction, DataCollatorWithPadding

from training.model2 import ShibaForClassification
from training.helpers import DataArguments, ShibaClassificationArgs, ClassificationDataCollator,\
    get_model_hyperparams, get_base_shiba_state_dict

from shiba import CodepointTokenizer

import wandb


os.environ["WANDB_DISABLED"] = "false"
os.environ["WANDB_API_KEY"] = "34cb67c5080571acc2617191cde3876ab418103b"


def main():
    transformers.logging.set_verbosity_info()
    parser = HfArgumentParser((DataArguments, ShibaClassificationArgs))
    data_args, training_args = parser.parse_args_into_dataclasses()

    PROJECT_NAME = "xnli"
    wandb.init(project=PROJECT_NAME, entity="dan_berrebbi")
    tokenizer = CodepointTokenizer()

    # Load dataset
    sw_xnli_dataset = load_dataset('xnli', 'sw')
    labels = sw_xnli_dataset['train'].features['label']
    data_collator = ClassificationDataCollator()

    # Define model
    model_hyperparams = get_model_hyperparams(training_args)

    model = ShibaForClassification(vocab_size=labels.num_classes,
                                   **model_hyperparams)

    if training_args.resume_from_checkpoint:
        print('Loading and using base shiba states from', training_args.resume_from_checkpoint)
        checkpoint_state_dict = torch.load(training_args.resume_from_checkpoint)
        model.shiba_model.load_state_dict(get_base_shiba_state_dict(checkpoint_state_dict), strict=False)

    def process_example(example: Dict) -> Dict:
        # Concat premise and hypothesis
        premise_ids = tokenizer.encode(example['premise'])['input_ids']
        hypothesis_ids = tokenizer.encode(example['hypothesis'])['input_ids']
        input_ids = torch.cat([premise_ids, torch.tensor([tokenizer.SEP]), hypothesis_ids])
        segment_ids = torch.cat([torch.ones_like(premise_ids),
                                 torch.tensor([0]),
                                 torch.ones_like(hypothesis_ids) * 2])

        return {
            'input_ids': input_ids[:model.config.max_length],
            'segment_ids': segment_ids[:model.config.max_length],
            'labels': example['label']
        }

    def compute_metrics(pred: EvalPrediction) -> Dict:
        accuracy_metric = torchmetrics.Accuracy(num_classes=3)
        F1_metric = torchmetrics.F1(multiclass=True)
        label_probs, embeddings = pred.predictions
        labels = torch.tensor(pred.label_ids)
        label_probs = torch.exp(torch.tensor(label_probs))  # undo the log in log softmax, get indices

        accuracy_metric.update(label_probs, labels)
        F1_metric.update(label_probs, labels)

        return {
            'accuracy': accuracy_metric.compute().item(),
            'F1': F1_metric.compute().item()
        }

    print(training_args)
    trainer = Trainer(model=model,
                      args=training_args,
                      data_collator=data_collator,
                      train_dataset=sw_xnli_dataset['validation'].map(process_example),
                      eval_dataset=sw_xnli_dataset['validation'].map(process_example),
                      compute_metrics=compute_metrics
                      )
    trainer.train()
    posttrain_metrics = trainer.predict(sw_xnli_dataset['test'].map(process_example)).metrics
    print(posttrain_metrics)

if __name__ == "__main__":
    main()

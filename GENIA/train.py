from GENIA.build_dataset import GeniaDataset
from transformers import \
    (RobertaTokenizer,
     RobertaConfig,
     TrainingArguments,
     Trainer,
     DataCollatorForLanguageModeling,
     RobertaForMaskedLM)

import pathlib
from transformers.adapters.configuration import AdapterConfig


def get_train_args():
    return TrainingArguments(
        learning_rate=1e-4,
        num_train_epochs=1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        logging_steps=200,
        output_dir="../GENIA/training_output",
        overwrite_output_dir=True,
        # The next line is important to ensure the dataset labels are properly passed to the model
        remove_unused_columns=False,
    )


def get_tokenizer():
    return RobertaTokenizer.from_pretrained("roberta-base")


def get_adapter_args():
    return AdapterConfig.load(
        config="pfeiffer",
        non_linearity="relu",
        reduction_factor=16,
    )


def get_data_collator(tokenizer, mlm_prob=0.15, ):
    return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mlm_prob)


def main():
    tokenizer = get_tokenizer()

    print('Load data')
    train_path = pathlib.Path(__file__).absolute().parent.joinpath('genia_parsed_corpus_train.txt')
    eval_path = pathlib.Path(__file__).absolute().parent.joinpath('genia_parsed_corpus_eval.txt')
    train_dataset = GeniaDataset(tokenizer=tokenizer, data_path=train_path)
    eval_dataset = GeniaDataset(tokenizer=tokenizer, data_path=eval_path)

    print('Load pre-trained ROBERTA')
    configuration = RobertaConfig()
    model = RobertaForMaskedLM(configuration)

    print('Load and set up adapter')
    task_name = 'mlm'
    adapter_config = get_adapter_args()
    # Enable adapter training
    model.add_adapter(task_name, config=adapter_config)
    # Freeze all model weights except for adapter weights
    model.train_adapter([task_name])
    model.set_active_adapters([task_name])

    # Get data collator
    data_collator = get_data_collator(tokenizer)

    # Get train args
    train_args = get_train_args()

    # Initialize trainer
    print('Initialize trainer')
    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    print('Train')
    trainer.train()

main()
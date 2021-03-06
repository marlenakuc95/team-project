from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import torch
import warnings

from transformers import (
                          CONFIG_MAPPING,
                          MODEL_FOR_MASKED_LM_MAPPING,
                          MODEL_FOR_CAUSAL_LM_MAPPING,
                          PreTrainedTokenizer,
                          TrainingArguments,
                          AutoConfig,
                          AutoTokenizer,
                          AutoModelWithLMHead,
                          AutoModelForCausalLM,
                          AutoModelForMaskedLM,
                          LineByLineTextDataset,
                          TextDataset,
                          DataCollatorForLanguageModeling,
                          DataCollatorForWholeWordMask,
                          DataCollatorForPermutationLanguageModeling,
                          PretrainedConfig,
                          Trainer,
                          set_seed,
                          )

# Look for gpu to use. Will use `cpu` by default if no gpu found.
# 2nd device is the free one
# Running the following command on the terminal solves the CUDA out of memory problem
#export CUDA_VISIBLE_DEVICES=1

class ModelDataArguments(object):
    r"""Define model and data configuration needed to perform pretraining.

    Eve though all arguments are optional there still needs to be a certain
    number of arguments that require values attributed.

    Arguments:

      train_data_file (:obj:`str`, `optional`):
        Path to your .txt file dataset. If you have an example on each line of
        the file make sure to use line_by_line=True. If the data file contains
        all text data without any special grouping use line_by_line=False to move
        a block_size window across the text file.
        This argument is optional and it will have a `None` value attributed
        inside the function.

      eval_data_file (:obj:`str`, `optional`):
        Path to evaluation .txt file. It has the same format as train_data_file.
        This argument is optional and it will have a `None` value attributed
        inside the function.

      line_by_line (:obj:`bool`, `optional`, defaults to :obj:`False`):
        If the train_data_file and eval_data_file contains separate examples on
        each line then line_by_line=True. If there is no separation between
        examples and train_data_file and eval_data_file contains continuous text
        then line_by_line=False and a window of block_size will be moved across
        the files to acquire examples.
        This argument is optional and it has a default value.

      mlm (:obj:`bool`, `optional`, defaults to :obj:`False`):
        Is a flag that changes loss function depending on model architecture.
        This variable needs to be set to True when working with masked language
        models like bert or roberta and set to False otherwise. There are
        functions that will raise ValueError if this argument is
        not set accordingly.
        This argument is optional and it has a default value.

      whole_word_mask (:obj:`bool`, `optional`, defaults to :obj:`False`):
        Used as flag to determine if we decide to use whole word masking or not.
        Whole word masking means that whole words will be masked during training
        instead of tokens which can be chunks of words.
        This argument is optional and it has a default value.

      mlm_probability(:obj:`float`, `optional`, defaults to :obj:`0.15`):
        Used when training masked language models. Needs to have mlm set to True.
        It represents the probability of masking tokens when training model.
        This argument is optional and it has a default value.

      plm_probability (:obj:`float`, `optional`, defaults to :obj:`float(1/6)`):
        Flag to define the ratio of length of a span of masked tokens to
        surrounding context length for permutation language modeling.
        Used for XLNet.
        This argument is optional and it has a default value.

      max_span_length (:obj:`int`, `optional`, defaults to :obj:`5`):
        Flag may also be used to limit the length of a span of masked tokens used
        for permutation language modeling. Used for XLNet.
        This argument is optional and it has a default value.

      block_size (:obj:`int`, `optional`, defaults to :obj:`-1`):
        It refers to the windows size that is moved across the text file.
        Set to -1 to use maximum allowed length.
        This argument is optional and it has a default value.

      overwrite_cache (:obj:`bool`, `optional`, defaults to :obj:`False`):
        If there are any cached files, overwrite them.
        This argument is optional and it has a default value.

      model_type (:obj:`str`, `optional`):
        Type of model used: bert, roberta, gpt2.
        More details: https://huggingface.co/transformers/pretrained_models.html
        This argument is optional and it will have a `None` value attributed
        inside the function.

      model_config_name (:obj:`str`, `optional`):
        Config of model used: bert, roberta, gpt2.
        More details: https://huggingface.co/transformers/pretrained_models.html
        This argument is optional and it will have a `None` value attributed
        inside the function.

      tokenizer_name: (:obj:`str`, `optional`)
        Tokenizer used to process data for training the model.
        It usually has same name as model_name_or_path: bert-base-cased,
        roberta-base, gpt2 etc.
        This argument is optional and it will have a `None` value attributed
        inside the function.

      model_name_or_path (:obj:`str`, `optional`):
        Path to existing transformers model or name of
        transformer model to be used: bert-base-cased, roberta-base, gpt2 etc.
        More details: https://huggingface.co/transformers/pretrained_models.html
        This argument is optional and it will have a `None` value attributed
        inside the function.

      model_cache_dir (:obj:`str`, `optional`):
        Path to cache files to save time when re-running code.
        This argument is optional and it will have a `None` value attributed
        inside the function.

    Raises:

          ValueError: If `CONFIG_MAPPING` is not loaded in global variables.

          ValueError: If `model_type` is not present in `CONFIG_MAPPING.keys()`.

          ValueError: If `model_type`, `model_config_name` and
            `model_name_or_path` variables are all `None`. At least one of them
            needs to be set.

          warnings: If `model_config_name` and `model_name_or_path` are both
            `None`, the model will be trained from scratch.

          ValueError: If `tokenizer_name` and `model_name_or_path` are both
            `None`. We need at least one of them set to load tokenizer.

    """

    def __init__(self, train_data_file=None, eval_data_file=None,
                 line_by_line=False, mlm=False, mlm_probability=0.15,
                 whole_word_mask=False, plm_probability=float(1 / 6),
                 max_span_length=5, block_size=-1, overwrite_cache=False,
                 model_type=None, model_config_name=None, tokenizer_name=None,
                 model_name_or_path=None, model_cache_dir=None):

        # Make sure CONFIG_MAPPING is imported from transformers module.
        if 'CONFIG_MAPPING' not in globals():
            raise ValueError('Could not find `CONFIG_MAPPING` imported! Make sure' \
                             ' to import it from `transformers` module!')

        # Make sure model_type is valid.
        if (model_type is not None) and (model_type not in CONFIG_MAPPING.keys()):
            raise ValueError('Invalid `model_type`! Use one of the following: %s' %
                             (str(list(CONFIG_MAPPING.keys()))))

        # Make sure that model_type, model_config_name and model_name_or_path
        # variables are not all `None`.
        if not any([model_type, model_config_name, model_name_or_path]):
            raise ValueError('You can`t have all `model_type`, `model_config_name`,' \
                             ' `model_name_or_path` be `None`! You need to have' \
                             'at least one of them set!')

        # Check if a new model will be loaded from scratch.
        if not any([model_config_name, model_name_or_path]):
            # Setup warning to show pretty. This is an overkill
            warnings.formatwarning = lambda message, category, *args, **kwargs: \
                '%s: %s\n' % (category.__name__, message)
            # Display warning.
            warnings.warn('You are planning to train a model from scratch! ????')

        # Check if a new tokenizer wants to be loaded.
        # This feature is not supported!
        if not any([tokenizer_name, model_name_or_path]):
            # Can't train tokenizer from scratch here! Raise error.
            raise ValueError('You want to train tokenizer from scratch! ' \
                             'That is not possible yet! You can train your own ' \
                             'tokenizer separately and use path here to load it!')

        # Set all data related arguments.
        self.train_data_file = train_data_file
        self.eval_data_file = eval_data_file
        self.line_by_line = line_by_line
        self.mlm = mlm
        self.whole_word_mask = whole_word_mask
        self.mlm_probability = mlm_probability
        self.plm_probability = plm_probability
        self.max_span_length = max_span_length
        self.block_size = block_size
        self.overwrite_cache = overwrite_cache

        # Set all model and tokenizer arguments.
        self.model_type = model_type
        self.model_config_name = model_config_name
        self.tokenizer_name = tokenizer_name
        self.model_name_or_path = model_name_or_path
        self.model_cache_dir = model_cache_dir
        return


def get_model_config(args: ModelDataArguments):
    r"""
    Get model configuration.

    Using the ModelDataArguments return the model configuration.

    Arguments:

      args (:obj:`ModelDataArguments`):
        Model and data configuration arguments needed to perform pretraining.

    Returns:

      :obj:`PretrainedConfig`: Model transformers configuration.

    Raises:

      ValueError: If `mlm=True` and `model_type` is NOT in ["bert",
            "roberta", "distilbert", "camembert"]. We need to use a masked
            language model in order to set `mlm=True`.
    """

    # Check model configuration.
    if args.model_config_name is not None:
        # Use model configure name if defined.
        model_config = AutoConfig.from_pretrained(args.model_config_name,
                                                  cache_dir=args.model_cache_dir)

    elif args.model_name_or_path is not None:
        # Use model name or path if defined.
        model_config = AutoConfig.from_pretrained(args.model_name_or_path,
                                                  cache_dir=args.model_cache_dir)

    else:
        # Use config mapping if building model from scratch.
        model_config = CONFIG_MAPPING[args.model_type]()

    # Make sure `mlm` flag is set for Masked Language Models (MLM).
    if (model_config.model_type in ["bert", "roberta", "distilbert",
                                    "camembert"]) and (args.mlm is False):
        raise ValueError('BERT and RoBERTa-like models do not have LM heads ' \
                         'butmasked LM heads. They must be run setting `mlm=True`')

    # Adjust block size for xlnet.
    if model_config.model_type == "xlnet":
        # xlnet used 512 tokens when training.
        args.block_size = 512
        # setup memory length
        model_config.mem_len = 1024

    return model_config


def get_tokenizer(args: ModelDataArguments):
    r"""
    Get model tokenizer.

    Using the ModelDataArguments return the model tokenizer and change
    `block_size` form `args` if needed.

    Arguments:

      args (:obj:`ModelDataArguments`):
        Model and data configuration arguments needed to perform pretraining.

    Returns:

      :obj:`PreTrainedTokenizer`: Model transformers tokenizer.

    """

    # Check tokenizer configuration.
    if args.tokenizer_name:
        # Use tokenizer name if define.
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name,
                                                  cache_dir=args.model_cache_dir)

    elif args.model_name_or_path:
        # Use tokenizer name of path if defined.
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                                  cache_dir=args.model_cache_dir)

    # Setp data block size.
    if args.block_size <= 0:
        # Set block size to maximum length of tokenizer.
        # Input block size will be the max possible for the model.
        # Some max lengths are very large and will cause a
        args.block_size = tokenizer.model_max_length
    else:
        # Never go beyond tokenizer maximum length.
        args.block_size = min(args.block_size, tokenizer.model_max_length)

    return tokenizer


def get_new_tokenizer(train_text_path):
    # Setup for from scratch BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

    # To split words by whitespace
    tokenizer.pre_tokenizer = Whitespace()


    # Training the tokenizer
    tokenizer.train([train_text_path], trainer)

    return tokenizer


def get_model(args: ModelDataArguments, model_config):
    r"""
    Get model.

    Using the ModelDataArguments return the actual model.

    Arguments:

      args (:obj:`ModelDataArguments`):
        Model and data configuration arguments needed to perform pretraining.

      model_config (:obj:`PretrainedConfig`):
        Model transformers configuration.

    Returns:

      :obj:`torch.nn.Module`: PyTorch model.

    """

    # Make sure MODEL_FOR_MASKED_LM_MAPPING and MODEL_FOR_CAUSAL_LM_MAPPING are
    # imported from transformers module.
    if ('MODEL_FOR_MASKED_LM_MAPPING' not in globals()) and \
            ('MODEL_FOR_CAUSAL_LM_MAPPING' not in globals()):
        raise ValueError('Could not find `MODEL_FOR_MASKED_LM_MAPPING` and' \
                         ' `MODEL_FOR_MASKED_LM_MAPPING` imported! Make sure to' \
                         ' import them from `transformers` module!')

    # Check if using pre-trained model or train from scratch.
    if args.model_name_or_path:
        # Use pre-trained model.
        if type(model_config) in MODEL_FOR_MASKED_LM_MAPPING.keys():
            # Masked language modeling head.
            return AutoModelForMaskedLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=model_config,
                cache_dir=args.model_cache_dir,
            )
        elif type(model_config) in MODEL_FOR_CAUSAL_LM_MAPPING.keys():
            # Causal language modeling head.
            return AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in
                             args.model_name_or_path),
                config=model_config,
                cache_dir=args.model_cache_dir)
        else:
            raise ValueError(
                'Invalid `model_name_or_path`! It should be in %s or %s!' %
                (str(MODEL_FOR_MASKED_LM_MAPPING.keys()),
                 str(MODEL_FOR_CAUSAL_LM_MAPPING.keys())))

    else:
        # Use model from configuration - train from scratch.
        print("Training new model from scratch!")
        model = AutoModelWithLMHead.from_config(config)
        return model

def get_dataset(args: ModelDataArguments, tokenizer,
                evaluate: bool = False):
    r"""
    Process dataset file into PyTorch Dataset.

    Using the ModelDataArguments return the actual model.

    Arguments:

      args (:obj:`ModelDataArguments`):
        Model and data configuration arguments needed to perform pretraining.

      tokenizer (:obj:`PreTrainedTokenizer`):
        Model transformers tokenizer.

      evaluate (:obj:`bool`, `optional`, defaults to :obj:`False`):
        If set to `True` the test / validation file is being handled.
        If set to `False` the train file is being handled.

    Returns:

      :obj:`Dataset`: PyTorch Dataset that contains file's data.

    """

    # Get file path for either train or evaluate.
    file_path = args.eval_data_file if evaluate else args.train_data_file

    # Check if `line_by_line` flag is set to `True`.
    if args.line_by_line:
        # Each example in data file is on each line.
        return LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path,
                                     block_size=args.block_size)

    else:
        # All data in file is put together without any separation.
        return TextDataset(tokenizer=tokenizer, file_path=file_path,
                           block_size=args.block_size,
                           overwrite_cache=args.overwrite_cache)


def get_collator(args: ModelDataArguments, model_config: PretrainedConfig,
                 tokenizer: PreTrainedTokenizer):
    r"""
    Get appropriate collator function.

    Collator function will be used to collate a PyTorch Dataset object.

    Arguments:

      args (:obj:`ModelDataArguments`):
        Model and data configuration arguments needed to perform pretraining.

      model_config (:obj:`PretrainedConfig`):
        Model transformers configuration.

      tokenizer (:obj:`PreTrainedTokenizer`):
        Model transformers tokenizer.

    Returns:

      :obj:`data_collator`: Transformers specific data collator.

    """

    # Special dataset handle depending on model type.
    if model_config.model_type == "xlnet":
        # Configure collator for XLNET.
        return DataCollatorForPermutationLanguageModeling(
            tokenizer=tokenizer,
            plm_probability=args.plm_probability,
            max_span_length=args.max_span_length,
        )
    else:
        # Configure data for rest of model types.
        if args.mlm and args.whole_word_mask:
            # Use whole word masking.
            return DataCollatorForWholeWordMask(
                tokenizer=tokenizer,
                mlm_probability=args.mlm_probability,
            )
        else:
            # Regular language modeling.
            return DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=args.mlm,
                mlm_probability=args.mlm_probability,
            )


# Define arguments for data, tokenizer and model arguments.
# See comments in `ModelDataArguments` class.
model_data_args = ModelDataArguments(
    train_data_file='/work-ceph/glavas-tp2021/team_project/pretraining/all_texts_train.txt',
    eval_data_file='/work-ceph/glavas-tp2021/team_project/pretraining/all_texts_test.txt',
    line_by_line=True,
    mlm=True,
    whole_word_mask=True,
    mlm_probability=0.15,
    plm_probability=float(1 / 6),
    max_span_length=5,
    block_size=50,
    overwrite_cache=False,
    model_type='distilbert',
    tokenizer_name = "fran-martinez/scibert_scivocab_cased_ner_jnlpba",
    model_cache_dir=None
)

# Define arguments for training
# Note: I only used the arguments I care about. `TrainingArguments` contains
# a lot more arguments. For more details check the awesome documentation:
# https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments
training_args = TrainingArguments(
    # The output directory where the model predictions
    # and checkpoints will be written.
    output_dir='/work-ceph/glavas-tp2021/team_project/pretraining/bert_model',

    # Overwrite the content of the output directory.
    overwrite_output_dir=True,

    # Whether to run training or not.
    do_train=True,

    # Whether to run evaluation on the dev or not.
    do_eval=True,

    # Batch size GPU/TPU core/CPU training.
    per_device_train_batch_size=10,

    # Batch size  GPU/TPU core/CPU for evaluation.
    per_device_eval_batch_size=10,

    # evaluation strategy to adopt during training
    # `no`: No evaluation during training.
    # `steps`: Evaluate every `eval_steps`.
    # `epoch`: Evaluate every end of epoch.
    evaluation_strategy='steps',

    # How often to show logs. I will se this to
    # plot history loss and calculate perplexity.
    logging_steps=700,

    # Number of update steps between two
    # evaluations if evaluation_strategy="steps".
    # Will default to the same value as l
    # logging_steps if not set.
    eval_steps=None,

    # Set prediction loss to `True` in order to
    # return loss for perplexity calculation.
    prediction_loss_only=True,

    # The initial learning rate for Adam.
    # Defaults to 5e-5.
    learning_rate=5e-5,

    # The weight decay to apply (if not zero).
    weight_decay=0,

    # Epsilon for the Adam optimizer.
    # Defaults to 1e-8
    adam_epsilon=1e-8,

    # Maximum gradient norm (for gradient
    # clipping). Defaults to 0.
    max_grad_norm=1.0,
    # Total number of training epochs to perform
    # (if not an integer, will perform the
    # decimal part percents of
    # the last epoch before stopping training).
    num_train_epochs=2,

    # Number of updates steps before two checkpoint saves.
    # Defaults to 500
    save_steps=-1,
)


# Load model configuration.
print('Loading model configuration...')
config = get_model_config(model_data_args)

# Load model tokenizer.
print('Loading model`s tokenizer...')
tokenizer = get_tokenizer(model_data_args)
#tokenizer = get_new_tokenizer("/Volumes/MertOzlutiras/MEGA/MERT/classes/spring_21/BASF-team-project/pretraining/all_texts_train.txt")


# Loading model.
print('Loading actual model...')
model = get_model(model_data_args, config)

# Resize model to fit all tokens in tokenizer.
model.resize_token_embeddings(len(tokenizer))


# Setup train dataset if `do_train` is set.
print('Creating train dataset...')
train_dataset = get_dataset(model_data_args, tokenizer=tokenizer, evaluate=False) if training_args.do_train else None

# Setup evaluation dataset if `do_eval` is set.
print('Creating evaluate dataset...')
eval_dataset = get_dataset(model_data_args, tokenizer=tokenizer, evaluate=True) if training_args.do_eval else None

# Get data collator to modify data format depending on type of model used.
data_collator = get_collator(model_data_args, config, tokenizer)

# Check how many logging prints you'll have. This is to avoid overflowing the
# notebook with a lot of prints. Display warning to user if the logging steps
# that will be displayed is larger than 100.
if (len(train_dataset) // training_args.per_device_train_batch_size \
    // training_args.logging_steps * training_args.num_train_epochs) > 100:
  # Display warning.
  warnings.warn('Your `logging_steps` value will will do a lot of printing!' \
                ' Consider increasing `logging_steps` to avoid overflowing' \
                ' the notebook with a lot of prints!')


# Initialize Trainer.
print('Loading `trainer`...')
trainer = Trainer(model=model,
                  args=training_args,
                  data_collator=data_collator,
                  train_dataset=train_dataset,
                  eval_dataset=eval_dataset,
                  )


# Check model path to save.
if training_args.do_train:
  print('Start training...')

  # Setup model path if the model to train loaded from a local path.
  model_path = (model_data_args.model_name_or_path
                if model_data_args.model_name_or_path is not None and
                os.path.isdir(model_data_args.model_name_or_path)
                else None
                )
  # Run training.
  trainer.train(model_path=model_path)
  # Save model.
  trainer.save_model()

  # For convenience, we also re-save the tokenizer to the same directory,
  # so that you can share your model easily on huggingface.co/models =).
  if trainer.is_world_process_zero():
    tokenizer.save_pretrained(training_args.output_dir)
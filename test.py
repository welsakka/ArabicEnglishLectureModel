from datasets import Dataset

import pandas as pd

from datasets import Audio

import gc

import torch

res = torch.cuda.is_available()
print(res)
## we will load the both of the data here.

train_df = pd.read_csv("train.csv")

test_df = pd.read_csv("test.csv")

## we will rename the columns as "audio", "sentence".

train_df.columns = ["audio", "sentence"]

test_df.columns = ["audio", "sentence"]

#Now we will create the dataset using the class methods Dataset.from_pandas() and cast the audio to an Audio datatype. For example:

train_dataset = Dataset.from_pandas(train_df)

test_dataset = Dataset.from_pandas(test_df)

#We will create arrays of each audio file and append those values as a column in the above datasets. To do this we will use the cast_column function from Dataset. We will also use sampling_rate as an argument so if there is any file we missed in preprocessing step.

train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))

test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))

###  Padding and spectrogram conversion are both handled by the Transformers Whisper feature extractor in a single line of code!
###  To prepare for our audio data, let’s now load the feature extractor from the pre-trained checkpoint:

## import feature extractor

from transformers import WhisperFeatureExtractor

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")

## Load WhisperTokenizer

from transformers import WhisperTokenizer

tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base", language="English", task="transcribe")
                                             
#Combine To Create A WhisperProcessor
#We can combine the tokenizer and feature extractor into a single WhisperProcessor class to make using them easier. 
#This processor object can be applied to audio inputs and model predictions as necessary and derives from the WhisperFeatureExtractor and WhisperProcessor.

from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained("openai/whisper-base", language="English", task="transcribe")

### Now we can write a function to prepare our data ready for the model:

    #Using the batch[„audio] function, we load and resample the audio data. Datasets, as previously mentioned, carry out any necessary resampling 
    #operations in real time.
    #From our 1-dimensional audio array, we compute the log-Mel spectrogram input features using the feature extractor.
    #Using the tokenizer, we encode the transcriptions to create label ids.

def prepare_dataset(examples):

    # compute log-Mel input features from input audio array

    audio = examples["audio"]

    examples["input_features"] = feature_extractor(

        audio["array"], sampling_rate=16000).input_features[0]

    del examples["audio"]

    sentences = examples["sentence"]

    # encode target text to label ids

    examples["labels"] = tokenizer(sentences).input_ids

    del examples["sentence"]

    return examples

#As we don’t need to carry this data, we are deleting the examples["audio"] and examples["labels"]. 
#Additionally, by erasing this data, RAM space is made available.
#Using the dataset’s.map method, we can apply the data preparation function to each of our training examples; 
#this procedure will take 30 to 40 minutes. Additionally, check that your disc has between 30 and 40 GB of free space, 
#as the map function will attempt to write data to the disc for a while.

train_dataset = train_dataset.map(prepare_dataset, num_proc=1)

test_dataset = test_dataset.map(prepare_dataset, num_proc=1)

#Define a data collector

import torch

from dataclasses import dataclass

from typing import Any, Dict, List, Union

@dataclass

class DataCollatorSpeechSeq2SeqWithPadding:

    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        # split inputs and labels since they have to be of different lengths and need different padding methods

        # first treat the audio inputs by simply returning torch tensors

        input_features = [{"input_features": feature["input_features"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences

        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # pad the labels to max length

        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,

        # cut bos token here as it’s append later anyways

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():

            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

#Let’s initialise the data collator we’ve just defined:

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

##Evaluation Metrics
## The evaluation metric that will be applied to our evaluation set is then defined. The "de-facto" metric for rating ASR systems, the Word Error Rate (WER) metric, will be used. Consult the WER docs for more details. The WER metric will be loaded from Evaluate:

import evaluate

metric = evaluate.load("wer")


#Following that, all we need to do is define a function that takes the model predictions as input and outputs the WER metric. In the label ids, this function, compute metrics, first replaces -100 for the pad_token_id (undoing the step we applied in the data collator to ignore padded tokens correctly in the loss). The predicted and label_ids are then converted to strings. The WER between the predictions and reference labels is calculated at the end.

def compute_metrics(pred):

    pred_ids = pred.predictions

    label_ids = pred.label_ids

    # replace -100 with the pad_token_id

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}
#Load a Pre-Trained Checkpoint

 #           Now let’s load the pre-trained Whisper base checkpoint. Again, this is trivial through use of  Transformers!

# Load a Pre-Trained Checkpoint

from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

#Before autoregressive generation begins, the Whisper model generates (forced_decoder_ids). which are token ids that are required as model outputs. These token ids regulate the transcription language and task for zero-shot ASR. As we’ll train the model to predict the correct language (Hindi) and task, we’ll set these ids to None for fine-tuning (transcription). Additionally, some tokens (suppress_tokens) are entirely suppressed during generation.

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

#Also read this enlightening blog post on the dangers of adversarial AI.
#Define the Training Arguments

#We define all the training-related parameters in the last step. The following is an explanation of a subset of parameters:

 #   output_dir: is a local directory in which you can save the model weights. This will be the repository name on the Hugging Face Hub.
  #  generation_max_length: maximum number of tokens to autoregressive generate throughout assessment.
  #  save_steps: during training, intermediate checkpoints can be saved and also be uploaded asynchronously to the Hub every save_steps training steps.
  #  eval_steps: during training, evaluation of intermediate checkpoints will be accomplished every eval_steps training steps.
  #  report_to: where to save training logs. Supported platforms are "azure_ml", "comet_ml", "mlflow", "neptune", "tensorboard" and "wandb". Pick your favorite or leave it as "tensorboard" to log to the Hub.
  #  Push_to_hub: Since we don’t want to force our models into hugging faces, we set this value to False.

#Contact the Seq2SeqTrainingArguments docs for more information on the other training arguments.

# Define the Training Arguments

from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(

    output_dir="./whisper-base-en",  # change to a repo name of your choice

    per_device_train_batch_size=16,

    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size

    learning_rate=1e-5,

    warmup_steps=500,

    max_steps=15000,

    gradient_checkpointing=True,

    fp16=True,

    evaluation_strategy="steps",

    per_device_eval_batch_size=1,

    predict_with_generate=True,

    generation_max_length=225,

    save_steps=500,

    eval_steps=500,

    # logging_steps=25,

    report_to=["tensorboard"],

    load_best_model_at_end=True,

    metric_for_best_model="wer",

    greater_is_better=False,

    push_to_hub=False,

)

#Along with our model, dataset, data collator, and compute_metrics function, we can send the training arguments to the Trainer:

from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(

    args=training_args,

    model=model,

    train_dataset=train_dataset,

    eval_dataset=test_dataset,

    data_collator=data_collator,

    compute_metrics=compute_metrics,

    tokenizer=processor.feature_extractor,

)
#Training

#To launch training, simply execute:

trainer.train()
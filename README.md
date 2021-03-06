# NLU-Benchmark : Natural Language Understanding Benchmark

[Korean(한국어)](./README_KOR.md)

**NLU-Benchmark** support natural language understanding experimental code for Close Domain chatbot or Task-Oriented artificial intelligence assistant system. Experiments provided by **NLU-Benchmark** include:

* Intent Classification : classify intent of the utterance or sentence
* Entity Recognition : extract entity information from utterance or sentence

---

![](figs/overview.png)

---

**NLU-Benchmark** is implemented based on Huggingface and PyTorch Lightning, and supports experiments with various language models.

## Dependencies
* torch==1.9.0
* pytorch-lightning==1.4.1
* transformers>=4.7.0
* pandas
* sentencepiece
* scikit-learn
* seqeval

## Usage
### 1. Download Dataset
**NLU-Benchmark** supports the following datasets.

| dataset name | language   | train | dev   | test  | link   |           
| :----------- | :--------: | :---: | :---: | :---: | :----: |
| ATIS         | `en`       | ✔     | ✔    | ✔     | [LINK](https://github.com/yuanxiaosc/BERT-for-Sequence-Labeling-and-Text-Classification/tree/master/data/atis_Intent_Detection_and_Slot_Filling)  |
| SNIPS        | `en`       | ✔     | ✔    | ✔     | [LINK](https://github.com/yuanxiaosc/BERT-for-Sequence-Labeling-and-Text-Classification/tree/master/data/snips_Intent_Detection_and_Slot_Filling)  |

* After downloading datasets from above links, create a folder having each dataset name under the `data` folder and save downloaded datasets there, e.g., in case of ATIS dataset, save `train`, `valid`, `test` folders in `data/atis`.

* The following files are stored in the folders downloaded through above links.
    * `seq.in` : utterance sentences are stored.
    * `label` : intents about each sentences in `seq.in` are saved.
    * `seq.out` : sequence labels for each sentences in `seq.in` are saved. theses sequence labels follow [BIO Format](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)).

### 2. Pre-processing Dataset
To perform intent classification and entity recognition experiments via **NLU-Benchmark**, you should build label dictionaries about each dataset.

If you run the following command, `intent.vocab`. a dictionary about intent labels, and `entity.vocab`, a dictionary about entity labels, are created.

* `--data_name` : dataset name to create label dictionary, e.g., `atis`, `snips`

```bash
python data/build_label_vocabs.py --data_name {$DATA_NAME}
```

### 3. Intent Classification : Training and Evaluation
Run intent classification experiment by the following command:

* `--model_type` :  type of model, e.g., `bert`, `albert`
* `--model_name` : checkpoint name of language model you wanna run experiment, e.g., `bert-base-cased`, `albert-base-v2`
* `--data_name` : dataset name to use training and evaluating, e.g., `atis`, `snips`
* `--do_train` : set training mode
* `--do_eval` : set evaluating mode
* `--num_train_epochs` : number of epochs in training
* `--gpu_id` : GPU id to be used when performing experiment
* `--batch_size` : batch size at training

```bash
python run_intent_classification.py --model_type {$MODEL_TYPE} \
                                    --model_name {$MODEL_NAME} \
                                    --data_name {$DATA_NAME} \
                                    --do_train \
                                    --do_eval \
                                    --num_train_epochs 20 \
                                    --gpu_id 0 \
                                    --batch_size 32
```

### 4. Entity Recognition : Training and Evaluation
Run entity recognition experiment by the following command:

* `--model_type` : type of model, e.g., `bert`, `albert`
* `--model_name` : checkpoint name of language model you wanna run experiment, e.g., `bert-base-cased`, `albert-base-v2`
* `--data_name` : dataset name to use training and evaluating, e.g., `atis`, `snips`
* `--do_train` : set training mode
* `--do_eval` : set evaluating mode
* `--num_train_epochs` : number of epochs in training
* `--gpu_id` : GPU id to be used when performing experiment
* `--batch_size` : batch size at training

```bash
python run_entity_recognition.py --model_type {$MODEL_TYPE} \
                                 --model_name {$MODEL_NAME} \
                                 --data_name {$DATA_NAME} \
                                 --do_train \
                                 --do_eval \
                                 --num_train_epochs 20 \
                                 --gpu_id 0 \
                                 --batch_size 32
```

## Result
Hyper parameters for experiments are as follows:

| Hyper Parameter    | Value                | 
| :----------------: | :------------------: |
| `max_seq_length`   | 128                  |
| `batch_size`       | 32                   |
| `learning_rate`    | 5e-5                 |
| `num_train_epochs` | 20                   |

* If you look at the experimental results below, you can see that performance of the `large` size models have low performances. I think this problem can be solved by increasing the epoch and batch size.

### 1. ATIS Dataset : Intent Classification

| Model Type    | Model Name                                                                                        | Accuracy(%) |
| :------------ | :------------------------------------------------------------------------------------------------ | :---------: |
| bert          | [bert-base-cased](https://huggingface.co/bert-base-cased)                                         | 97.31       |
|               | [bert-base-uncased](https://huggingface.co/bert-base-uncased)                                     | 96.84       |
|               | [bert-large-cased](https://huggingface.co/bert-large-cased)                                       | 97.64       |
|               | [bert-large-uncased](https://huggingface.co/bert-large-uncased)                                   | **97.87**   |
| distilbert    | [distilbert-base-cased](https://huggingface.co/distilbert-base-cased)                             | 96.97       |
|               | [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)                         | **97.31**   |
| roberta       | [roberta-base](https://huggingface.co/roberta-base)                                               | 95.74       |
|               | [roberta-large](https://huggingface.co/roberta-large)                                             | **97.64**   |
| albert        | [albert-base-v2](https://huggingface.co/albert-base-v2)                                           | **97.08**   |
|               | [albert-large-v2](https://huggingface.co/albert-large-v2)                                         | 83.76       |
| xlnet         | [xlnet-base-cased](https://huggingface.co/xlnet-base-cased)                                       | **97.76**   |
|               | [xlnet-large-cased](https://huggingface.co/xlnet-large-cased)                                     | 70.77       |
| electra       | [google/electra-small-discriminator](https://huggingface.co/google/electra-small-discriminator)   | 88.91       |
|               | [google/electra-base-discriminator](https://huggingface.co/google/electra-base-discriminator)     | **95.29**   |
|               | [google/electra-large-discriminator](https://huggingface.co/google/electra-large-discriminator)   | 70.77       |

### 2. ATIS Dataset : Entity Recognition

| Model Type    | Model Name                                                                                        | Precision(%) | Recall(%)    | F1 Score(%)  |
| :------------ | :------------------------------------------------------------------------------------------------ | :----------: | :----------: | :----------: |
| bert          | [bert-base-cased](https://huggingface.co/bert-base-cased)                                         | 94.91        | 95.48        | 95.20        |
|               | [bert-base-uncased](https://huggingface.co/bert-base-uncased)                                     | 94.89        | 95.69        | 95.29        |
|               | [bert-large-cased](https://huggingface.co/bert-large-cased)                                       | 95.36        | 95.73        | 95.54        |
|               | [bert-large-uncased](https://huggingface.co/bert-large-uncased)                                   | **95.53**    | **95.80**    | **95.67**    |
| distilbert    | [distilbert-base-cased](https://huggingface.co/distilbert-base-cased)                             | **94.68**    | **95.38**    | **95.03**    |
|               | [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)                         | 94.38        | **95.38**    | 94.88        |
| roberta       | [roberta-base](https://huggingface.co/roberta-base)                                               | 94.78        | 95.48        | 95.13        |
|               | [roberta-large](https://huggingface.co/roberta-large)                                             | **94.99**    | **95.62**    | **95.31**    |
| albert        | [albert-base-v2](https://huggingface.co/albert-base-v2)                                           | **94.74**    | **95.27**    | **95.00**    |
|               | [albert-large-v2](https://huggingface.co/albert-large-v2)                                         | 94.24        | 95.24        | 94.74        |
| xlnet         | [xlnet-base-cased](https://huggingface.co/xlnet-base-cased)                                       | **95.02**    | **95.52**    | **95.27**    |
|               | [xlnet-large-cased](https://huggingface.co/xlnet-large-cased)                                     | 94.86        | 95.13        | 95.00        |
| electra       | [google/electra-small-discriminator](https://huggingface.co/google/electra-small-discriminator)   | 83.70        | 83.99        | 83.84        |
|               | [google/electra-base-discriminator](https://huggingface.co/google/electra-base-discriminator)     | **94.72**    | **95.62**    | **95.17**    |
|               | [google/electra-large-discriminator](https://huggingface.co/google/electra-large-discriminator)   | 94.45        | 95.45        | 94.95        |

### 3. SNIPS Dataset : Intent Classification

| Model Type    | Model Name                                                                                        | Accuracy(%) |
| :------------ | :------------------------------------------------------------------------------------------------ | :---------: |
| bert          | [bert-base-cased](https://huggingface.co/bert-base-cased)                                         | 97.14       |
|               | [bert-base-uncased](https://huggingface.co/bert-base-uncased)                                     | 97.71       |
|               | [bert-large-cased](https://huggingface.co/bert-large-cased)                                       | **98.14**   |
|               | [bert-large-uncased](https://huggingface.co/bert-large-uncased)                                   | 98.00       |
| distilbert    | [distilbert-base-cased](https://huggingface.co/distilbert-base-cased)                             | **98.14**   |
|               | [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)                         | 97.57       |
| roberta       | [roberta-base](https://huggingface.co/roberta-base)                                               | 97.71       |
|               | [roberta-large](https://huggingface.co/roberta-large)                                             | **98.14**   |
| albert        | [albert-base-v2](https://huggingface.co/albert-base-v2)                                           | **98.00**   |
|               | [albert-large-v2](https://huggingface.co/albert-large-v2)                                         | 96.28       |
| xlnet         | [xlnet-base-cased](https://huggingface.co/xlnet-base-cased)                                       | **97.14**   |
|               | [xlnet-large-cased](https://huggingface.co/xlnet-large-cased)                                     | 17.57       |
| electra       | [google/electra-small-discriminator](https://huggingface.co/google/electra-small-discriminator)   | **97.57**   |
|               | [google/electra-base-discriminator](https://huggingface.co/google/electra-base-discriminator)     | 96.99       |
|               | [google/electra-large-discriminator](https://huggingface.co/google/electra-large-discriminator)   | 11.42       |

### 4. SNIPS Dataset : Entity Recognition

| Model Type    | Model Name                                                                                        | Precision(%) | Recall(%)    | F1 Score(%)  |
| :------------ | :------------------------------------------------------------------------------------------------ | :----------: | :----------: | :----------: |
| bert          | [bert-base-cased](https://huggingface.co/bert-base-cased)                                         | 93.27        | 94.58        | 93.92        |
|               | [bert-base-uncased](https://huggingface.co/bert-base-uncased)                                     | 94.32        | 95.58        | 94.95        |
|               | [bert-large-cased](https://huggingface.co/bert-large-cased)                                       | 93.97        | 95.86        | 94.91        |
|               | [bert-large-uncased](https://huggingface.co/bert-large-uncased)                                   | **95.08**    | **96.14**    | **95.61**    |
| distilbert    | [distilbert-base-cased](https://huggingface.co/distilbert-base-cased)                             | 93.74        | 95.41        | **94.57**    |
|               | [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)                         | **93.59**    | **95.47**    | 94.52        |
| roberta       | [roberta-base](https://huggingface.co/roberta-base)                                               | 90.63        | 93.01        | 91.81        |
|               | [roberta-large](https://huggingface.co/roberta-large)                                             | **93.19**    | **94.86**    | **94.01**    |
| albert        | [albert-base-v2](https://huggingface.co/albert-base-v2)                                           | **93.57**    | **95.25**    | **94.40**    |
|               | [albert-large-v2](https://huggingface.co/albert-large-v2)                                         | 91.58        | 93.57        | 92.56        |
| xlnet         | [xlnet-base-cased](https://huggingface.co/xlnet-base-cased)                                       | 93.74        | 95.41        | 94.57        |
|               | [xlnet-large-cased](https://huggingface.co/xlnet-large-cased)                                     | **94.85**    | **95.75**    | **95.30**    |
| electra       | [google/electra-small-discriminator](https://huggingface.co/google/electra-small-discriminator)   | 92.31        | 94.58        | 93.43        |
|               | [google/electra-base-discriminator](https://huggingface.co/google/electra-base-discriminator)     | **93.36**    | **95.19**    | **94.27**    |
|               | [google/electra-large-discriminator](https://huggingface.co/google/electra-large-discriminator)   | 92.83        | 94.86        | 93.83        |

## TODO List
- [x] README (Version. EN)
- [ ] add CRF layers

## References
- [yuanxiaosc/BERT-for-Sequence-Labeling-and-Text-Classification](https://github.com/yuanxiaosc/BERT-for-Sequence-Labeling-and-Text-Classification)
- [huggingface/transformers](https://github.com/huggingface/transformers)
- [monologg/KoBERT-NER](https://github.com/monologg/KoBERT-NER)

---

If you have any additional questions, please register an issue in this repository or contact sehunhu5247@gmail.com.
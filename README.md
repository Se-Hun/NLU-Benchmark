# NLU-Benchmark : Natural Language Understanding Benchmark

**NLU-Benchmark**는 Close Domain 챗봇 또는 Task-Oriented 인공지능 비서 시스템을 위한 자연어 이해 실험 코드를 제공합니다. **NLU-Benchmark**에서 제공하는 실험은 다음과 같습니다.

* 의도 분류(Intent Classification) : 발화문이 어떠한 의도를 담고 있는지 분류
* 엔터티 인식(Entity Recognition) : 발화문에서의 엔터티 정보 추출

---

![](figs/overview.png)

---

**NLU-Benchmark**는 Hugginface와 PyTorch Lightning을 기반으로 코드가 작성되었으며, 다양한 언어 모델들에 대한 실험을 지원합니다.

## Dependencies
* torch==1.9.0
* pytorch-lightning==1.4.1
* transformers>=4.7.0
* pandas
* sentencepiece
* scikit-learn
* seqeval

## Usage
### 1. 데이터셋 다운로드
**NLU-Benchmark**에서는 다음과 같은 데이터셋들을 지원합니다.

| 데이터셋 이름 | 언어   | train | dev   | test  | 링크   |           
| :---------- | :---: | :---: | :---: | :---: | :---: |
| ATIS        | `en`  | ✔     | ✔    | ✔     | [LINK](https://github.com/yuanxiaosc/BERT-for-Sequence-Labeling-and-Text-Classification/tree/master/data/atis_Intent_Detection_and_Slot_Filling)  |
| SNIPS       | `en`  | ✔     | ✔    | ✔     | [LINK](https://github.com/yuanxiaosc/BERT-for-Sequence-Labeling-and-Text-Classification/tree/master/data/snips_Intent_Detection_and_Slot_Filling)  |

* 데이터셋들을 링크에서 다운로드 받은 후에, `data` 폴더 밑에 각 데이터셋의 이름별 폴더를 생성한 후에 그곳에 다운로드 받은 데이터셋들을 저장합니다. ex) ATIS 데이터셋의 경우, `data/atis` 폴더에 `train`, `valid`, `test`를 저장

* 링크를 통해 다운로드 받은 폴더들에는 다음과 같은 파일들이 저장되어 있습니다.
    * `seq.in` : 발화 문장들이 저장되어 있습니다.
    * `label` : `seq.in`에서의 각 문장들에 대한 의도(intent)가 저장되어 있습니다.
    * `seq.out` : `seq.in`에서의 각 문장들에 대한 Sequence Label들이 저장되어 있습니다. 저장되어 있는 Sequence Label은 [BIO Format](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging))을 따릅니다.

### 2. 데이터셋 전처리
**NLU-Benchmark**를 통해 의도 분류 실험과 엔터티 인식 실험을 수행하려면 각 데이터셋의 레이블 사전이 필요합니다. 

다음과 같은 명령어를 수행하면 의도 레이블들에 대한 사전인 `intent.vocab`과 엔터티 레이블들에 대한 사전인 `entity.vocab`이 생성됩니다.

* `--data_name` : 레이블 사전을 생성하고자 하는 데이터셋 이름 ex) `atis`, `snips`

```bash
python data/build_label_vocabs.py --data_name {$DATA_NAME}
```

### 3. Intent Classification : Training and Evaluation
다음과 같은 명령을 실행하여 의도 분류 실험을 수행합니다.

```bash
python run_intent_classification.py --model_type {$MODEL_TYPE} \
                                    --model_name {$MODEL_NAME} \
                                    --data_name {$DATA_NAME} \
                                    --do_train \
                                    --do_eval \
                                    --num_train_epochs 10 \
                                    --gpu_id 0 \
                                    --batch_size 32
```

### 4. Entity Recognition : Training and Evaluation
다음과 같은 명령을 실행하여 엔터티 인식을 수행합니다.

```bash
python run_entity_recognition.py --model_type {$MODEL_TYPE} \
                                 --model_name {$MODEL_NAME} \
                                 --data_name {$DATA_NAME} \
                                 --do_train \
                                 --do_eval \
                                 --num_train_epochs 10 \
                                 --gpu_id 0 \
                                 --batch_size 32
```

## Result
실험에 적용한 하이퍼파라미터는 다음과 같습니다.

| Hyper Parameter    | Value                | 
| :----------------: | :------------------: |
| `max_seq_length`   | 128                  |
| `batch_size`       | 32                   |
| `learning_rate`    | 5e-5                 |
| `num_train_epochs` | 20                   |

### 1. Intent Classification

| Model Type    | Model Name                                                                                        | Accuracy(%) |
| :------------ | :------------------------------------------------------------------------------------------------ | :---------: |
| bert          | [bert-base-cased](https://huggingface.co/bert-base-cased)                                         |             |
|               | [bert-base-uncased](https://huggingface.co/bert-base-uncased)                                     |             |
|               | [bert-large-cased](https://huggingface.co/bert-large-cased)                                       |             |
|               | [bert-large-uncased](https://huggingface.co/bert-large-uncased)                                   |             |
| distilbert    | [distilbert-base-cased](https://huggingface.co/distilbert-base-cased)                             |             |
|               | [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)                         |             |
| roberta       | [roberta-base](https://huggingface.co/roberta-base)                                               |             |
|               | [roberta-large](https://huggingface.co/roberta-large)                                             |             |
| albert        | [albert-base-v2](https://huggingface.co/albert-base-v2)                                           |             |
|               | [albert-large-v2](https://huggingface.co/albert-large-v2)                                         |             |
| xlnet         | [xlnet-base-cased](https://huggingface.co/xlnet-base-cased)                                       |             |
|               | [xlnet-large-cased](https://huggingface.co/xlnet-large-cased)                                     |             |
| xlm           | [xlm-mlm-en-2048](https://huggingface.co/xlm-mlm-en-2048)                                         |             |
| electra       | [google/electra-small-discriminator](https://huggingface.co/google/electra-small-discriminator)   |             |
|               | [google/electra-base-discriminator](https://huggingface.co/google/electra-base-discriminator)     |             |
|               | [google/electra-large-discriminator](https://huggingface.co/google/electra-large-discriminator)   |             |

### 2. Entity Recognition

| Model Type    | Model Name                                                                                        | Precision(%) | Recall(%)    | F1 Score(%)  |
| :------------ | :------------------------------------------------------------------------------------------------ | :----------: | :----------: | :----------: |
| bert          | [bert-base-cased](https://huggingface.co/bert-base-cased)                                         |              |              |              |
|               | [bert-base-uncased](https://huggingface.co/bert-base-uncased)                                     |              |              |              |
|               | [bert-large-cased](https://huggingface.co/bert-large-cased)                                       |              |              |              |
|               | [bert-large-uncased](https://huggingface.co/bert-large-uncased)                                   |              |              |              |
| distilbert    | [distilbert-base-cased](https://huggingface.co/distilbert-base-cased)                             |              |              |              |
|               | [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)                         |              |              |              |
| roberta       | [roberta-base](https://huggingface.co/roberta-base)                                               |              |              |              |
|               | [roberta-large](https://huggingface.co/roberta-large)                                             |              |              |              |
| albert        | [albert-base-v2](https://huggingface.co/albert-base-v2)                                           |              |              |              |
|               | [albert-large-v2](https://huggingface.co/albert-large-v2)                                         |              |              |              |
| xlnet         | [xlnet-base-cased](https://huggingface.co/xlnet-base-cased)                                       |              |              |              |
|               | [xlnet-large-cased](https://huggingface.co/xlnet-large-cased)                                     |              |              |              |
| xlm           | [xlm-mlm-en-2048](https://huggingface.co/xlm-mlm-en-2048)                                         |              |              |              |
| electra       | [google/electra-small-discriminator](https://huggingface.co/google/electra-small-discriminator)   |              |              |              |
|               | [google/electra-base-discriminator](https://huggingface.co/google/electra-base-discriminator)     |              |              |              |
|               | [google/electra-large-discriminator](https://huggingface.co/google/electra-large-discriminator)   |              |              |              |

## TODO List
- [ ] README (Version. EN)

## References
- [yuanxiaosc/BERT-for-Sequence-Labeling-and-Text-Classification](https://github.com/yuanxiaosc/BERT-for-Sequence-Labeling-and-Text-Classification)
- [huggingface/transformers](https://github.com/huggingface/transformers)
- [monologg/KoBERT-NER](https://github.com/monologg/KoBERT-NER)
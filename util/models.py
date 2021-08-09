from transformers import (
    BertForSequenceClassification,
    DistilBertForSequenceClassification,
    RobertaForSequenceClassification,
    AlbertForSequenceClassification,
    XLNetForSequenceClassification,
    XLMForSequenceClassification,
    ElectraForSequenceClassification
)
from transformers import (
    BertForTokenClassification,
    DistilBertForTokenClassification,
    RobertaForTokenClassification,
    AlbertForTokenClassification,
    XLNetForTokenClassification,
    XLMForTokenClassification,
    ElectraForTokenClassification
)
from transformers import (
    BertTokenizer,
    DistilBertTokenizer,
    RobertaTokenizer,
    AlbertTokenizer,
    XLNetTokenizer,
    XLMTokenizer,
    ElectraTokenizer
)


INTENT_MODEL_CLASSES = {
    "bert": BertForSequenceClassification,
    "distilbert": DistilBertForSequenceClassification,
    "roberta": RobertaForSequenceClassification,
    "albert": AlbertForSequenceClassification,
    "xlnet": XLNetForSequenceClassification,
    "xlm": XLMForSequenceClassification,
    "electra": ElectraForSequenceClassification
}
ENTITY_MODEL_CLASSES = {
    "bert": BertForTokenClassification,
    "distilbert": DistilBertForTokenClassification,
    "roberta": RobertaForTokenClassification,
    "albert": AlbertForTokenClassification,
    "xlnet": XLNetForTokenClassification,
    "xlm": XLMForTokenClassification,
    "electra": ElectraForTokenClassification
}
TOKENIZER_CLASSES = {
    "bert": BertTokenizer,
    "distilbert": DistilBertTokenizer,
    "roberta": RobertaTokenizer,
    "albert": AlbertTokenizer,
    "xlnet": XLNetTokenizer,
    "xlm": XLMTokenizer,
    "electra": ElectraTokenizer
}


def get_intent_model(model_type, model_name, num_intents):
    model_class = INTENT_MODEL_CLASSES[model_type]
    model = model_class.from_pretrained(model_name, num_labels=num_intents)

    return model

def get_entity_model(model_type, model_name, num_entities):
    model_class = ENTITY_MODEL_CLASSES[model_type]
    model = model_class.from_pretrained(model_name, num_labels=num_entities)

    return model

def get_tokenizer(model_type, model_name, do_lower_case):
    tokenizer_class = TOKENIZER_CLASSES[model_type]
    tokenizer = tokenizer_class.from_pretrained(model_name, do_lower_case=do_lower_case)
    return tokenizer

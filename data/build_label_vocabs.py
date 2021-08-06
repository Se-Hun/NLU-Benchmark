import os
import argparse

from util.utils import load_txt


def build_intent_vocab(data_dir):
    # load txt files
    train_intent_data = load_txt(os.path.join(data_dir, "train", "label"))
    valid_intent_data = load_txt(os.path.join(data_dir, "valid", "label"))
    test_intent_data = load_txt(os.path.join(data_dir, "test", "label"))

    # collect intent labels
    intent_vocab = set(train_intent_data) | set(valid_intent_data) | set(test_intent_data)
    intent_vocab = list(sorted(list(intent_vocab)))

    # dump intent vocab file
    intent_vocab_file_path = os.path.join(data_dir, "intent.vocab")
    with open(intent_vocab_file_path, 'w', encoding='utf-8') as f:
        for idx, label in enumerate(intent_vocab):
            print("{}\t{}".format(label, idx), file=f)
    print("Intent Vocab file is dumped at ", intent_vocab_file_path)

def build_entity_vocab(data_dir):
    # load txt files
    train_entity_data = load_txt(os.path.join(data_dir, "train", "seq.out"))
    valid_entity_data = load_txt(os.path.join(data_dir, "valid", "seq.out"))
    test_entity_data = load_txt(os.path.join(data_dir, "test", "seq.out"))

    # collect entity labels
    all_train_entity_data = [line.split() for line in train_entity_data]
    all_valid_entity_data = [line.split() for line in valid_entity_data]
    all_test_entity_data = [line.split() for line in test_entity_data]

    all_train_entity_data = sum(all_train_entity_data, [])
    all_valid_entity_data = sum(all_valid_entity_data, [])
    all_test_entity_data = sum(all_test_entity_data, [])

    entity_vocab = set(all_train_entity_data) | set(all_valid_entity_data) | set(all_test_entity_data)
    entity_vocab = list(entity_vocab)

    # In order to process the missing BIO format
    for bio_entity in entity_vocab:
        if "-" in bio_entity:
            b_or_i, entity = bio_entity.split("-")
            if b_or_i == "B" and "I-"+entity not in entity_vocab:
                entity_vocab.append("I-" + entity)
            if b_or_i == "I" and "B-"+entity not in entity_vocab:
                entity_vocab.append("B-" + entity)
    entity_vocab = sorted(entity_vocab)

    # dump entity vocab file
    entity_vocab_file_path = os.path.join(data_dir, "entity.vocab")
    with open(entity_vocab_file_path, 'w', encoding='utf-8') as f:
        for idx, label in enumerate(entity_vocab):
            print("{}\t{}".format(label, idx), file=f)
    print("Entity Vocab file is dumped at ", entity_vocab_file_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default="atis", type=str,
                        help="Type data name for pre-processing.")
    args = parser.parse_args()

    data_dir = os.path.join("./", args.data_name.lower())

    build_intent_vocab(data_dir)
    build_entity_vocab(data_dir)

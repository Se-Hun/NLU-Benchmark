import os
import json
import platform
import argparse
from glob import glob

from seqeval import metrics
from transformers import AdamW, get_linear_schedule_with_warmup

import torch
from torch.nn import CrossEntropyLoss
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from dataset import EntityRecognitionDataModule
from util import ENTITY_MODEL_CLASSES, get_entity_model


class EntityRecognizer(LightningModule):
    def __init__(
        self,
        data_name: str,
        model_type: str,
        model_name: str,
        entity_map: dict,
        learning_rate: float = 5e-5
    ):
        """
            `EntityRecognizer` run entity recognition.

            Args:
                data_name (str): dataset name.
                model_type (str): model type, e.g., `bert`
                model_name (str): model name, e.g., `bert-base-cased`
                num_entities (int): number of entities.
                learning_rate (float, optional): learning rate for optimizer. defaults to 5e-5.
            """
        super().__init__()
        self.save_hyperparameters()

        # save entity_map
        self.entity_map = entity_map
        num_entities = len(entity_map)

        # load model
        model = get_entity_model(model_type, model_name, num_entities)
        self.model = model

    def forward(self, x):
        return self.model(**x)

    def training_step(self, batch, batch_idx):
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "labels": batch[3]
        }

        if self.hparams.model_type in ["distilbert", "roberta"]:
            del inputs["token_type_ids"]  # Distilbert don't use segment_ids.

        outputs = self(inputs)

        loss = outputs[0]
        result = {"loss": loss}
        return result

    def validation_step(self, batch, batch_idx):
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "labels": batch[3]
        }

        if self.hparams.model_type in ["distilbert", "roberta"]:
            del inputs["token_type_ids"]  # Distilbert don't use segment_ids.

        outputs = self(inputs)

        loss, logits = outputs[:2]
        preds = torch.argmax(logits, dim=2)

        result = {"val_loss": loss, "preds": preds, "labels": batch[3]}
        return result

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()

        out_label_list = [[] for _ in range(labels.shape[0])]
        preds_list = [[] for _ in range(preds.shape[0])]

        pad_token_label_id = CrossEntropyLoss().ignore_index

        label_map = {i: label for label, i in self.entity_map.items()}

        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                if labels[i, j] != pad_token_label_id:
                    out_label_list[i].append(label_map[labels[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        val_f1 = metrics.f1_score(out_label_list, preds_list)

        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_f1", val_f1, prog_bar=True)
        return {"val_loss": val_loss, "val_f1": val_f1}

    def test_step(self, batch, batch_idx):
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "labels": batch[3]
        }

        if self.hparams.model_type in ["distilbert", "roberta"]:
            del inputs["token_type_ids"]  # Distilbert don't use segment_ids.

        outputs = self(inputs)

        _, logits = outputs[:2]
        preds = torch.argmax(logits, dim=2)

        result = {"preds": preds, "labels": batch[3]}
        return result

    def test_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()

        out_label_list = [[] for _ in range(labels.shape[0])]
        preds_list = [[] for _ in range(preds.shape[0])]

        pad_token_label_id = CrossEntropyLoss().ignore_index

        label_map = {i: label for label, i in self.entity_map.items()}

        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                if labels[i, j] != pad_token_label_id:
                    out_label_list[i].append(label_map[labels[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        test_precision = metrics.precision_score(out_label_list, preds_list)
        test_recall = metrics.recall_score(out_label_list, preds_list)
        test_f1 = metrics.f1_score(out_label_list, preds_list)

        results = {"precision": test_precision, "recall": test_recall, "f1": test_f1}

        result_file = os.path.join(self.trainer.checkpoint_callback.dirpath, "result.json")
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
            print("Result file is dumped at ", result_file)

        print(json.dumps(results, indent=4))
        return results

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=1e-8)

        t_total = len(self.train_dataloader()) // self.trainer.max_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)

        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=5e-5)
        return parser

def main():
    # Argument Setting -------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_type", type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(ENTITY_MODEL_CLASSES.keys()))
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name of pre-trained model. you can search at huggingface models.")
    parser.add_argument("--data_name", type=str, required=True,
                        help="Data name selected in the list: " + ", ".join(
                            EntityRecognitionDataModule.get_supported_dataset()))

    # Other parameters
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")

    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")

    parser.add_argument("--num_train_epochs", default=10, type=int, help="Epochs at train time.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("--gpu_id", type=str, default="0",
                        help="Gpu device id.")

    parser.add_argument("--seed", default=42, type=int, help="Seed number")

    parser = Trainer.add_argparse_args(parser)
    parser = EntityRecognizer.add_model_specific_args(parser)
    args = parser.parse_args()
    # ------------------------------------------------------------------------------------------------------------------

    # set seed
    seed_everything(args.seed)

    # load DataModule
    args.model_type = args.model_type.lower()
    args.model_name = args.model_name.lower()
    args.data_name = args.data_name.lower()
    dm = EntityRecognitionDataModule(args.data_name, args.model_type, args.model_name,
                                     args.max_seq_length, args.batch_size)
    dm.prepare_data()
    entity_map = dm.entity_vocab

    # load Callbacks and Loggers
    model_dir = './model/{}/{}/{}'.format(args.data_name, "entity", args.model_name.replace("/", "-"))
    model_checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        dirpath=model_dir,
        filename='{epoch:02d}-{val_f1:.3f}'
    )

    tensorboard_logger = TensorBoardLogger(
        save_dir=model_dir, name=''  # <-- if experiment name(=name) is empty, sub directory is not made.
    )

    # load Trainer
    trainer = Trainer(
        gpus=args.gpu_id if platform.system() != 'Windows' else 1,  # <-- for dev. pc
        logger=tensorboard_logger,
        callbacks=[model_checkpoint_callback],
        max_epochs=args.num_train_epochs
    )

    # Do train !
    if args.do_train:
        model = EntityRecognizer(args.data_name, args.model_type, args.model_name, entity_map)
        dm.setup('fit')
        trainer.fit(model, dm)

    # Do eval !
    if args.do_eval:
        model_files = glob(os.path.join(trainer.checkpoint_callback.dirpath, "*.ckpt"))
        best_fn = sorted(model_files, key=lambda fn: fn.split("=")[-1])[0]
        print("[Evaluation] Best Model File name is {}".format(best_fn))

        model = EntityRecognizer.load_from_checkpoint(best_fn)
        dm.setup('test')
        trainer.test(model, datamodule=dm)

if __name__ == '__main__':
    main()
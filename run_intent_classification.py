import os
import json
import platform
import argparse
from glob import glob

from transformers import AdamW, get_linear_schedule_with_warmup

import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from dataset import IntentClassificationDataModule
from util import INTENT_MODEL_CLASSES, get_intent_model


class IntentClassifier(LightningModule):
    def __init__(
        self,
        data_name: str,
        model_type: str,
        model_name: str,
        num_intents: int,
        learning_rate: float=5e-5
    ):
        """
        `IntentClassifier` run intent classification.

        Args:
            data_name (str):
            model_type (str):
            model_name (str):
            num_intents (int):
            learning_rate (float, optional):
        """
        super().__init__()
        self.save_hyperparameters()

        # load model
        model = get_intent_model(model_type, model_name, num_intents)
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

        if self.hparams.model_type in ["distilbert"]:
            del inputs["token_type_ids"] # Distilbert don't use segment_ids.

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

        if self.hparams.model_type in ["distilbert"]:
            del inputs["token_type_ids"]  # Distilbert don't use segment_ids.

        outputs = self(inputs)

        loss, logits = outputs[:2]
        preds = torch.argmax(logits, dim=1)

        result = {"val_loss": loss, "preds": preds, "labels": batch[3]}
        return result

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        preds = torch.cat([x["preds"] for x in outputs])
        labels = torch.cat([x["labels"] for x in outputs])

        correct_count = torch.sum(labels == preds)
        val_acc = float(correct_count / len(labels))

        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_acc", val_acc, prog_bar=True)
        return {"val_loss": val_loss, "val_acc": val_acc}

    def test_step(self, batch, batch_idx):
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "labels": batch[3]
        }

        if self.hparams.model_type in ["distilbert"]:
            del inputs["token_type_ids"]  # Distilbert don't use segment_ids.

        outputs = self(inputs)

        _, logits = outputs[:2]
        preds = torch.argmax(logits, dim=1)

        result = {"preds": preds, "labels": batch[3]}
        return result

    def test_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs])
        labels = torch.cat([x["labels"] for x in outputs])

        correct_count = torch.sum(labels == preds)
        test_acc = float(correct_count / len(labels))

        results = {"accuracy": test_acc}

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
                        help="Model type selected in the list: " + ", ".join(INTENT_MODEL_CLASSES.keys()))
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name of pre-trained model. you can search at huggingface models.")
    parser.add_argument("--data_name", type=str, required=True,
                        help="Data name selected in the list: " + ", ".join(IntentClassificationDataModule.get_supported_dataset()))

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
    parser = IntentClassifier.add_model_specific_args(parser)
    args = parser.parse_args()
    # ------------------------------------------------------------------------------------------------------------------

    # set seed
    seed_everything(args.seed)

    # load DataModule
    args.model_type = args.model_type.lower()
    args.model_name = args.model_name.lower()
    args.data_name = args.data_name.lower()
    dm = IntentClassificationDataModule(args.data_name, args.model_type, args.model_name,
                                        args.max_seq_length, args.batch_size)
    dm.prepare_data()
    num_intents = dm.num_intents

    # load Callbacks and Loggers
    model_dir = './model/{}/{}/{}'.format(args.data_name, "intent", args.model_name.replace("/", "-"))
    model_checkpoint_callback = ModelCheckpoint(
        monitor='val_acc', # or 'val_loss'
        mode='max', # or 'min'
        dirpath=model_dir,
        filename='{epoch:02d}-{val_loss:.2f}'
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
        model = IntentClassifier(args.data_name, args.model_type, args.model_name, num_intents)
        dm.setup('fit')
        trainer.fit(model, dm)

    # Do eval !
    if args.do_eval:
        model_files = glob(os.path.join(trainer.checkpoint_callback.dirpath, "*.ckpt"))
        best_fn = sorted(model_files, key=lambda fn: fn.split("=")[-1])[0]
        print("[Evaluation] Best Model File name is {}".format(best_fn))

        model = IntentClassifier.load_from_checkpoint(best_fn)
        dm.setup('test')
        trainer.test(model, datamodule=dm)

if __name__ == '__main__':
    main()
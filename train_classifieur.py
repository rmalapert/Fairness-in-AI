from pathlib import Path
from typing import Any
import argparse
from torchvision.models import resnet18, ResNet18_Weights
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.nn.functional import cross_entropy
from torchmetrics.classification import BinaryConfusionMatrix, ConfusionMatrix
from torchvision.transforms.v2 import (
    Compose,
    Resize,
    CenterCrop,
    ToImage,
    ToDtype,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
)
from torchvision.datasets import ImageFolder
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import LightningModule, Trainer
from sklearn.metrics import balanced_accuracy_score, accuracy_score

import io
import shutil
import os
from pathlib import Path
import pandas as pd
from time import time
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir",
        type=str,
        required=True,
        help="The folder in which the log will be writen",
    )
    parser.add_argument(
        "--datadir",
        type=str,
        required=True,
        help="The folder has to follow the torchvision ImageFolder structure (one subfolder per partition : train, valid, test and for each one subfolder per class)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="The path to the metadata csv file",
    )
    parser.add_argument(
        "--weights_col",
        type=str,
        default="WEIGHTS",
        required=False,
        help="The name of the column of the csv file, that contains the weights per instance",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default="25",
        required=False,
        help="Max number of epochs",
    )
    parser.add_argument(
        "--csv_out",
        type=str,
        required=True,
        help="The path to the preds csv file",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=False,
        help="The path to ckpt model. Used only if the training is not activated",
    )
    parser.add_argument(
        "--preds_col",
        type=str,
        default="preds",
        required=False,
        help="The name of the column of the csv output file, that contains the predictions per instance",
    )
    parser.add_argument(
        "--train",
        type=str,
        default="True",
        required=False,
        help="(default True): If True the training is perfomed",
    )
    parser.add_argument(
        "--pred",
        type=str,
        default="True",
        required=False,
        help="(default True): If True the prediction is perfomed. If train is also set to True the argument ckpt_path is ignored",
    )
    opt = parser.parse_args()
    return opt


transforms_valid = Compose(
    [
        Resize((256, 256)),
        CenterCrop(224),
        ToImage(),
        ToDtype(
            torch.float32,
            scale=True,
        ),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

transforms_train = Compose(
    [
        Resize((256, 256)),
        RandomCrop(224),
        RandomHorizontalFlip(),
        ToImage(),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return balanced_accuracy_score(labels.detach().cpu(), outputs_idx.detach().cpu())


class ChestXRayClassifier(LightningModule):
    def __init__(self, pth_path=None, adamax=False, cosine=True, nb_classes=40) -> None:
        super().__init__()
        self.model = make_model(pth_path, nb_classes)
        self.adamax = adamax
        self.cosine = cosine
        self.nb_classes = nb_classes
        if nb_classes==2:
            self.cm = BinaryConfusionMatrix()
        else:
            self.cm = ConfusionMatrix(task="multiclass", num_classes=self.nb_classes)

    def plot_cm(self, logits, labels, name, batch_idx):
        self.cm(torch.argmax(logits, dim=1), labels)
        fig, ax = plt.subplots(figsize=(10, 10))
        if self.nb_classes==2:
            self.cm.plot(ax=ax, labels=["malade", "sain"])
        else:
            self.cm.plot(ax=ax)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)
        im = Compose([ToImage(), ToDtype(torch.float32, scale=True)])(Image.open(buf))
        self.logger.experiment.add_image(name, im, batch_idx)
        self.cm.reset()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        logits = self.forward(imgs)
        loss = cross_entropy(logits, labels)
        self.log("train_loss", loss)
        acc = accuracy(logits, labels)
        self.log("train_acc", acc)
        # Logging one confusion matrix per epoch is enough and avoids high memory usage.
        if batch_idx == 0:
            self.plot_cm(
                name="train_confusion_matrix",
                logits=logits,
                labels=labels,
                batch_idx=self.current_epoch,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        logits = self.forward(imgs)
        loss = cross_entropy(logits, labels)
        self.log("val_loss", loss)
        acc = accuracy(logits, labels)
        self.log("val_acc", acc)
        if batch_idx == 0:
            self.plot_cm(
                name="valid_confusion_matrix",
                logits=logits,
                labels=labels,
                batch_idx=self.current_epoch,
            )
        return loss

    def configure_optimizers(self) -> Any:
        if self.adamax:
            optimizer = torch.optim.Adamax(
                self.parameters(), lr=1e-3, weight_decay=0.001
            )
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        if self.cosine:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.trainer.max_epochs, 1e-7, -1 #, True
            )
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[4, 9, 15], gamma=0.2
            )
        return ([optimizer], [scheduler])


def make_model(pth_path=None, nb_classes=40, V1=False):
    if V1:
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, nb_classes)
    if pth_path is not None:
        model.load_state_dict(torch.load(pth_path, map_location="cpu"))
    return model


def get_weights(imgs, img_names, img_weights, weight=None):
    """
    Returns the list of weights sorted in the same order as the images in the dataloader
    args:
        imgs: dataloader ordered list of tuples (image path, truth)
        img_names: dataframe ordered list of images name
        img_weights: dataframe ordered list of images weights
    """
    sorted_weights = []
    if not (weight is None):
        for img_path, target in imgs:
            sorted_weights.append(weight[target])
    else:
        for img_path, _ in imgs:
            img_basename = Path(img_path).name
            for idx, val in enumerate(img_names):
                if val == img_basename:
                    sorted_weights.append(img_weights[idx])
                    break

    sorted_weights = torch.Tensor(sorted_weights)
    sorted_weights = sorted_weights.double()
    return sorted_weights


def get_class_weights(imgs):
    target = [t for (_, t) in imgs]
    train_class_sample_count = np.array(
        [len(np.where(target == t)[0]) for t in np.unique(target)]
    )
    return 1.0 / train_class_sample_count


def preds_todf(df, dataset, label_decoder, model, preds_col):
    num_workers = 0 if os.name == "nt" else os.cpu_count()
    print('num_workers set to :', num_workers)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device =='cuda':
      batch_size = 512
    else:
      batch_size=max(16,num_workers)
    model = model.to(device)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )
    idx = 0
    for _, (imgs, labels) in enumerate(tqdm(dataloader)):
        logits = model(imgs.to(device)).detach().cpu().numpy()
        for logit, label in zip (logits, labels):
            img_name = Path(dataset.imgs[idx][0]).name
            pred = logit.argmax()
            df.loc[df["Image Index"] == img_name, preds_col] = label_decoder[pred]
            df.loc[df["Image Index"] == img_name, "labels"] = label_decoder[label.item()]
            for l in range(len(logit)):
                df.loc[df["Image Index"] == img_name, f"{preds_col}_logit{l}"] = logit[l]
            idx+=1
    return df

# def preds_todf_old(df, dataset, label_decoder, model, preds_col):
#     for idx in range(len(dataset.imgs)):
#         img_name = dataset.imgs[idx][0].split("/")[-1]
#         img, label = dataset.__getitem__(idx)
#         logit = model(img.unsqueeze(0))
#         pred = logit.max(1)[1].detach().cpu().item()
#         df.loc[df["Image Index"] == img_name, preds_col] = label_decoder[pred]
#         df.loc[df["Image Index"] == img_name, "labels"] = label_decoder[label]
#         for l in range(len(logit)):
#             df.loc[df["Image Index"] == img_name, f"{preds_col}_logit{l}"] = logit[0][l].detach().cpu().item()
#     return df

# def pred_classifier_old(
#     datadir: str, ckpt_path: str, csv_in: str, csv_out: str, preds_col: str = "preds"
# ):
#     """Make prediction with the model on the data
#     Args:
#         - datadir, str : The folder has to follow the torchvision ImageFolder structure
#                          (one subfolder per partition : train, valid, test and for each one subfolder per class)
#         - ckpt_path, str : The path to ckpt model
#         - csv_in, str: The path to the metadata csv input file
#         - csv_out, str: The path to the metadata csv output file, with "label" and preds_col added
#         - preds_col, str : The name of the column of the csv output file, that contains the predictions per instance
#     """
#     train_datadir = f"{datadir}/train/"
#     valid_datadir = f"{datadir}/valid/"
#     train_dataset = ImageFolder(train_datadir, transform=transforms_valid)
#     label_encoder = train_dataset.class_to_idx
#     label_decoder = {}
#     for k, v in label_encoder.items():
#         label_decoder[v] = k
#     model = ChestXRayClassifier(adamax=True, cosine=True, nb_classes=len(label_encoder))
#     model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["state_dict"])
#     model.eval()
#     val_dataset = ImageFolder(valid_datadir, transform=transforms_valid)
#     df = pd.read_csv(csv_in)
#     df[preds_col] = None
#     print("Start prediction on train dataset")
#     t = time()
#     df = preds_todf_old(df, train_dataset, label_decoder, model, preds_col)
#     print(f"Predictions done in {time()-t}")
#     print("Start prediction on validation dataset")
#     df = preds_todf_old(df, val_dataset, label_decoder, model, preds_col)
#     print(f"Predictions done in {time()-t}")
#     df.to_csv(csv_out, index=False)
#     print(balanced_accuracy_score(df.labels, df[preds_col]))
#     print(accuracy_score(df.labels, df[preds_col]))


def pred_classifier(
    datadir: str, ckpt_path: str, csv_in: str, csv_out: str, preds_col: str = "preds"
):
    """Make prediction with the model on the data
    Args:
        - datadir, str : The folder has to follow the torchvision ImageFolder structure
                         (one subfolder per partition : train, valid, test and for each one subfolder per class)
        - ckpt_path, str : The path to ckpt model
        - csv_in, str: The path to the metadata csv input file
        - csv_out, str: The path to the metadata csv output file, with "label" and preds_col added
        - preds_col, str : The name of the column of the csv output file, that contains the predictions per instance
    """
    train_datadir = f"{datadir}/train/"
    valid_datadir = f"{datadir}/valid/"
    train_dataset = ImageFolder(train_datadir, transform=transforms_valid)
    label_encoder = train_dataset.class_to_idx
    label_decoder = {}
    for k, v in label_encoder.items():
        label_decoder[v] = k
    
    model = ChestXRayClassifier(adamax=True, cosine=True, nb_classes=len(label_encoder))
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["state_dict"])
    model.eval()
    val_dataset = ImageFolder(valid_datadir, transform=transforms_valid)
    df = pd.read_csv(csv_in)
    df[preds_col] = None
    print("Start prediction on train dataset")
    t = time()
    df = preds_todf(df, train_dataset, label_decoder, model, preds_col)
    print(f"Predictions done in {time()-t}")
    print("Start prediction on validation dataset")
    df = preds_todf(df, val_dataset, label_decoder, model, preds_col)
    print(f"Predictions done in {time()-t}")
    df.to_csv(csv_out, index=False)
    print("Global (train+validation) balanced accuracy without weigths", balanced_accuracy_score(df.labels, df[preds_col]))
    print("Global (train+validation) accuracy without weigths",accuracy_score(df.labels, df[preds_col]))


def train_classifier(
    logdir: str,
    datadir: str,
    csv: str,
    weights_col: str = "WEIGHTS",
    max_epochs: int = 25,
):
    """Train an image classifier using Resnet18 and outputs predictions for all partitions (train, valid)
    Args:
        - logdir, str : The folder in which the log will be writen
        - datadir, str : The folder has to follow the torchvision ImageFolder structure
                         (one subfolder per partition : train, valid, test and for each one subfolder per class)
        - csv, str: The path to the metadata csv file
        - weights_col, str : The name of the column of the csv file, that contains the weights per instance
        - max_epochs,, int : Max number of epochs
    Return:
        A tuple (ckpt_path, ckpt_score), path of the checkpoint and its score

    """
    t = time()
    train_datadir = f"{datadir}/train/"
    valid_datadir = f"{datadir}/valid/"
    df = pd.read_csv(csv)
    print(csv, f"{logdir}/csv_in_{weights_col}.csv")
    os.makedirs(logdir, exist_ok=True)
    shutil.copy(csv, f"{logdir}/csv_in_{weights_col}.csv")

    logger = TensorBoardLogger(f"{logdir}/", name="tensorboard_logs")

    cb_ckpt_best = ModelCheckpoint(
        dirpath=f"{logdir}/",
        monitor="val_loss",
        filename="best-val-loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )
    cb_es = EarlyStopping(monitor="val_loss", patience=10)
    callbacks = [cb_ckpt_best, cb_es]
    ### Execution set up
    num_workers = 0 if os.name == "nt" else os.cpu_count()
    print('num_workers set to :', num_workers)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device =='cuda':
      batch_size = 512
      log_n_steps = 1
    else:
      batch_size=max(16,num_workers)
      log_n_steps = int(512/batch_size)
    ### Define the trainer
    trainer = Trainer(
      logger=logger,
      log_every_n_steps=log_n_steps,
      callbacks=callbacks, 
      max_epochs=max_epochs, 
      accelerator="auto", 
      devices='auto')
    train_dataset = ImageFolder(train_datadir, transform=transforms_train)
    label_encoder = train_dataset.class_to_idx
    val_dataset = ImageFolder(valid_datadir, transform=transforms_valid)
    train_weights = get_weights(train_dataset.imgs, df["Image Index"], df[weights_col])
    #valid_weights = get_weights(val_dataset.imgs, df["Image Index"], df[weights_col])

    print(trainer.accelerator)
    print('num_workers set to :', num_workers)
    print('batch_size set to :', batch_size)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=WeightedRandomSampler(train_weights, len(train_weights)),
        num_workers=num_workers)
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    model = ChestXRayClassifier(adamax=True, cosine=True, nb_classes=len(label_encoder))
    print(f"Start training")
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )
    print(f"End of training {time()-t}")
    t = time()
    return cb_ckpt_best.best_model_path, cb_ckpt_best.best_model_score.item()


if __name__ == "__main__":
    t = time()
    args = parse_opt()
    print(args)
    if args.train == "True":
        ckpt_path, ckpt_score = train_classifier(
            logdir=args.logdir,
            datadir=args.datadir,
            csv=args.csv,
            weights_col=args.weights_col,
            max_epochs=args.max_epochs,
        )
        print(ckpt_path, ckpt_score)
        if args.pred == "True":
            pred_classifier(
                datadir=args.datadir,
                ckpt_path=ckpt_path,
                csv_in=args.csv,
                csv_out=args.csv_out,
                preds_col=args.preds_col,
            )
    elif args.pred == "True":
        pred_classifier(
            datadir=args.datadir,
            ckpt_path=args.ckpt_path,
            csv_in=args.csv,
            csv_out=args.csv_out,
            preds_col=args.preds_col,
        )

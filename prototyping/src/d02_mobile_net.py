import os
import datetime
from copy import deepcopy
from pathlib import Path
import logging
from typing import Tuple, Any

from PIL import Image
import numpy as np
from torchvision.datasets import ImageFolder
import torch
from torch import nn

# from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Subset, SubsetRandomSampler, DataLoader
import torchvision.transforms.v2 as transforms
import mlflow
from mlflow.models.signature import infer_signature

# import random
from ignite.engine import create_supervised_trainer
from ignite.handlers import FastaiLRFinder
from scipy.ndimage import gaussian_filter1d
from torch.optim import Optimizer
from torchvision.models import mobilenet_v2, MobileNetV2

logging.getLogger("mlflow.utils.requirements_utils").setLevel(logging.ERROR)
logging.getLogger("mlflow.store.model_registry.abstract_store").setLevel(logging.ERROR)


class CustomImageFolder(ImageFolder):
    def __init__(self, image_path, transform=None):
        super().__init__(image_path, transform, loader=self.custom_pil_loader)

    def get_balanced_idx(self) -> tuple[list, list]:
        only_test_idx = range(0, len(self), 5)
        only_test_idx = list(only_test_idx)

        train_idx = range(len(self))
        train_idx = set(train_idx) - set(only_test_idx)
        train_idx = list(train_idx)

        #####
        class_indices = {cls: [] for cls in range(len(self.classes))}
        for idx, label in enumerate(self.targets):
            class_indices[label].append(idx)

        max_len = max(len(idxs) for idxs in class_indices.values())

        balanced_subsets = []
        for cls, idxs in class_indices.items():
            repeat_factor = max_len // len(idxs)
            remainder = max_len % len(idxs)
            balanced_subsets.append(
                Subset(self, idxs * repeat_factor + idxs[:remainder])
            )
        #####
        test_idx = list(range(len(self)))  # all samples are used for tests

        return train_idx, test_idx

    def __getitem__(
        self, index: int
    ) -> tuple[Any, Any]:  # overwriting original method to output target as float32
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, np.float32(target)

    @staticmethod
    def custom_pil_loader(path: str) -> Image.Image:
        with open(path, "rb") as f:
            img = Image.open(f)
            img.load()

            if img.mode in ("1", "L"):
                return img
            else:
                return img.convert("RGB")


class Trainer:
    # device = None
    tracking = None
    batch_size_props = {
        "img_shape": (0, 0, 0),
        "batch_size": 2,
        "memory_limited": False,
    }
    # loss_fn = None

    def __init__(
        self,
        device: str,
        tracking: bool = False,
        experiment: str = "default",
        tracking_uri: str = "http://127.0.0.1:8080",
    ):
        self.tracking = tracking

        if self.tracking:
            mlflow.set_tracking_uri(uri=tracking_uri)
            mlflow.set_experiment(experiment)

        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()  # SqueezedBCEWithLogitsLoss()

    def _train(
        self, dataloader: DataLoader, model: MobileNetV2, optimizer: Optimizer
    ) -> None:
        model.train()

        for X, y in dataloader:
            X = X.to(self.device)
            y = y.to(self.device)
            torch.manual_seed(42)

            pred = model(X)
            loss = self.loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    def _test(
        self, dataloader: DataLoader, model: MobileNetV2
    ) -> tuple[float, int, int]:
        errors = 0
        conf_errors = 0
        loss = 0

        model.eval()
        with torch.no_grad():
            for X, y in dataloader:
                X = X.to(self.device)
                y = y.to(self.device)
                pred = model(X)

                loss += self.loss_fn(pred, y).item()
                pred = torch.sigmoid(pred.flatten())
                errors += torch.sum(torch.round(pred) != y).item()
                pred[(pred < 0.75) & (pred > 0.25)] = -1
                conf_errors += torch.sum(torch.round(pred) != y).item()

        return loss / len(dataloader), errors, conf_errors  # type: ignore

    def _tracking_start(
        self,
        run_name: str,
        model: MobileNetV2,
        lr: float,
        total_iters: int,
        weight_decay: float,
        img_channels: int,
    ) -> None:
        mlflow.end_run()  # in case previous training was interrupted
        # cur_run_name = run_name + '_' + \
        # str(len(mlflow.search_runs(filter_string=f'tag.mlflow.runName LIKE "{run_name}_%"')) + 1)

        mlflow.start_run(run_name=run_name)
        mlflow.set_tag("layers", model.layers)
        mlflow.log_params(
            {
                "lr": lr,
                "total_iters": total_iters,
                "weight_decay": weight_decay,
                "img_channels": img_channels,
            }
        )

    def _tracking_end(self, model: MobileNetV2, train_loader: DataLoader) -> None:
        run = mlflow.active_run()
        if run is not None:
            signature = infer_signature(
                train_loader.dataset[0][0].numpy(), train_loader.dataset[0][1]
            )
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                registered_model_name=run.info.run_name,
                signature=signature,
            )
            mlflow.end_run()

    def _test_and_log(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        model: MobileNetV2,
        e: int,
    ) -> int:
        test_loss, test_errors, test_errors_conf = self._test(test_loader, model)
        train_loss, train_errors, train_errors_conf = self._test(train_loader, model)

        if self.tracking:
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "train_errors": train_errors,
                    "test_errors": test_errors,
                    "train_errors_conf": train_errors_conf,
                    "test_errors_conf": test_errors_conf,
                },
                step=e,
                synchronous=False,
            )
        else:
            print(
                f"{datetime.datetime.now()}\t{e}\tErrors: {train_errors}-{test_errors}\tConf Errors: {train_errors_conf}-{test_errors_conf}\tLoss: {train_loss:.6f}-{test_loss:.6f}"
            )

        return test_errors_conf + train_errors_conf

    def train_model(
        self,
        folder: str,
        epochs: int = 100,
        weight_decay: float = 1e-7,
    ) -> MobileNetV2:
        run_name = f"{folder}"
        run_name = run_name.replace("/", "_")

        img_shape = self.get_img_shape(f"{folder}/assembly")

        loaders = self.get_dataloaders(folder, img_shape)
        torch.manual_seed(42)
        # model = NN(img_shape[0]).to(self.device)
        model = mobilenet_v2(pretrained=True).to(self.device)
        model.classifier = nn.Sequential(nn.Linear(1280, len(loaders["train"].dataset.classes)), nn.LogSoftmax(1))  # type: ignore

        # optimizer = torch.optim.AdamW(model.parameters(), weight_decay=weight_decay, fused=True)
        optimizer = torch.optim.Adadelta(model.parameters(), weight_decay=weight_decay)

        # lr = self.find_lr(model, optimizer, loaders['lr'], loss_fn)
        lr = -1
        scheduler = -1
        # scheduler = OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(loaders['train']), epochs=epochs, pct_start=0.5, div_factor=50)

        if self.tracking:
            self._tracking_start(
                run_name, model, lr, epochs, weight_decay, img_shape[0]
            )

        print("Training...")
        e = 0
        errors = float("inf")

        while e < epochs and errors > 0:
            e += 1
            """
            if e == total_iters:
                print(f'Unsuccessful learning with final {errors} errors')
                break
            """
            self._train(loaders["train"], model, optimizer)

            errors = self._test_and_log(loaders["train"], loaders["test"], model, e)

        if errors > 0:
            print(f"Unsuccessful training with final {errors} errors")
        else:
            print(f"Training finished successfully in {e} epochs")

        self._tracking_end(model, loaders["train"])

        if self.device == "cuda":
            torch.cuda.empty_cache()

        return model

    # @staticmethod
    def get_dataloaders(self, folder: str, img_shape: tuple) -> dict[str, DataLoader]:
        normalize_arg = [0.5] * img_shape[0]
        find_bs_transforms = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(normalize_arg, normalize_arg),
                # transforms.CenterCrop(size=(img_shape[1] * 2, img_shape[2] * 2)),
            ]
        )

        train_transforms = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.ColorJitter(brightness=(0.8, 1.2)),
                transforms.Normalize(normalize_arg, normalize_arg),
                transforms.RandomRotation(degrees=25, interpolation=transforms.InterpolationMode.BILINEAR),  # type: ignore
                # transforms.RandomPerspective(distortion_scale=0.2),
                # transforms.RandomCrop(
                #  size=(img_shape[1] * 2, img_shape[2] * 2),
                #  padding=(img_shape[2], img_shape[1]),
                # ),
            ]
        )

        test_transforms = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(normalize_arg, normalize_arg),
            ]
        )

        find_bs_ds = CustomImageFolder(folder, transform=find_bs_transforms)
        train_ds = CustomImageFolder(folder, transform=train_transforms)
        test_ds = CustomImageFolder(folder, transform=test_transforms)

        train_idx, test_idx = train_ds.get_balanced_idx()

        test_dataset = Subset(test_ds, test_idx)

        batch_size = self.get_batch_size(img_shape, find_bs_ds, train_idx)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            sampler=SubsetRandomSampler(train_idx, torch.Generator().manual_seed(42)),
        )

        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True
        )

        return {"train": train_loader, "test": test_loader}

    @staticmethod
    def get_img_shape(img_folder: str) -> tuple[int, int, int]:
        img_name = os.listdir(img_folder)[0]

        with open(f"{img_folder}/{img_name}", "rb") as f:
            img = Image.open(f)
            img.load()

            channels = 3

            if img.mode in ("1", "L"):
                channels = 1

        return channels, img.height, img.width

    def get_batch_size(self, img_shape: tuple, ds: ImageFolder, train_idx: list) -> int:
        if self.batch_size_props["img_shape"] != img_shape:
            self.batch_size_props = {
                "img_shape": img_shape,
                "batch_size": 2,
                "memory_limited": False,
            }

        if (
            not self.batch_size_props["memory_limited"]
            and self.batch_size_props["batch_size"] < len(train_idx) // 2
        ):
            model = mobilenet_v2(pretrained=True).to(self.device)
            model.classifier = nn.Sequential(
                nn.Linear(1280, len(ds.classes)), nn.LogSoftmax(1)
            )
            optimizer = torch.optim.Adadelta(model.parameters())
            batch_size = 0

            if self.batch_size_props["batch_size"] == 2:
                test_res = self.test_batch_size(ds, train_idx, 2, model, optimizer)
                if not test_res[0]:
                    raise MemoryError(
                        "Not enough memory. Learning process failed with batch_size 2."
                    )
                batch_size_2_mem = test_res[1]

                test_res = self.test_batch_size(ds, train_idx, 3, model, optimizer)
                if not test_res[0]:
                    batch_size = 2
                    self.batch_size_props["memory_limited"] = True
                else:
                    self.batch_size_props["batch_size"] = 3
                    batch_size = torch.cuda.mem_get_info()[0] // (
                        batch_size_2_mem - test_res[1]
                    )

            if not self.batch_size_props["memory_limited"]:
                if self.batch_size_props["batch_size"] != 3:
                    batch_size = self.batch_size_props["batch_size"] + 1

                if batch_size > len(train_idx) // 2:
                    batch_size = len(train_idx) // 2

                if batch_size > self.batch_size_props["batch_size"]:
                    batch_size = self.pick_batch_size(
                        batch_size, ds, train_idx, model, optimizer
                    )

                    print("Selected batch_size", batch_size)
                    del model, optimizer
                    torch.cuda.empty_cache()
        else:
            if self.batch_size_props["batch_size"] > len(train_idx) // 2:
                batch_size = len(train_idx) // 2
            else:
                batch_size = self.batch_size_props["batch_size"]
            print("Reused batch_size", batch_size)

        if batch_size > len(train_idx) // 2:
            print("ATATA!!!!!!!")

        return batch_size

    def pick_batch_size(
        self,
        batch_size: int,
        ds: ImageFolder,
        train_idx: list,
        model: MobileNetV2,
        optimizer: Optimizer,
    ) -> int:
        smallest_failed_batch_size = float("inf")
        checked_batch_size = self.batch_size_props["batch_size"]
        memory_limit = False

        while True:
            test_res = self.test_batch_size(ds, train_idx, batch_size, model, optimizer)

            if test_res[0]:
                prev_checked_batch_size = checked_batch_size
                checked_batch_size = batch_size
                batch_size = (
                    checked_batch_size + (batch_size - prev_checked_batch_size) // 2
                )
                if batch_size == checked_batch_size:
                    batch_size += 1
            else:
                smallest_failed_batch_size = batch_size
                batch_size = checked_batch_size + (batch_size - checked_batch_size) // 2
                memory_limit = True

            if (
                batch_size == smallest_failed_batch_size
                or batch_size == checked_batch_size
            ):
                batch_size = checked_batch_size
                break

        if batch_size > self.batch_size_props["batch_size"]:
            self.batch_size_props["batch_size"] = batch_size
            if memory_limit:
                self.batch_size_props["memory_limited"] = True

        return batch_size

    def test_batch_size(
        self,
        ds: ImageFolder,
        train_idx: list,
        batch_size: int,
        model: MobileNetV2,
        optimizer: Optimizer,
    ) -> tuple[bool, int]:
        res = True
        print("Testing batch_size", batch_size)

        try:
            dataloader = DataLoader(
                ds,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
                drop_last=True,
                sampler=SubsetRandomSampler(
                    train_idx[: batch_size * 2], torch.Generator().manual_seed(42)
                ),
            )
            for _ in range(3):
                self._train(dataloader, model, optimizer)
                if torch.cuda.mem_get_info()[0] == 0:
                    res = False
                    break
                self._test(dataloader, model)
                if torch.cuda.mem_get_info()[0] == 0:
                    res = False
                    break
        except RuntimeError:
            res = False
        if res:
            print("success")
        else:
            print("fail")

        free_mem = torch.cuda.mem_get_info()[0]
        print("Freemem:", free_mem)

        torch.cuda.empty_cache()

        return res, free_mem


class SqueezedBCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> float:
        # if target.dim() == 1:
        target = target.unsqueeze(1)
        return self.loss(input, target)

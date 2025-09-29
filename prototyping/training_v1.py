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

logging.getLogger("mlflow.utils.requirements_utils").setLevel(logging.ERROR)
logging.getLogger("mlflow.store.model_registry.abstract_store").setLevel(logging.ERROR)


class CustomImageFolder(ImageFolder):
    def __init__(self, image_path, target_class=None, transform=None):
        self.target_class = target_class
        super().__init__(image_path, transform, loader=self.custom_pil_loader)

    def find_classes(self, directory: str) -> tuple[list, dict]:
        classes = sorted(
            entry.name for entry in os.scandir(directory) if entry.is_dir()
        )
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        if self.target_class:
            class_to_idx = dict.fromkeys(classes, 0)
            class_to_idx[self.target_class] = 1
            class_to_idx["assembly"] = 1
        else:
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        return classes, class_to_idx

    def get_balanced_idx(self) -> tuple[list, list]:
        test_idx = range(0, len(self), 5)
        test_idx = list(test_idx)

        train_idx = range(len(self))
        train_idx = set(train_idx) - set(test_idx)
        train_idx = list(train_idx)

        target_in_train_idx = [i for i in train_idx if self.targets[i] == 1.0]
        target_in_train_num = len(target_in_train_idx)
        other_in_train_num = len(train_idx) - target_in_train_num

        add_idx = []
        i = 0

        while target_in_train_num + len(add_idx) < other_in_train_num:
            add_idx.append(target_in_train_idx[i])
            i += 1

            if i == target_in_train_num:
                i = 0

        train_idx += add_idx

        print(
            f"Initial target samples amount: {target_in_train_num}; non target samples amount: {other_in_train_num}; final amount: {len(train_idx)}"
        )

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


class NN(nn.Module):
    def __init__(self, img_channels: int):
        super().__init__()
        torch.manual_seed(42)

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=img_channels,
                out_channels=32,
                kernel_size=3,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            NN._conv_dw(32, 64, 1),
            NN._conv_dw(64, 128, 2),
            NN._conv_dw(128, 128, 1),
            NN._conv_dw(128, 256, 2),
            NN._conv_dw(256, 256, 1),
            NN._conv_dw(256, 512, 2),
            NN._conv_dw(512, 512, 1),
            NN._conv_dw(512, 512, 1),
            NN._conv_dw(512, 512, 1),
            NN._conv_dw(512, 512, 1),
            NN._conv_dw(512, 512, 1),
            NN._conv_dw(512, 1024, 2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        for l in self.layers:
            print(x.shape)
            x = l(x)
        return(x)
        """
        return self.layers(x)

    @staticmethod
    def _conv_dw(in_ch: int, out_ch: int, s: int) -> nn.Sequential:
        return nn.Sequential(
            # dw
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=in_ch,
                kernel_size=3,
                padding=1,
                stride=s,
                groups=in_ch,
                bias=False,
            ),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            # pw
            nn.Conv2d(
                in_channels=in_ch, out_channels=out_ch, kernel_size=1, bias=False
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


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
        self.loss_fn = SqueezedBCEWithLogitsLoss()

    def _train(self, dataloader: DataLoader, model: NN, optimizer: Optimizer) -> None:
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

    def _test(self, dataloader: DataLoader, model: NN) -> tuple[float, int, int]:
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

        return loss / len(dataloader), errors, conf_errors

    def _tracking_start(
        self,
        run_name: str,
        model: NN,
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

    def _tracking_end(self, model: NN, train_loader: DataLoader) -> None:
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
        self, train_loader: DataLoader, test_loader: DataLoader, model: NN, e: int
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
        target_class: str,
        epochs: int = 100,
        weight_decay: float = 1e-7,
    ) -> NN:
        run_name = f"{folder}_{target_class}"
        run_name = run_name.replace("/", "_")

        img_shape = self.get_img_shape(f"{folder}/{target_class}")

        torch.manual_seed(42)
        model = NN(img_shape[0]).to(self.device)

        loaders = self.get_dataloaders(folder, target_class, img_shape)
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
            self._train(loaders["train"], model, optimizer, scheduler)

            errors = self._test_and_log(loaders["train"], loaders["test"], model, e)

        if errors > 0:
            print(f"Unsuccessful training with final {errors} errors")
        else:
            print(f"Training finished successfully in {e} epochs")

        self._tracking_end(model, loaders["train"])

        if self.device == "cuda":
            torch.cuda.empty_cache()

        return model

    def find_lr(
        self,
        model: NN,
        optimizer: Optimizer,
        dataloader: DataLoader,
        loss_fn: nn.Module,
    ) -> float:
        print("Strating LRFinder")

        lr_trainer = create_supervised_trainer(
            model, optimizer, loss_fn, device=self.device
        )

        lr_finder = CustomLRFinder()
        to_save = {"optimizer": optimizer, "mode;": model}
        with lr_finder.attach(
            lr_trainer, start_lr=1e-8, num_iter=60, to_save=to_save, diverge_th=2.0
        ) as trainer_with_lr_finder:
            trainer_with_lr_finder.run(dataloader)

        # print(lr_finder.get_results())

        lr_finder.plot(skip_start=0, skip_end=0)
        print("LR:", lr_finder.lr_suggestion())

        if self.device == "cuda":
            torch.cuda.empty_cache()

        return lr_finder.lr_suggestion()

    # @staticmethod
    def get_dataloaders(self, folder: str, target_class: str, img_shape: tuple) -> dict:
        normalize_arg = [0.5] * img_shape[0]
        find_bs_transforms = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(normalize_arg, normalize_arg),
            ]
        )

        train_transforms = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.ColorJitter(brightness=(0.8, 1.2)),
                transforms.Normalize(normalize_arg, normalize_arg),
                transforms.RandomRotation(degrees=25),  # type: ignore
                transforms.RandomPerspective(distortion_scale=0.2),
            ]
        )

        test_transforms = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(normalize_arg, normalize_arg),
            ]
        )

        find_bs_ds = CustomImageFolder(
            folder, target_class=target_class, transform=find_bs_transforms
        )
        train_ds = CustomImageFolder(
            folder, target_class=target_class, transform=train_transforms
        )
        test_ds = CustomImageFolder(
            folder, target_class=target_class, transform=test_transforms
        )

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
            model = NN(img_shape[0]).to(self.device)
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
        model: NN,
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
        model: NN,
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
                self._train(dataloader, model, optimizer, scheduler=None)
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


class CustomLRFinder(FastaiLRFinder):
    def lr_suggestion(self) -> float:
        """
        Returns:
            Learning rate at the minimum numerical gradient
            (ignoring the increasing part of the curve)
        """
        if not self._history:
            raise RuntimeError(
                "learning rate finder didn't run yet so lr_suggestion can't be returned"
            )

        self._history["loss"] = gaussian_filter1d(
            self._history["loss"], sigma=2
        ).tolist()

        loss = self._history["loss"]
        min_loss_idx = torch.tensor(loss).argmin()

        # Ignore the increasing part of the curve
        decreasing_losses = self._history["loss"][: int(min_loss_idx.item()) + 1]

        while (
            len(decreasing_losses) < 3 and len(self._history["loss"]) > 3
        ):  # removing beginning of lr-loss curve for situations where min loss is in the beginning
            self._history["loss"] = self._history["loss"][1:]
            self._history["lr"] = self._history["lr"][1:]

            loss = self._history["loss"]
            min_loss_idx = torch.tensor(loss).argmin()
            decreasing_losses = self._history["loss"][: int(min_loss_idx.item()) + 1]

        if len(decreasing_losses) < 3:
            raise RuntimeError(
                "FastaiLRFinder got unexpected curve shape, the curve should be somehow U-shaped, "
                "please decrease start_lr or increase end_lr to resolve this issue."
            )
        losses = torch.tensor(decreasing_losses)
        grads = torch.tensor(
            [0.5 * (losses[i + 1] - losses[i - 1]) for i in range(1, len(losses) - 1)]
        )
        min_grad_idx = grads.argmin() + 1
        return self._history["lr"][int(min_grad_idx)]

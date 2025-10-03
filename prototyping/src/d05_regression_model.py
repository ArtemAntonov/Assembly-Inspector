import os
import datetime
from copy import deepcopy
from pathlib import Path
import logging
from typing import Tuple, Any

from PIL import Image
import numpy as np
from torchvision.datasets import ImageFolder, DatasetFolder
import torch
from torch import nn
from numpy.typing import NDArray

# from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Subset, SubsetRandomSampler, DataLoader, random_split
import torchvision.transforms.v2 as transforms
import mlflow
from mlflow.models.signature import infer_signature
from torchtune.datasets import ConcatDataset

# import random
from ignite.engine import create_supervised_trainer
from ignite.handlers import FastaiLRFinder
from scipy.ndimage import gaussian_filter1d
from torch.optim import Optimizer
from sklearn.utils.class_weight import compute_class_weight
from torch.profiler import profile, ProfilerActivity, record_function
from contextlib import nullcontext

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
            # class_to_idx["assembly"] = 1  # test for better results############################?>?>??????????????????????
        else:
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        return classes, class_to_idx

    def get_balanced_idx(self) -> tuple[list, list]:
        only_test_idx = range(0, len(self), 5)
        only_test_idx = list(only_test_idx)

        train_idx = range(len(self))
        train_idx = set(train_idx) - set(only_test_idx)
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


from torchvision.models import mobilenet_v3_large, MobileNetV2
from torchvision.models import squeezenet1_1, shufflenet_v2_x1_5
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models import efficientnet_b0


class NN(nn.Module):
    def __init__(self, img_channels: int):
        super().__init__()

        self.layers = mobilenet_v3_large(
            weights="MobileNet_V3_Large_Weights.IMAGENET1K_V1"
        )

        if img_channels == 1:
            first_conv = self.layers.features[0][0]  # This is the first Conv2d

            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )

            # If pretrained, average weights across RGB channels
            with torch.no_grad():
                new_conv.weight[:] = first_conv.weight.mean(dim=1, keepdim=True)

            # Replace in model
            self.layers.features[0][0] = new_conv
            # print(self.layers)
        self.layers.classifier[3] = nn.Sequential(
            nn.Linear(1280, 1024), nn.ReLU(), nn.Linear(1024, 1)
        )
        # self.layers.classifier[3] = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 1))

        self.layers = squeezenet1_1(weights="SqueezeNet1_1_Weights.IMAGENET1K_V1")

        self.layers.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Conv2d(512, 1000, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1000, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 1),
            # nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            # nn.Flatten(),
            # nn.Linear(1000, 500),
            # nn.ReLU(),
            # nn.Linear(500, 1),
            # nn.Softplus(),
            # nn.Conv2d(1000, 1, kernel_size=(1, 1), stride=(1, 1)),
            # nn.AdaptiveAvgPool2d((1, 1))
        )

        self.layers = efficientnet_b0(weights="EfficientNet_B0_Weights.IMAGENET1K_V1")
        self.layers.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True), nn.Linear(1280, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.layers(x)
        return res


class Trainer:
    BATCH_SIZE_LIMIT = 16
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
        # self.loss_fn = nn.MSELoss()
        self.loss_fn = nn.L1Loss()

    @staticmethod
    def trace_handler(p: torch.profiler.profile):
        sort_by_keyword = "self_" + "cuda" + "_time_total"
        output = p.key_averages().table(sort_by=sort_by_keyword, row_limit=10)
        print(output)
        # p.export_chrome_trace("./profiler_traces/trace_" + str(p.step_num) + ".json")
        # p.export_memory_timeline("./profiler_traces/trace_mem" + str(p.step_num) + ".json")

    def _train(
        self,
        dataloader: DataLoader,
        model: NN,
        optimizer: Optimizer,
        enable_profiler: bool,
    ) -> None:
        model.train()
        activities = [ProfilerActivity.CUDA]
        with (
            profile(
                activities=activities,
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                on_trace_ready=self.trace_handler,
            )
            if enable_profiler
            else nullcontext()
        ) as p:

            for X, y in dataloader:
                X = X.to(self.device)
                y = y.to(self.device).float()

                pred = model(X).view(-1)

                loss = self.loss_fn(pred, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if p is not None:
                    p.step()

    def _test(self, dataloader: DataLoader, model: NN) -> tuple[Any | float, int, int]:
        errors = 0
        conf_errors = 0
        loss = 0
        total_samples = 0

        model.eval()
        with torch.no_grad():
            total_samples += len(dataloader.dataset)

            for X, y in dataloader:
                X = X.to(self.device)
                y = y.to(self.device).float()
                pred = model(X).view(-1)

                loss += self.loss_fn(pred, y).item()
                errors += torch.sum(torch.round(pred) != y).item()
                conf_errors += torch.sum(
                    (torch.round(pred) != y)
                    | ((torch.frac(pred) > 0.25) & (torch.frac(pred) < 0.75))
                ).item()

        return loss / total_samples, int(errors), int(conf_errors)

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
                "batch_size": self.batch_size_props["batch_size"],
            }
        )

    def _tracking_end(self, model: NN, train_loader: DataLoader) -> None:
        run = mlflow.active_run()
        if run is not None:
            signature = infer_signature(train_loader.dataset[0][0].numpy(), 1.5)
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                registered_model_name=run.info.run_name,
                signature=signature,
            )
            mlflow.end_run()

    def _test_and_log(self, loaders: dict, model: NN, e: int) -> int:
        test_loss, test_errors, conf_test_errors = self._test(loaders["test"], model)
        train_loss, train_errors, conf_train_errors = self._test(
            loaders["train"], model
        )
        assembly_loss, assembly_errors, conf_assembly_errors = self._test(
            loaders["assembly"], model
        )

        if self.tracking:
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "assembly_loss": assembly_loss,
                    "train_errors": train_errors,
                    "test_errors": test_errors,
                    "assembly_errors": assembly_errors,
                    "conf_train_errors": conf_train_errors,
                    "conf_test_errors": conf_test_errors,
                    "conf_assembly_errors": conf_assembly_errors,
                },
                step=e,
                synchronous=False,
            )
        else:
            print(
                f"{datetime.datetime.now()}\t{e}\tErrors: {train_errors}-{test_errors}-{assembly_errors}\tLoss: {train_loss:.6f}-{test_loss:.6f}-{assembly_loss:.6f}"
            )

        return test_errors + train_errors

    def train_model(
        self,
        folder: str,
        part: str,
        num_in_assembly: int,
        epochs: int = 300,
        weight_decay: float = 1e-7,
        model: NN | None = None,
        enable_profiler: bool = False,
    ) -> NN:
        if enable_profiler:
            torch.profiler._utils._init_for_cuda_graphs()
        run_name = f"{folder}_{part}_counter"
        run_name = run_name.replace("/", "_")

        img_shape = self.get_img_shape(f"{folder}/{part}")

        if model is None:
            model = NN(img_shape[0]).to(self.device)
        else:
            print("Training existing model")

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        loaders, weights = self.get_dataloaders(
            f"{folder}/{part}", img_shape, num_in_assembly, part
        )
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
            self._train(loaders["train"], model, optimizer, enable_profiler and e == 1)
            errors = self._test_and_log(loaders, model, e)

        if errors > 0:
            print(f"Unsuccessful training with final {errors} errors")
        else:
            print(f"Training finished successfully in {e} epochs")

        self._tracking_end(model, loaders["train"])

        if self.device == "cuda":
            torch.cuda.empty_cache()

        return model

    # @staticmethod
    def get_dataloaders(
        self, folder: str, img_shape: tuple, num_in_assembly: int, part: str
    ) -> tuple[dict, NDArray]:
        normalize_arg = [0.5] * img_shape[0]
        train_transforms = [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.ColorJitter(brightness=(0.8, 1.2)),
            transforms.Normalize(normalize_arg, normalize_arg),
            # transforms.RandomCrop(size=(img_shape[1]*2, img_shape[2]*2), padding=(img_shape[2], img_shape[1]), padding_mode='edge')
        ]
        test_transforms = [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(normalize_arg, normalize_arg),
        ]

        if img_shape[0] == 1:
            train_transforms = [
                transforms.Grayscale(num_output_channels=1),
            ] + train_transforms
            test_transforms = [
                transforms.Grayscale(num_output_channels=1),
            ] + test_transforms
            #transforms.extend([????????????????

        train_transforms += [
            transforms.RandomRotation(degrees=25, interpolation=transforms.InterpolationMode.BILINEAR),  # type: ignore
            transforms.RandomPerspective(distortion_scale=0.2),
        ]

        train_transforms = transforms.Compose(train_transforms)
        test_transforms = transforms.Compose(test_transforms)

        ds = ImageFolder(root=f"{folder}", transform=img_transforms)

        idx_to_class = {v: int(k) for k, v in ds.class_to_idx.items()}

        for i in range(len(ds)):
            ds.samples[i] = (ds.samples[i][0], idx_to_class[ds.samples[i][1]])

        folders = str.split(folder, "/")
        ds_src = ImageFolder(
            root=f"./data/temp/{folders[-3]}/{folders[-2]}", transform=img_transforms
        )

        assembly_class_idx = ds_src.class_to_idx["assembly"]
        assembly_indices = [
            i for i, (_, y) in enumerate(ds_src.samples) if y == assembly_class_idx
        ]
        part_class_idx = ds_src.class_to_idx[part]
        src_train_indices = [
            i
            for i, (_, y) in enumerate(ds_src.samples)
            if y not in [assembly_class_idx, part_class_idx]
        ]

        for i in assembly_indices:
            ds_src.targets[i] = num_in_assembly
            path, _ = ds_src.samples[i]
            ds_src.samples[i] = (path, num_in_assembly)
        ds_assembly = Subset(ds_src, assembly_indices)

        for i in src_train_indices:
            ds_src.targets[i] = 0
            path, _ = ds_src.samples[i]
            ds_src.samples[i] = (path, 0)
        ds_src_train = Subset(ds_src, src_train_indices)

        batch_size = 16  # self.get_batch_size(img_shape, ds_lst[proper_ds_idx])

        generator = torch.Generator().manual_seed(42)
        # splits_ds = random_split(ds, [0.8, 0.2], generator)
        from sklearn.model_selection import train_test_split

        train_idx, test_idx = train_test_split(
            list(range(len(ds))), test_size=0.2, stratify=ds.targets, random_state=42
        )
        splits_ds = Subset(ds, train_idx), Subset(ds, test_idx)

        splits_src = random_split(ds_src_train, [0.8, 0.2], generator)
        splits_ass = random_split(ds_assembly, [0.2, 0.8], generator)

        train_ds = ConcatDataset([splits_ds[0], splits_src[0], splits_ass[0]])
        test_ds = ConcatDataset([splits_ds[1], splits_src[1]])

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            drop_last=True,  # for aspects with small amounts of samples
            shuffle=True,
        )

        test_loader = DataLoader(
            test_ds, batch_size=batch_size, num_workers=4, pin_memory=True
        )
        loader_assembly = DataLoader(
            splits_ass[1], batch_size=batch_size, num_workers=4, pin_memory=True
        )

        classes = np.array(range(num_in_assembly + 2))
        train_targets = []
        for _, targets in train_loader:
            train_targets.append(targets)
        train_targets = torch.cat(train_targets).numpy()

        weights = compute_class_weight(
            class_weight="balanced", classes=classes, y=train_targets
        )

        return {
            "train": train_loader,
            "test": test_loader,
            "assembly": loader_assembly,
        }, weights

    @staticmethod
    def get_img_shape(img_folder: str) -> tuple[int, int, int]:
        img_name = os.listdir(f"{img_folder}/1")[0]

        with open(f"{img_folder}/1/{img_name}", "rb") as f:
            img = Image.open(f)
            img.load()

            channels = 3

            if img.mode in ("1", "L"):
                channels = 1

        return channels, img.height, img.width

    def get_batch_size(self, img_shape: tuple, ds: ImageFolder) -> int:
        if self.batch_size_props["img_shape"] != img_shape:
            self.batch_size_props = {
                "img_shape": img_shape,
                "batch_size": 2,
                "memory_limited": False,
            }

            # if (
            # not self.batch_size_props["memory_limited"]
            # and self.batch_size_props["batch_size"] < len(train_idx) // 2
            # ):
            if self.device == "cuda":
                model = NN(img_shape[0]).to(self.device)
                optimizer = torch.optim.Adadelta(model.parameters())
                batch_size = 0

                if self.batch_size_props["batch_size"] == 2:
                    test_res = self.test_batch_size(ds, 2, model, optimizer)
                    if not test_res[0]:
                        raise MemoryError(
                            "Not enough memory. Learning process failed with batch_size 2."
                        )
                    batch_size_2_mem = test_res[1]

                    test_res = self.test_batch_size(ds, 3, model, optimizer)
                    if not test_res[0]:
                        batch_size = 2
                        self.batch_size_props["memory_limited"] = True
                    else:
                        self.batch_size_props["batch_size"] = 3
                        batch_size = torch.cuda.mem_get_info()[0] // (
                            batch_size_2_mem - test_res[1]
                        )  # memory limited
                        batch_size = min(
                            batch_size, 16
                        )  # batch size bigger than 256 can be too big

                if not self.batch_size_props["memory_limited"]:
                    if self.batch_size_props["batch_size"] != 3:
                        batch_size = self.batch_size_props["batch_size"] + 1

                    if batch_size > self.batch_size_props["batch_size"]:
                        batch_size = self.pick_batch_size(
                            batch_size, ds, model, optimizer
                        )

                        print("Selected batch_size", batch_size)
                        del model, optimizer
                        torch.cuda.empty_cache()
            else:
                batch_size = 16
                print("Selected batch_size", batch_size)
        else:
            batch_size = self.batch_size_props["batch_size"]
            print("Reused batch_size", batch_size)

        return batch_size

    def pick_batch_size(
        self,
        batch_size: int,
        ds: ImageFolder,
        model: NN,
        optimizer: Optimizer,
    ) -> int:
        smallest_failed_batch_size = float("inf")
        checked_batch_size = self.batch_size_props["batch_size"]
        memory_limit = False
        suggested_bs = batch_size

        while True:
            test_res = self.test_batch_size(ds, batch_size, model, optimizer)

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
                checked_batch_size == suggested_bs
                or batch_size == smallest_failed_batch_size
                or batch_size == checked_batch_size
                or (smallest_failed_batch_size - checked_batch_size)
                / smallest_failed_batch_size
                < 0.05
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
            )
            for _ in range(3):
                self._train(
                    [
                        dataloader,
                    ],
                    model,
                    optimizer,
                )
                if torch.cuda.mem_get_info()[0] == 0:
                    res = False
                    break
                self._test(
                    [
                        dataloader,
                    ],
                    model,
                )
                if torch.cuda.mem_get_info()[0] == 0:
                    res = False
                    break
        except RuntimeError:
            res = False
            raise
        if res:
            print("success")
        else:
            print("fail")

        free_mem = torch.cuda.mem_get_info()[0]
        print("Freemem:", free_mem)

        torch.cuda.empty_cache()

        return res, free_mem

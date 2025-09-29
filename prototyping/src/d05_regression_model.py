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

# from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Subset, SubsetRandomSampler, DataLoader, random_split
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
            #class_to_idx["assembly"] = 1  # test for better results############################?>?>??????????????????????
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

        self.layers = mobilenet_v3_large(weights="MobileNet_V3_Large_Weights.IMAGENET1K_V1")        
        
        if img_channels == 1:
            first_conv = self.layers.features[0][0]   # This is the first Conv2d
            
            new_conv = nn.Conv2d(
                in_channels=1, 
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            )

            # If pretrained, average weights across RGB channels
            with torch.no_grad():
                new_conv.weight[:] = first_conv.weight.mean(dim=1, keepdim=True)

            # Replace in model
            self.layers.features[0][0] = new_conv
            #print(self.layers)
        self.layers.classifier[3] = nn.Sequential(nn.Linear(1280, 1024), nn.ReLU(), nn.Linear(1024, 1))
        #self.layers.classifier[3] = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 1))
        
        self.layers = squeezenet1_1(weights="SqueezeNet1_1_Weights.IMAGENET1K_V1")
        
        self.layers.classifier = nn.Sequential(
                                nn.Dropout(p=0.5, inplace=False),
                                nn.Conv2d(512, 1000, kernel_size=(1, 1), stride=(1, 1)),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(1000, 1, kernel_size=(1, 1), stride=(1, 1)),
                                nn.AdaptiveAvgPool2d((1, 1)),
                                
                                #nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                #nn.Flatten(),
                                #nn.Linear(1000, 500),
                                #nn.ReLU(),
                                #nn.Linear(500, 1),
                                #nn.Softplus(),
                                
                                #nn.Conv2d(1000, 1, kernel_size=(1, 1), stride=(1, 1)),
                                #nn.AdaptiveAvgPool2d((1, 1))                              
                                
                                )
        self.layers.classifier = nn.Sequential(
                                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                nn.Flatten(),
                                nn.ReLU(),
                                nn.Linear(512, 256),
                                nn.ReLU(),
                                nn.Linear(256, 1),
                                
                                #nn.Conv2d(1000, 1, kernel_size=(1, 1), stride=(1, 1)),
                                #nn.AdaptiveAvgPool2d((1, 1))                              
                                
                                )

        
        """
        self.layers = ssdlite320_mobilenet_v3_large(weights="SSDLite320_MobileNet_V3_Large_Weights.COCO_V1")
        self.layers.transform = nn.Sequential(nn.AdaptiveMaxPool2d(output_size=1),nn.Flatten(),  nn.Linear(128, 1))
        """
        """
        self.layers = shufflenet_v2_x1_5(weights="ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1")
        self.layers.fc = nn.Linear(1024, 1)
        """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        for l in self.layers:
            print(x.shape)
            x = l(x)
        return(x)
        """
        res = self.layers(x)
        return res#.squeeze()

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
        #self.loss_fn = nn.MSELoss()
        self.loss_fn = nn.L1Loss()

    def _train(self, dataloaders: list[DataLoader], model: NN, optimizer: Optimizer) -> None:
        model.train()
        for dataloader in dataloaders:
            for X, y in dataloader:
                X = X.to(self.device)
                y = y.to(self.device).float()
                torch.manual_seed(42)

                pred = model(X).view(-1)
                loss = self.loss_fn(pred, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

    def _test(self, dataloaders: list[DataLoader], model: NN) -> tuple[Any | float, int]:
        errors = 0
        loss = 0
        total_samples = 0
        pred_unique = []

        model.eval()
        with torch.no_grad():
            for dataloader in dataloaders:
                total_samples += len(dataloader.dataset)

                for X, y in dataloader:
                    X = X.to(self.device)
                    y = y.to(self.device).float()
                    pred = model(X).view(-1)
                    for p in set(pred.unique().tolist()):
                        if p not in pred_unique:
                            pred_unique.append(p)
                    loss += self.loss_fn(pred, y).item()
                    errors += torch.sum((torch.round(pred) != y) | ((torch.frac(pred) > 0.25) & (torch.frac(pred) < 0.75))).item()

        return loss / total_samples, int(errors)

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
                "batch_size": self.batch_size_props["batch_size"]
            }
        )

    def _tracking_end(self, model: NN, train_loaders: list[DataLoader]) -> None:
        run = mlflow.active_run()
        if run is not None:
            signature = infer_signature(
                train_loaders[0].dataset[0][0].numpy(), 1.5
            )
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                registered_model_name=run.info.run_name,
                signature=signature,
            )
            mlflow.end_run()

    def _test_and_log(
        self, loaders: dict, model: NN, e: int
    ) -> int:
        test_loss, test_errors  = self._test(loaders["test"], model)
        train_loss, train_errors = self._test(loaders["train"], model)
        #assembly_loss, assembly_errors = self._test([loaders["assembly"],], model)

        if self.tracking:
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "train_errors": train_errors,
                    "test_errors": test_errors,
                    #"assembly_loss": assembly_loss,
                    #"assembly_errors": assembly_errors
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
        epochs: int = 100,
        weight_decay: float = 1e-7,
        model: NN | None = None
    ) -> NN:
        run_name = f"{folder}_{part}_counter"
        run_name = run_name.replace("/", "_")

        img_shape = self.get_img_shape(f"{folder}/{part}")

        if model is None:
            torch.manual_seed(42)
            model = NN(img_shape[0]).to(self.device)
        else:
            print('Training existing model')

        loaders = self.get_dataloaders(f"{folder}/{part}", img_shape, num_in_assembly, part)
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
            self._train(loaders["train"], model, optimizer)            
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
    def get_dataloaders(self, folder: str, img_shape: tuple, num_in_assembly: int, part:str) -> dict:
        ds_lst = []
        normalize_arg = [0.5] * img_shape[0]
        img_transforms = [transforms.ToImage(),
                            transforms.ToDtype(torch.float32, scale=True),
                            transforms.Normalize(normalize_arg, normalize_arg)]
        if img_shape[0] == 1:
            img_transforms = [transforms.Grayscale(num_output_channels=1),] + img_transforms

        img_transforms = transforms.Compose(img_transforms)

        for aspect in os.listdir(folder):
            for direction in ['h', 'v']:
                ds = ImageFolder(root=f"{folder}/{aspect}/{direction}", transform=img_transforms)
                ds_lst.append(ds)

        folders = str.split(folder, "/")     
        ds_src = ImageFolder(root=f"./data/temp/{folders[-3]}/{folders[-2]}", transform=img_transforms)

        assembly_class_idx = ds_src.class_to_idx["assembly"]
        assembly_indices = [i for i, (_, y) in enumerate(ds_src.samples) if y == assembly_class_idx] 
        part_class_idx = ds_src.class_to_idx[part]
        src_train_indices = [i for i, (_, y) in enumerate(ds_src.samples) if y not in [assembly_class_idx, part_class_idx]] 

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
        ds_lst.append(ds_src_train)

        proper_ds_idx = 0

        while len(ds_lst[proper_ds_idx]) < self.BATCH_SIZE_LIMIT:
            proper_ds_idx += 1  # error if all datasets are tiny

        batch_size = 16#self.get_batch_size(img_shape, ds_lst[proper_ds_idx])

        train_loaders = []
        test_loaders = []

        generator = torch.Generator().manual_seed(42)

        for ds in ds_lst:
            splits = random_split(ds, [0.8, 0.2], generator)

            train_loader = DataLoader(
                splits[0],
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
                drop_last=True,  # for aspects with small amounts of samples
                shuffle=True
            )
            train_loaders.append(train_loader)

            test_loader = DataLoader(
                splits[1], batch_size=batch_size, num_workers=4, pin_memory=True
            )
            test_loaders.append(test_loader)

        splits = random_split(ds_assembly, [0.2, 0.8], generator)
        loader_assembly_train = DataLoader(splits[0], batch_size=batch_size, num_workers=4, pin_memory=True)
        train_loaders.append(loader_assembly_train)
        loader_assembly_test = DataLoader(splits[1], batch_size=batch_size, num_workers=4, pin_memory=True)
        test_loaders.append(loader_assembly_test)
        loader_assembly = None#DataLoader(ds_assembly, batch_size=batch_size, num_workers=4, pin_memory=True)
        
        return {"train": train_loaders, "test": test_loaders, "assembly": loader_assembly}

    @staticmethod
    def get_img_shape(img_folder: str) -> tuple[int, int, int]:
        aspect = os.listdir(img_folder)[0]
        img_name = os.listdir(f"{img_folder}/{aspect}/h/1")[0]

        with open(f"{img_folder}/{aspect}/h/1/{img_name}", "rb") as f:
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

        #if (
            #not self.batch_size_props["memory_limited"]
            #and self.batch_size_props["batch_size"] < len(train_idx) // 2
        #):
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
                        batch_size = min(batch_size, 16)  # batch size bigger than 256 can be too big

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
                or (smallest_failed_batch_size - checked_batch_size) / smallest_failed_batch_size < 0.05
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
                drop_last=True
            )
            for _ in range(3):
                self._train([dataloader,], model, optimizer)
                if torch.cuda.mem_get_info()[0] == 0:
                    res = False
                    break
                self._test([dataloader,], model)
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



#!/usr/bin/env python
# coding: utf-8

import math
import time

import torch
import torchaudio

from draugr import Split
from draugr.torch_utilities.evaluation.classification import find_n_misclassified
from draugr.torch_utilities.optimisation.parameters.trainable import (
    trainable_parameters,
)

from neodroidaudition.classification.procs import (
    single_epoch_evaluation,
    single_epoch_fitting,
)

torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False

from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from draugr.torch_utilities import (
    TensorBoardPytorchWriter,
    get_model_hash,
    get_num_parameters,
    global_torch_device,
    load_model,
    save_model,
)
from draugr.random_utilities import seed_stack
from neodroidaudition import PROJECT_APP_PATH
from neodroidaudition.classification import M5
from neodroidaudition.data import SpeechCommands
from neodroidaudition.torch_utilities.collation import collate_transform_wrapped

model_name, model_storage_path = (
    "m5_speech_command",
    PROJECT_APP_PATH.user_data / "speech_command" / "models",
)
device = global_torch_device(True)

if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False
drop_last_train = False

seed_stack(0)
batch_size = 256
new_sample_rate = 8000


def training_procedure(n_epoch=60, train: bool = True, load_previous: bool = True):
    train_set = SpeechCommands(split=Split.Training)

    waveform, sample_rate, category, speaker_id, utterance_number = train_set[0]
    transform = torch.nn.Sequential(
        torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
    )

    model = M5(n_input=transform(waveform).shape[0], n_output=len(train_set.categories))

    persistence_model_name = (
        f"{model_name}_{get_model_hash(transform)}_{get_model_hash(model)}"
    )
    if load_previous:
        candidate = load_model(
            model_name=persistence_model_name,
            model_directory=model_storage_path,
            raise_on_failure=False,
        )
        if candidate:
            model = candidate

    transform.to(device)
    model.to(device)

    collate_fn = collate_transform_wrapped(train_set.label_to_index, transform)

    # print(f"Trainable parameters: {named_trainable_parameters(model)}")
    print(
        f"Number of trainable parameters: {get_num_parameters(model, only_trainable=True)}"
    )

    if train:
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last_train,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        valid_loader = DataLoader(
            SpeechCommands(split=Split.Validation),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        optimiser = optim.Adam(
            trainable_parameters(model), lr=0.01, weight_decay=0.0001
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimiser, step_size=20, gamma=0.1
        )  # reduce the learning after 20 epochs by a factor of 10

        with TensorBoardPytorchWriter(
            PROJECT_APP_PATH.user_log / model_name / str(time.time())
        ) as writer:
            best_valid_acc = math.inf
            for epoch in tqdm(range(1, n_epoch + 1), total=n_epoch, desc="Epoch #"):
                single_epoch_fitting(
                    model,
                    optimiser,
                    train_loader,
                    epoch=epoch,
                    writer=writer,
                    device_=device,
                )
                scheduler.step()
                acc = single_epoch_evaluation(
                    model,
                    valid_loader,
                    subset=Split.Validation,
                    epoch=epoch,
                    writer=writer,
                    device=device,
                )

                if acc < best_valid_acc:
                    best_valid_acc = acc
                    save_model(
                        model,
                        model_name=persistence_model_name,
                        save_directory=model_storage_path,
                    )
                    writer.blip("new_best_model", epoch)
    else:
        seed_stack(0)
        with TensorBoardPytorchWriter(
            PROJECT_APP_PATH.user_log / model_name / str(time.time())
        ) as writer:
            test_loader = DataLoader(
                SpeechCommands(split=Split.Testing),
                batch_size=1,
                shuffle=True,
                drop_last=False,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            acc = single_epoch_evaluation(
                model,
                test_loader,
                subset=Split.Testing,
                epoch=0,
                writer=writer,
                device=device,
            )
            print(acc)
            find_n_misclassified(model, test_loader, mapper=train_set.index_to_label)


training_procedure(train=False)
# training_procedure(train=True)

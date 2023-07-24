import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import datetime
import os

from loss import MTloss
from utils.Data_cs import CityScapes
from utils.anchors import get_anchors
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import SSDDataset, ssd_dataset_collate
from utils.utils import download_weights, get_classes, show_config
from utils.utils_fit import fit_one_epoch, multi_task_trainer
from options import Option
from Mtdd import MTANMobilenet


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    options = Option()

    class_names, num_classes = get_classes(options.classes_path)
    num_classes += 1
    anchors = get_anchors(options.input_shape, options.anchors_size)

    model = MTANMobilenet(num_classes)
    print("Load pretrained mobilenet weight.")

    model_dict = model.state_dict()
    pretrained_dict = torch.load(
        "/content/drive/MyDrive/MTDD/hardfrom100/ep200-loss1.475-val_loss1.735.pth", map_location=device)
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)

    print("\nSuccessful Load Key:", str(load_key)[
          :500], "……\nSuccessful Load Key Num:", len(load_key))
    print("\nFail To Load Key:", str(no_load_key)[
          :500], "……\nFail To Load Key num:", len(no_load_key))

    with open(options.train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(options.val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()

    num_train = len(train_lines)
    num_val = len(val_lines)

    # model.block_mobilenet()
    optimizer = optim.Adam(model.parameters(), lr=options.init_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 250)
    # optimizer = optim.SGD(model.parameters(), options.init_lr, momentum = 0.9, weight_decay = 5e-4)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.5)

    criterion = MTloss(num_classes)

    time_str = datetime.datetime.strftime(
        datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(options.save_dir, "loss_" + str(time_str))
    loss_history = LossHistory(log_dir, model, input_shape=options.input_shape)

    eval_flag = True
    eval_period = 10

    epoch_step = num_train // options.batch_size
    epoch_step_val = num_val // options.batch_size

    train_dataset = CityScapes(train_lines, options.input_shape,
                               anchors, options.batch_size, num_classes, is_train=True)
    val_dataset = CityScapes(val_lines, options.input_shape,
                             anchors, options.batch_size, num_classes, is_train=False)

    cityscapes_train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=options.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        # collate_fn=ssd_dataset_collate,
    )
    cityscapes_val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=options.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        # collate_fn=ssd_dataset_collate,
    )

    eval_callback = EvalCallback(model, options.input_shape, anchors, class_names, num_classes, val_lines, log_dir, options.Cuda,
                                 eval_flag=eval_flag, period=eval_period)
    multi_task_trainer(
        cityscapes_train_loader,
        cityscapes_val_loader,
        model,
        device,
        optimizer,
        scheduler,
        criterion,
        loss_history,
        eval_callback,
        options,
        epoch_step,
        epoch_step_val,
        250
    )

import os

import torch
from tqdm import tqdm
import numpy as np
from utils.utils import get_lr
from utils.callbacks import EvalCallback, LossHistory


def fit_one_epoch(model_train, model, ssd_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    total_loss = 0
    val_loss = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(
            total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                targets = targets.cuda(local_rank)
        if not fp16:

            out = model_train(images)

            optimizer.zero_grad()

            loss = ssd_loss.forward(targets, out)

            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():

                out = model_train(images)

                optimizer.zero_grad()

                loss = ssd_loss.forward(targets, out)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val,
                    desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                targets = targets.cuda(local_rank)

            out = model_train(images)
            optimizer.zero_grad()
            loss = ssd_loss.forward(targets, out)
            val_loss += loss.item()

            if local_rank == 0:
                pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),
                                    'lr': get_lr(optimizer)})
                pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(
            epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:' + str(epoch+1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' %
              (total_loss / epoch_step, val_loss / epoch_step_val))

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" %
                       (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(
                save_dir, "best_epoch_weights.pth"))

        torch.save(model.state_dict(), os.path.join(
            save_dir, "last_epoch_weights.pth"))


def depth_error(x_pred, x_output):
    device = x_pred.device
    binary_mask = (torch.sum(x_output, dim=1) != -1).unsqueeze(1).to(device)
    x_pred_true = x_pred.masked_select(binary_mask)
    x_output_true = x_output.masked_select(binary_mask)
    abs_err = torch.abs(x_pred_true - x_output_true)
    rel_err = torch.abs(x_pred_true - x_output_true) / x_output_true
    return (torch.sum(abs_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)).item(), \
           (torch.sum(rel_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)).item()


def multi_task_trainer(train_loader, test_loader, multi_task_model, device, optimizer, scheduler, loss, loss_history, eval_callback,
                       opt, epoch_step, epoch_step_val, total_epoch=200):
    # train_batch = len(train_loader)
    train_batch = len(train_loader) - 1
    test_batch = len(test_loader)
    print("train_batch: {}".format(train_batch))
    print("test_batch: {}".format(test_batch))
    print("weighting mode: {}".format(opt.weight))
    T = opt.temp
    avg_cost = np.zeros([total_epoch, 24], dtype=np.float32)
    lambda_weight = np.ones([2, total_epoch])

    multi_task_model.train()
    multi_task_model.to(device)
    print('Start Train')

    for index in range(total_epoch):

        if index < 205:
            scheduler.step()
            continue

        ssd_loss = 0
        depth_loss = 0
        total_loss = 0
        ssd_loss_val = 0
        depth_loss_val = 0
        val_loss = 0
        pbar = tqdm(total=train_batch,
                    desc=f'Epoch {index + 1}/{total_epoch}', postfix=dict, mininterval=0.3)
        iter_count = 0
        val_iter_count = 0
        depth_abs = 0
        depth_rel = 0
        depth_abs_v = 0
        depth_rel_v = 0

        multi_task_model.train()
        train_dataset = iter(train_loader)

        for k in range(train_batch):
            iter_count += 1
            train_data, train_label, train_depth = train_dataset.next()
            train_data, train_label = train_data.to(
                device), train_label.to(device)
            train_depth = train_depth.to(device)
            out, logsigma = multi_task_model(train_data)
            ssd_pred, depth_pred = out[0], out[1]
            optimizer.zero_grad()
            train_loss = [loss(ssd_pred, train_label, 'detection'),
                          loss(depth_pred, train_depth, 'depth'),
                          ]

            if opt.weight == 'equal' or opt.weight == 'dwa':
                losses = sum([lambda_weight[i, index] * train_loss[i]
                             for i in range(2)])
            else:
                losses = sum(
                    1 / (2 * torch.exp(logsigma[i])) * train_loss[i] + logsigma[i] / 2 for i in range(2))

            losses.backward()
            optimizer.step()

            d_abs, d_rel = depth_error(depth_pred, train_depth)
            total_loss += losses.item()
            ssd_loss += train_loss[0].item()
            depth_loss += train_loss[1].item()
            depth_abs += d_abs
            depth_rel += d_rel

            pbar.set_postfix(**{'total_loss': total_loss / (k + 1),
                                'lr': get_lr(optimizer),
                                'ssd_loss': ssd_loss / (k + 1),
                                'depth_loss': depth_loss / (k + 1),
                                'depth_abs': d_abs / (k + 1),
                                'depth_rel': d_rel / (k + 1)
                                })
            pbar.update(1)

        if True:
            # evaluating test data
            multi_task_model.eval()
            with torch.no_grad():  # operations inside don't track history
                test_dataset = iter(test_loader)
                for k in range(test_batch):
                    val_iter_count += 1
                    test_data, test_label, test_depth = test_dataset.next()
                    test_data, test_label = test_data.to(
                        device), test_label.to(device)
                    test_depth = test_depth.to(device)

                    out, logsigma = multi_task_model(test_data)
                    ssd_pred, depth_pred = out[0], out[1]
                    test_loss = [loss(ssd_pred, test_label, 'detection'),
                                 loss(depth_pred, test_depth, 'depth'),
                                 ]

                    if opt.weight == 'equal' or opt.weight == 'dwa':
                        losses = sum([lambda_weight[i, index] *
                                     test_loss[i] for i in range(2)])
                    else:
                        losses = sum(
                            1 / (2 * torch.exp(logsigma[i])) * test_loss[i] + logsigma[i] / 2 for i in range(2))

                    d_abs_v, d_rel_v = depth_error(depth_pred, test_depth)

                    val_loss += losses.item()
                    ssd_loss_val += test_loss[0].item()
                    depth_loss_val += test_loss[1].item()
                    depth_abs_v += d_abs_v
                    depth_rel_v += d_rel_v

                    pbar.set_postfix(**{'val_loss': val_loss / (k + 1),
                                        'ssd_loss_val': ssd_loss_val / (k + 1),
                                        'depth_loss_val': depth_loss_val / (k + 1),
                                        'depth_abs_v': depth_abs_v / (k+1),
                                        "depth_rel_v": depth_rel_v / (k+1)
                                        })
                    pbar.update(1)

        scheduler.step()
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(index + 1, total_loss/iter_count, val_loss/val_iter_count, ssd_loss /
                                 iter_count, depth_loss/iter_count, ssd_loss_val/val_iter_count, depth_loss_val/val_iter_count)
        eval_callback.on_epoch_end(index + 1, multi_task_model)
        print('Epoch:' + str(index + 1) + '/' + str(total_epoch))
        print(logsigma[0], logsigma[1])
        print('Total Loss: %.3f || Val Loss: %.3f ' %
              (total_loss / iter_count, val_loss / val_iter_count))

        if (index + 1) % opt.save_period == 0 or index + 1 == total_epoch:
            torch.save(multi_task_model.state_dict(), os.path.join(opt.save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (
                index + 1, total_loss / iter_count, val_loss / val_iter_count)))

        if len(loss_history.val_loss) <= 1 or (val_loss / val_iter_count) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(multi_task_model.state_dict(), os.path.join(
                opt.save_dir, "best_epoch_weights.pth"))

        torch.save(multi_task_model.state_dict(), os.path.join(
            opt.save_dir, "last_epoch_weights.pth"))

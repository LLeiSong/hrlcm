"""
This is a script to trainer.
Reference: https://github.com/lukasliebel/dfc2020_baseline/blob/master/code/train.py
Author: Lei Song
Maintainer: Lei Song (lsong@clarku.edu)
"""

from augmentation import *
from dataset import *
from tqdm.auto import tqdm
import metrics
from loss import loss_coteaching


class Trainer:
    def __init__(self, args):
        self.args = args

    def train(self, model, train_loader, loss_fn, optimizer, writer, step):
        """Train a single model

        :param model: the model to train
        :param train_loader: train Dataloader
        :param loss_fn: loss function
        :param optimizer: optimizer for training
        :param writer: defined writer for statistics
        :param step: global step so far
        :return: updated model and global step
        """
        # Set model to train mode
        model.train()

        # Training loop
        pbar = tqdm(total=len(train_loader), desc="[Train]")
        loss_total = 0
        for i, (image, target, _) in enumerate(train_loader):
            # Move data to gpu if model is on gpu
            if self.args.use_gpu:
                image, target = image.cuda(), target.cuda()

            # Forward pass
            prediction = model(image)
            loss = loss_fn(prediction, target)
            loss_total += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Recalculate step for log progress, validate, and save checkpoint
            global_step = i + step

            # Write current train loss to tensorboard at every step
            writer.add_scalar("train/loss", loss, global_step=global_step)

            # Update progressbar
            pbar.set_description("[Train] Loss: {:.4f}".format(
                round(loss.item(), 4)))
            pbar.update()

        # Close progressbar
        pbar.set_description("[Train] Loss: {:.4f}".format(
            round(loss_total / len(train_loader), 4)))
        pbar.close()

        # Flush to disk
        writer.flush()
        return model, global_step

    def co_train(self, model1, model2, train_loader, loss_fn,
                 optimizer1, optimizer2, noise_or_not, writer, forget_rate, step):
        """Co-train using two models

        :param loss_fn:
        :param noise_or_not:
        :param forget_rate:
        :param optimizer2:
        :param optimizer1:
        :param model1: model 1
        :param model2: model 2
        :param train_loader: train Dataloader
        :param writer: defined writer for statistics
        :param step: global step so far
        :return: updated model1, updated model2 and global step
        """
        # Set model to train mode
        model1.train()
        model2.train()

        # Training loop
        pbar = tqdm(total=len(train_loader), desc="[Train]")
        loss1_total = 0
        loss2_total = 0
        pure_ratio1_list = []
        pure_ratio2_list = []
        for i, (image, target, indexes) in enumerate(train_loader):
            ind = indexes.cpu().numpy().transpose()
            # Move data to gpu if model is on gpu
            if self.args.use_gpu:
                image, target = image.cuda(), target.cuda()

            # Forward pass
            logits1 = model1(image)
            logits2 = model2(image)
            loss1, loss2, pure_ratio1, pure_ratio2 = loss_fn(
                logits1, logits2, target, forget_rate, ind, noise_or_not)
            loss1_total += loss1.item()
            loss2_total += loss2.item()
            pure_ratio1_list.append(100 * pure_ratio1)
            pure_ratio2_list.append(100 * pure_ratio2)

            # Backward pass
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            # Recalculate step for log progress, validate, and save checkpoint
            global_step = i + step

            # Write current train loss to tensorboard at every step
            writer.add_scalar("train/loss1", loss1, global_step=global_step)
            writer.add_scalar("train/loss2", loss2, global_step=global_step)
            writer.add_scalar("train/pure_ratio1", np.sum(pure_ratio1_list)/len(pure_ratio1_list),
                              global_step=global_step)
            writer.add_scalar("train/pure_ratio2", np.sum(pure_ratio2_list) / len(pure_ratio2_list),
                              global_step=global_step)

            # Update progressbar
            pbar.set_description("[Train] Loss1: {:.4f}, Loss2: {:.4f}".format(
                round(loss1.item(), 4), round(loss2.item(), 4)))
            pbar.update()

        # Close progressbar
        pbar.set_description("[Train] Loss1: {:.4f}, Loss2: {:.4f}".format(
            round(loss1_total / len(train_loader), 4),
            round(loss2_total / len(train_loader), 4)))
        pbar.close()

        # Flush to disk
        writer.flush()
        return model1, model2, pure_ratio1_list, pure_ratio2_list, global_step

    def validate(self, model, validate_loader, step, loss_fn, writer):
        """Validate for single model

        :param model: trained model
        :param validate_loader: validate Dataloader
        :param step: global step
        :param loss_fn: loss function
        :param writer: defined writer for statistics
        :return: None
        """
        # Set model to evaluation mode
        model.eval()

        # Validate loop
        pbar = tqdm(total=len(validate_loader), desc="[Val]")
        loss_total = 0
        conf_mat = metrics.ConfMatrix(validate_loader.dataset.n_classes)
        for i, (image, target) in enumerate(validate_loader):
            # Move data to gpu if model is on gpu
            if self.args.use_gpu:
                image, target = image.cuda(), target.cuda()

            # Forward pass
            with torch.no_grad():
                prediction = model(image)
            loss = loss_fn(prediction, target)
            loss_total += loss.cpu().item()

            # Calculate error metrics
            conf_mat.add_batch(target, prediction.max(1)[1])

            # Update progressbar
            pbar.set_description("[Val] Loss: {:.4f}".format(
                round(loss.item(), 4)))
            pbar.update()

        # Write validation metrics to tensorboard
        writer.add_scalar("validate/loss",
                          loss_total / len(validate_loader), global_step=step)
        writer.add_scalar("validate/AA", conf_mat.get_aa(),
                          global_step=step)
        writer.add_scalar("validate/mIoU", conf_mat.get_mIoU(),
                          global_step=step)

        # Close progressbar
        pbar.set_description("[Val] Loss: {:.4f}, AA: {:.2f}%, mIoU: {:.2f}%"
                             .format(round(loss_total / len(validate_loader), 4),
                                     round(conf_mat.get_aa() * 100, 4),
                                     round(conf_mat.get_mIoU() * 100, 4)))
        pbar.close()

        # Flush to disk
        writer.flush()

    def co_validate(self, model1, model2, validate_loader, step, loss_fn, writer):
        """Validate for co-trained two models

        :param model1: model 1
        :param model2: model 2
        :param validate_loader: validate Dataloader
        :param step: global step
        :param loss_fn: loss function
        :param writer: defined writer for statistics
        :return: None
        """
        # Set model to evaluation mode
        model1.eval()
        model2.eval()

        # Validate loop
        pbar = tqdm(total=len(validate_loader), desc="[Val]")
        loss1_total = 0
        loss2_total = 0
        conf_mat1 = metrics.ConfMatrix(validate_loader.dataset.n_classes)
        conf_mat2 = metrics.ConfMatrix(validate_loader.dataset.n_classes)
        for i, (image, target) in enumerate(validate_loader):
            # Move data to gpu if model is on gpu
            if self.args.use_gpu:
                image, target = image.cuda(), target.cuda()

            # Forward pass
            with torch.no_grad():
                logits1 = model1(image)
                logits2 = model2(image)

            loss1 = loss_fn(logits1, target)
            loss1_total += loss1.cpu().item()
            loss2 = loss_fn(logits2, target)
            loss2_total += loss2.cpu().item()

            # Calculate error metrics
            conf_mat1.add_batch(target, logits1.max(1)[1])
            conf_mat2.add_batch(target, logits2.max(1)[1])

            # Update progressbar
            pbar.set_description("[Val] Loss1: {:.4f}, Loss2: {:.4f}".format(
                round(loss1.item(), 4), round(loss2.item(), 4)))
            pbar.update()

        # Write validation metrics to tensorboard
        writer.add_scalar("validate/loss1",
                          loss1_total / len(validate_loader), global_step=step)
        writer.add_scalar("validate/AA1", conf_mat1.get_aa(),
                          global_step=step)
        writer.add_scalar("validate/mIoU1", conf_mat1.get_mIoU(),
                          global_step=step)
        writer.add_scalar("validate/loss2",
                          loss2_total / len(validate_loader), global_step=step)
        writer.add_scalar("validate/AA2", conf_mat2.get_aa(),
                          global_step=step)
        writer.add_scalar("validate/mIoU2", conf_mat2.get_mIoU(),
                          global_step=step)

        # Close progressbar
        pbar.set_description("[Val] Loss1: {:.4f}, AA1: {:.2f}%, mIoU1: {:.2f}%, Loss2: {:.4f}, AA2: {:.2f}%, "
                             "mIoU2: {:.2f}% "
                             .format(round(loss1_total / len(validate_loader), 4),
                                     round(conf_mat1.get_aa() * 100, 4),
                                     round(conf_mat1.get_mIoU() * 100, 4),
                                     round(loss2_total / len(validate_loader), 4),
                                     round(conf_mat2.get_aa() * 100, 4),
                                     round(conf_mat2.get_mIoU() * 100, 4)
                                     ))
        pbar.close()

        # Flush to disk
        writer.flush()

    def export_model(self, model, optimizer=None, name=None, step=None):
        # Set output filename
        if name is not None:
            out_file = name
        else:
            out_file = "checkpoint"
            if step is not None:
                out_file += "_step_" + str(step)
        out_file = os.path.join(self.args.checkpoint_dir, out_file + ".pth")

        # Save model
        data = {"model_state_dict": model.state_dict()}
        if step is not None:
            data["step"] = step
        if optimizer is not None:
            data["optimizer_state_dict"] = optimizer.state_dict()
        torch.save(data, out_file)

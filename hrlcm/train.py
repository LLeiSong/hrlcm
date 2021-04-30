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


class Trainer:
    def __init__(self, args):
        self.args = args

    def train(self, model, train_loader, loss_fn, optimizer, writer, step):
        # Set model to train mode
        model.train()

        # Training loop
        pbar = tqdm(total=len(train_loader), desc="[Train]")
        loss_total = 0
        for i, (image, target) in enumerate(train_loader):
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

    def validate(self, model, validate_loader, step, loss_fn, writer):
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
            pbar.set_description("[Train] Loss: {:.4f}".format(
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
                             .format(round(loss / len(validate_loader), 4),
                                     round(conf_mat.get_aa() * 100, 4),
                                     round(conf_mat.get_mIoU() * 100, 4)))
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

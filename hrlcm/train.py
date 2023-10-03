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
        pbar = tqdm(total=len(train_loader), desc="[Train]", dynamic_ncols=True)
        loss_total = 0
        for i, (image, target, indexes) in enumerate(train_loader):
            # Shrink target
            target = target[:, 4:-4, 4:-4]
            
            # Add replicate padding
            # image = F.pad(image, (4, 4, 4, 4), 'replicate')

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
            writer.add_scalar("Train/loss", loss, global_step=global_step)

            # Update progressbar
            pbar.set_description("[Train] Loss: {:.3f}".format(
                round(loss.item(), 3)))
            pbar.update()

        # Close progressbar
        pbar.set_description("[Train] Loss: {:.3f}".format(
            round(loss_total / len(train_loader), 3)))
        pbar.close()

        # Flush to disk
        writer.flush()
        return model, global_step

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
        pbar = tqdm(total=len(validate_loader), desc="[Val]", dynamic_ncols=True)
        loss_total = 0
        conf_mat = metrics.ConfMatrix(validate_loader.dataset.n_classes)
        for i, (image, target) in enumerate(validate_loader):
            # Shrink target
            target = target[:, 4:-4, 4:-4]
            
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
            pbar.set_description("[Val] Loss: {:.3f}".format(
                round(loss.item(), 3)))
            pbar.update()

        # Write validation metrics to tensorboard
        writer.add_scalar("Validate/loss",
                          loss_total / len(validate_loader), global_step=step)
        writer.add_scalar("Validate/AA", conf_mat.get_aa(),
                          global_step=step)
        writer.add_scalar("Validate/mIoU", conf_mat.get_mIoU(),
                          global_step=step)
        # Add single class accuracy
        confmatrix = np.diagonal(conf_mat.norm_on_lines())
        writer.add_scalar("Validate/Cropland", confmatrix[0], global_step=step)
        writer.add_scalar("Validate/Forest", confmatrix[1], global_step=step)
        writer.add_scalar("Validate/Grassland", confmatrix[2], global_step=step)
        writer.add_scalar("Validate/Shrubland", confmatrix[3], global_step=step)
        writer.add_scalar("Validate/Water", confmatrix[4], global_step=step)
        writer.add_scalar("Validate/Urban", confmatrix[5], global_step=step)
        writer.add_scalar("Validate/Bareland", confmatrix[6], global_step=step)

        # Close progressbar
        pbar.set_description("[Val] Loss: {:.3f}, AA: {:.3f}%, mIoU: {:.3f}%"
                             .format(round(loss_total / len(validate_loader), 3),
                                     round(conf_mat.get_aa() * 100, 3),
                                     round(conf_mat.get_mIoU() * 100, 3)))
        pbar.close()

        # Flush to disk
        writer.flush()
        
        # Return the average accuracy
        return conf_mat.get_aa(), conf_mat.get_mIoU()

    def export_model(self, model, optimizer=None, scheduler=None, name=None, step=None):
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
        if scheduler is not None:
            data['scheduler_state_dict'] = scheduler.state_dict()
        torch.save(data, out_file)

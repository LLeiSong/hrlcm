from augmentation import *
from dataset import *
from tqdm import tqdm
import metrics


class Trainer:
    def __init__(self, args):
        self.args = args

    def train(self, model, train_loader, validate_loader, loss_fn, optimizer, writer, step):
        # Set model to train mode
        model.train()

        # Training loop
        pbar = tqdm(total=len(train_loader), desc="[Train]")
        global_step = 0
        for i, image, target in enumerate(train_loader):
            # Move data to gpu if model is on gpu
            if self.args.use_gpu:
                image, target = image.cuda(), target.cuda()

            # Forward pass
            prediction = model(image)
            loss = loss_fn(prediction, target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Recalculate step for log progress, validate, and save checkpoint
            global_step = i + step

            # Write current train loss to tensorboard at every step
            writer.add_scalar("train/loss", loss, global_step=global_step)

            # Run validation
            if global_step > 0 and global_step % self.args.val_freq == 0:
                self.validate(model, validate_loader, global_step, loss_fn, writer)

            # Save checkpoint
            if global_step > 0 and global_step % self.args.save_freq == 0:
                self.export_model(model, optimizer=optimizer, step=global_step)

            # Update progressbar
            pbar.set_description("[Train] Loss: {:.4f}".format(
                round(loss.item(), 4)))
            pbar.update()

        # Close progressbar and flush to disk
        pbar.close()
        writer.flush()
        return model, global_step

    def validate(self, model, validate_loader, step, loss_fn, writer):
        # Set model to evaluation mode
        model.eval()

        # Validate loop
        pbar = tqdm(total=len(validate_loader), desc="[Val]")
        loss = 0
        conf_mat = metrics.ConfMatrix(validate_loader.dataset.n_classes)
        for i, image, target in enumerate(validate_loader):
            # Move data to gpu if model is on gpu
            if self.args.use_gpu:
                image, target = image.cuda(), target.cuda()

            # Forward pass
            with torch.no_grad():
                prediction = model(image)
            loss += loss_fn(prediction, target).cpu().item()

            # Calculate error metrics
            conf_mat.add_batch(target, prediction.max(1)[1])

            # Update progressbar
            pbar.update()

        # Write validation metrics to tensorboard
        writer.add_scalar("validate/loss",
                          loss / len(validate_loader), global_step=step)
        writer.add_scalar("validate/AA", conf_mat.get_aa(),
                          global_step=step)
        writer.add_scalar("validate/mIoU", conf_mat.get_mIoU(),
                          global_step=step)

        # Close progressbar
        pbar.set_description("[Val] AA: {:.2f}%".format(
                conf_mat.get_aa() * 100))
        pbar.close()

        # Flush to disk
        writer.flush()
        model.train()
        return

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

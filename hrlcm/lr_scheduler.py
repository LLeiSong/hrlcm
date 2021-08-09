import torch
import torch_optimizer as optim


def get_compose_lr(model, epochs, min_lr=0.0001, max_lr=0.001, stage1=50, stage2=130, stage3=170):
    """Util function to customize a learning rate scheduler.
    Note: not a perfect function, user could customize this function based on needs.
        Args:
            model: the defined model
            epochs (int): the overall epochs
            min_lr (float): the last min learning rate
            max_lr (float): the initial max learning rate
            stage1 (int): epoch number 1 to change
            stage2 (int): epoch number 2 to change
            stage3 (int): epoch number 3 to change
        Returns:
            numpy.ndarray: the array of image values
        """

    # Use an arbitrary optimizer
    optimizer_lr = optim.AdaBound(model.parameters(), lr=max_lr, final_lr=0.01, amsbound=True)
    lrs = []
    for i in range(epochs):
        optimizer_lr.step()
        if i <= stage1:
            if i == stage1:
                lr_scheduler_1 = torch.optim.lr_scheduler.CyclicLR(optimizer_lr, base_lr=max_lr - 0.0002,
                                                                   max_lr=max_lr + 0.0002,
                                                                   step_size_up=1, step_size_down=3,
                                                                   gamma=0.97, cycle_momentum=False,
                                                                   mode='exp_range')
                lr_scheduler_1.step()
        elif i <= stage2:
            lr_scheduler_1.step()
            if i == stage2:
                lr_scheduler_2 = torch.optim.lr_scheduler.CyclicLR(optimizer_lr, base_lr=(max_lr - 0.0002) / 2,
                                                                   max_lr=(max_lr + 0.0002) / 2,
                                                                   step_size_up=1, step_size_down=5,
                                                                   gamma=0.94, cycle_momentum=False,
                                                                   mode='exp_range')
                lr_scheduler_2.step()
        elif stage2 < i <= stage3:
            lr_scheduler_2.step()
            if i == stage3:
                for param_group in optimizer_lr.param_groups:
                    param_group["lr"] = min_lr
        lrs.append(
            optimizer_lr.param_groups[0]["lr"]
        )
    del optimizer_lr
    return lrs

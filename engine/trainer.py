
import torch
import torch.distributed as dist
from apex import amp
from utils.common import get_world_size

def reduce_loss(loss):
    world_size = get_world_size()
    if world_size < 2:
        return loss
    with torch.no_grad():
        all_loss = []
        dist.reduce(all_loss, dst=0)
        if dist.get_rank() == 0:
            all_loss /= world_size
    return all_loss

def do_train():
    model.train()
    for image, target in data_loader:
        
        scheduler.step()

        image = image.to(device)
        target = target.to(device)

        loss = model(image, target)

        # TODO: accumulate all loss one master process/ visual training process
        loss_reduced = reduce_loss(loss)

        # TODO: apply loss scaling for mixed-precision recipe
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        
        # something else, such as checkpoint, logging, ...
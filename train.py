
import argparse
import torch
from apex import amp

from utils.common import synchronize

def main():
    parser = argparse.ArgumentParser(description="Distributed training")
    # TODO: add necessarily arguments here
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
        )
        synchronize()

    # TODO: define model structure
    """About model
    model = build_model()
    device = torch.device("cuda")
    model.to(device)
    """

    # TODO: define optimizer and lr_scheduler
    """About optimizer and scheduler
    optimizer = make_optimizer()
    scheduler = make_lr_scheduler()
    """

    # TODO: whether use mixed-precision training
    """mixed-precision training, powed by apex
    use_mixed_precision = True
    amp_opt_level = 'o1' if use_mixed_precision else 'o0'
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)
    """

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            # this should be removed if update BatchNorm stats
            broadcast_buffers=False,
        )

    # save file flag
    save_to_disk = get_rank() == 0

    # TODO: dataset
    """define datasets
    dataset = build_dataset()
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    """

    # TODO: dataloader
    """
    # define collator: BatchCollator
    # define num_workers

    # two ways: 1. define batch_size; 2. define batch_sampler
    # 1.
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=BatchCollator(),
    )

    # 2. 
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=BtachCollator(),
    )
    """

    # TODO: train and validatation
    """
    do_train()
    inference()
    """

if __name__ == "__main__":
    main()

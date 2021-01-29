import operator
from typing import Union

from detectron2.data.build import trivial_batch_collator, worker_init_reset_seed
from detectron2.data.common import AspectRatioGroupedDataset
from detectron2.utils.comm import get_world_size
import torch

__all__ = ["build_batch_data_loader"]


def build_batch_data_loader(  # type: ignore[no-untyped-def]
    dataset,
    sampler,
    total_batch_size: int,
    *,
    aspect_ratio_grouping: bool = False,
    num_workers: int = 0,
    drop_last: bool = True,
) -> Union[torch.utils.data.DataLoader, AspectRatioGroupedDataset]:
    """
    Build a batched dataloader for training.
    Modified from detectron2 to expose the `drop_last` option.

    Args:
        dataset (torch.utils.data.Dataset): map-style PyTorch dataset. Can be indexed.
        sampler (torch.utils.data.sampler.Sampler): a sampler that produces indices
        total_batch_size, aspect_ratio_grouping, num_workers): see
            :func:`build_detection_train_loader`.

    Returns:
        iterable[list]. Length of each list is the batch size of the current
            GPU. Each element in the list comes from the dataset.
    """
    world_size = get_world_size()
    assert (
        total_batch_size > 0 and total_batch_size % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size, world_size
    )

    batch_size = total_batch_size // world_size
    if aspect_ratio_grouping:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            num_workers=num_workers,
            batch_sampler=None,
            collate_fn=operator.itemgetter(
                0
            ),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        return AspectRatioGroupedDataset(data_loader, batch_size)
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, batch_size, drop_last=drop_last
        )  # srnet: expose drop_last to caller
        return torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=trivial_batch_collator,
            worker_init_fn=worker_init_reset_seed,
        )

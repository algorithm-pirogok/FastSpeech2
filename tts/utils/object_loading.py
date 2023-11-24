from torch.utils.data import ConcatDataset, DataLoader

from hydra.utils import instantiate
from tts.collate_fn.collate import collate_fn


def get_dataloaders(clf):
    datasets = []
    for ds in clf["datasets"]:
        datasets.append(instantiate(ds))
    assert len(datasets)
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)
    else:
        dataset = datasets[0]

    # create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=clf.batch_expand_size * clf.batch_size,
        shuffle=True,
        num_workers=clf.get("num_workers", 1),
        collate_fn=collate_fn,
        drop_last=True,
    )

    return dataloader

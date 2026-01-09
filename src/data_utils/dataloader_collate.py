"""Dataloader Collate.

The collate function of the dataloader has to be modified, since we return
4 objects instead of just 1 in the WidarDataset.

Author:
    Yvan Satyawan (modified)
"""
import torch


def widar_collate_fn(samples):
    """Collate function for the Widar dataset.

    Each sample is a tuple of:
        (amp, phase, bvp, info_dict)

    We stack each non-None tensor and keep metadata as a dictionary.

    Returns:
        A tuple of (amps, phases, bvps, infos) with tensors or None.
    """
    amps = torch.stack([s[0] for s in samples]) if samples[0][0] is not None else None
    phases = torch.stack([s[1] for s in samples]) if samples[0][1] is not None else None
    bvps = torch.stack([s[2] for s in samples]) if samples[0][2] is not None else None

    # Collect info keys (all samples assumed to share keys)
    info_keys = samples[0][3].keys()
    infos = {k: [s[3][k] for s in samples] for k in info_keys}

    # Convert gesture to tensor (if it's there)
    if "gesture" in infos:
        infos["gesture"] = torch.tensor(infos["gesture"], dtype=torch.long)

    return amps, phases, bvps, infos


if __name__ == '__main__':
    from pathlib import Path
    from torch.utils.data import DataLoader
    from src.data_utils.widar_dataset import WidarDataset

    # Example usage with domain filtering and 10% fine-tuning
    dataset = WidarDataset(
        Path("C:\Tue\Master\Jaar_2\graduation\code\own_approach\data"),
        split_name="train",
        dataset_type="single_user",
        return_bvp=True,
        bvp_agg="sum"
    )

    dataloader = DataLoader(dataset, batch_size=10, collate_fn=widar_collate_fn)

    for batch in dataloader:
        amps, phases, bvps, infos = batch
        print("Amplitude shape:", None if amps is None else amps.shape)
        print("Phase shape:", None if phases is None else phases.shape)
        print("BVP shape:", None if bvps is None else bvps.shape)
        print("info:", infos)
        break

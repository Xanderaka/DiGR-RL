import pickle
from pathlib import Path
from time import perf_counter
from typing import List, Optional

import csiread
import numpy as np
import torch
from scipy.io import loadmat
from scipy.io.matlab import MatReadError
from torch.utils.data import Dataset

from src.signal_processing.pipeline import Pipeline

class WidarDataset(Dataset):
    csi_length = 2000

    def __init__(self, root_path: Path, split_name: str, dataset_type: str,
                 return_bvp: bool = True,
                 bvp_agg: Optional[str] = None,
                 return_csi: bool = True,
                 amp_pipeline: Optional[Pipeline] = Pipeline([torch.from_numpy]),
                 phase_pipeline: Optional[Pipeline] = Pipeline([torch.from_numpy]),
                 pregenerated: Optional[bool] = None):
        print(f"Loading dataset {split_name}")
        start_time = perf_counter()
        self.data_path = root_path

        # Validate inputs
        valid_splits = ("train", "validation", "test_room", "test_location", "test")
        if split_name not in valid_splits:
            raise ValueError(f"Invalid split_name: {split_name}")
        self.split_name = split_name
        self.dataset_type = dataset_type
        self.return_bvp = return_bvp
        self.return_csi = return_csi

        if (bvp_agg is not None) and (bvp_agg not in ("stack", "1d", "sum")):
            raise ValueError("Invalid bvp_agg.")
        if (bvp_agg is None) and return_bvp:
            raise ValueError("bvp_agg must be specified if return_bvp is True.")
        self.bvp_agg = bvp_agg

        self.amp_pipeline = amp_pipeline or Pipeline([])
        self.phase_pipeline = phase_pipeline or Pipeline([])

        self.pregenerated = False
        valid_dataset_types = ("small", "single_domain", "single_user", "room",
                               "crossfi", "full", "single_domain_small", "single_user_small")
        if dataset_type not in valid_dataset_types:
            raise ValueError("Invalid dataset_type.")

        # Load index file
        if dataset_type == "full":
            index_fp = self.data_path / f"{split_name}_index.pkl"
        else:
            self.pregen_dir = root_path / f"pregenerated_{dataset_type}"
            self.pregenerated = pregenerated if pregenerated is not None else self.pregen_dir.exists()
            print(f"{'Found' if self.pregenerated else 'No'} pregenerated data dir.")
            self.pregen_dir /= split_name
            data_dir = root_path / f"widar_{dataset_type}"
            index_fp = data_dir / f"{split_name}_index_{dataset_type}.pkl"
            self.data_path = data_dir / split_name

        with open(index_fp, "rb") as f:
            index_file = pickle.load(f)
            self.data_records = index_file["samples"]
            self.total_samples = index_file["num_total_samples"]
            self.index_to_csi_index = index_file["index_to_csi_index"]

        # Build domain-to-index mapping
        domain_set = {(
            rec["user"],
            rec["torso_location"],
            rec["face_orientation"],
            rec["room_num"]
        ) for rec in self.data_records}
        self.domain_to_idx = {domain: idx for idx, domain in enumerate(sorted(domain_set))}

        print(f"Loading complete. Took {perf_counter() - start_time:.2f} s.")

    def _load_csi_file(self, csi_file_path: Path) -> np.ndarray:
        if self.dataset_type != "full":
            csi_file_path = self.data_path / csi_file_path.name
        csidata = csiread.Intel(str(csi_file_path), if_report=False)
        csidata.read()
        return csidata.get_scaled_csi_sm(True)[:, :, :, :1]

    def _load_bvp_file(self, bvp_file_path: Path) -> np.ndarray:
        if self.dataset_type != "full":
            bvp_file_path = self.data_path / bvp_file_path.name
        try:
            bvp = loadmat(str(bvp_file_path))["velocity_spectrum_ro"].astype(np.float32)
        except MatReadError:
            bvp = np.zeros((20, 20, 28), dtype=np.float32)

        # Aggregate BVP as specified
        match self.bvp_agg:
            case "stack":
                bvp = np.moveaxis(bvp, -1, 0)
                out = np.zeros((28, 20, 20), dtype=np.float32)
                out[:bvp.shape[0]] = bvp[:28]
            case "1d":
                bvp = np.moveaxis(bvp, -1, 0).reshape((bvp.shape[0], 400))
                out = np.zeros((1, 28, 400), dtype=np.float32)
                out[0, :bvp.shape[0], :] = bvp
            case "sum":
                bvp = np.sum(bvp, axis=2, dtype=np.float32).reshape((1, 20, 20))
                out = bvp
            case _:
                raise ValueError("Invalid bvp_agg value.")

        out = (out - np.min(out)) / (np.max(out) - np.min(out))
        return out

    def _stack_csi_arrays(self, csi_arrays: List[np.ndarray]) -> np.ndarray:
        stacked = np.zeros((self.csi_length, 30, 18), dtype=complex)
        for i, arr in enumerate(csi_arrays):
            arr = arr[:, :, :, 0]
            cutoff = min(arr.shape[0], self.csi_length)
            stacked[:cutoff, :, i * 3:(i + 1) * 3] = arr[:cutoff]
        return stacked

    def __getitem__(self, item):
        data_idx, csi_idx = self.index_to_csi_index[item]
        data_record = self.data_records[data_idx]
        csi_fps = data_record[f"csi_paths_{csi_idx}"]

        # Load BVP
        bvp = None
        if self.return_bvp:
            bvp_paths = data_record[f"bvp_paths_{csi_idx}"]
            bvp_path = Path(bvp_paths[0]) if isinstance(bvp_paths, list) else Path(bvp_paths)
            bvp = torch.from_numpy(self._load_bvp_file(bvp_path))

        # Info dict
        info = {
            "user": data_record["user"],
            "room_num": data_record["room_num"],
            "date": data_record["date"],
            "torso_location": data_record["torso_location"],
            "face_orientation": data_record["face_orientation"],
            "csi_fps": [Path(fp).name for fp in csi_fps],
            "gesture": data_record["gesture"],
        }
        domain = (info["user"], info["torso_location"], info["face_orientation"], info["room_num"])
        info["domain_label"] = info["domain_idx"] = self.domain_to_idx[domain]

        # Load CSI
        if self.return_csi:
            if self.pregenerated:
                file_name = Path(info["csi_fps"][0]).name.split("-r")[0] + ".npz"
                data = np.load(self.pregen_dir / file_name)
                amp, phase = torch.tensor(data["x_amp"]), torch.tensor(data["x_phase"])
            else:
                csi_files = [self._load_csi_file(Path(fp)) if isinstance(fp, str) else self._load_csi_file(fp) for fp in csi_fps]
                csi = self._stack_csi_arrays(csi_files)
                amp = self.amp_pipeline(np.abs(csi).astype(np.float32))
                phase = self.phase_pipeline(np.angle(csi).astype(np.float32))
            amp = amp.to(torch.float32)
            phase = phase.to(torch.float32)
        else:
            amp, phase = None, None

        return amp, phase, bvp, info

    def __len__(self):
        return self.total_samples

    def __str__(self):
        return f"WidarDataset: {self.split_name} ({self.dataset_type})"

    def __repr__(self):
        return f"WidarDataset({self.split_name}, {self.data_path}, ({self.dataset_type}))"


if __name__ == '__main__':
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument("FP", type=Path, help="Path to the data root.")
    args = p.parse_args()
    d1 = WidarDataset(args.FP, "train", dataset_type="single_user_small", return_bvp=False)
    print(d1[0])
    breakpoint()

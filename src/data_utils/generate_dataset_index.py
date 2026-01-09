"""Generate Dataset Index.

Creates an index of file paths to use for the dataset.

# The small dataset is made up of
- The chosen room, user, and torso locations for the specified split
- 10% stratified random set, stratified based on room id, user id, and gesture
- 2 randomly chosen repetitions for each unique sample

The index is in the form of a dictionary.

The dictionary contains keys [`samples`, `num_total_samples`,
 `index_to_csi_index`].

- `samples` is a list of dictionaries where each dictionary item contains:
    - "user": int
    - "room_num": int
    - "torso_location": int
    - "face_orientation": int
    - "gesture": int
    - "repetitions": int
    - "csi_path_{n}": Path
    - "bvp_path_{n}": Path
    where n is the repetitions number.

- `num_total_samples` is the total number of samples, counting repetitions, in
  the split.

- `index_to_csi_index` is used to figure out which sample index and csi index
  a given index refers to. For example, index 1 is sample 0 and csi 1

  4 index files are saved:
    - training_idx.pkl
    - validation_idx.pkl
    - test_room_idx.pkl
    - test_torso_loc_idx.pkl

`_small` is used as a suffix if not the whole dataset is used.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
from argparse import ArgumentParser, Namespace
from pathlib import Path
from warnings import warn
import random
import pickle
import time

from sklearn.model_selection import train_test_split
from scipy.io import loadmat
from tqdm import tqdm
import numpy as np

from src.data_utils import TRAINING_SELECTION, VALIDATION_SELECTION, \
    TEST_ROOM_SELECTION, TEST_LOCATION_SELECTION, \
    ROOM_DATE_MAPPING, DATE_ROOM_MAPPING, SINGLE_DOMAIN_SELECTION, \
    TRAINING_SULO, VALIDATION_SULO, TRAINING_ROOM, VALIDATION_ROOM

def convert_paths_to_str(obj):
    """Recursively convert all Path objects in a structure to strings."""
    if isinstance(obj, dict):
        return {k: convert_paths_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_paths_to_str(v) for v in obj]
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return obj

def parse_args() -> Namespace:
    """Parse command line arguments."""
    p = ArgumentParser()
    p.add_argument("DATA_FP", type=Path,
                   help="Path to the Widar3.0 dataset.")
    p.add_argument("-n", "--num_repetitions", type=int, nargs="?", default=None,
                   help="Number of repetitions to choose for each sample."
                        "Giving none will result in a full dataset index being"
                        "generated.")
    p.add_argument("-s", "--single_domain", action="store_true",
                   help="Set to only get samples of a single domain, in this "
                        "case user 2 in room 1.")
    p.add_argument("-u", "--single_user", action="store_true",
                   help="Sets to single-user leave-out, where user 5 is left "
                        "out from room 1's training set.")
    p.add_argument("-r", "--room", action="store_true",
                   help="Sets to test on the room where trained on room 1 and "
                        "validated on room 2")

    return p.parse_args()


def find_bvp_of_csi(bvp_dir: Path, sample_record: dict, suppress_warnings: bool = False) -> list[Path]:
    """Finds the corresponding BVP file of a CSI file."""
    bvp_search_path = bvp_dir / f"{sample_record['date']}-VS" / "6-link"
    bvp_search_path /= f'user{sample_record["user"] + 1}'
    bvp_paths = []
    for csi_stem in sample_record['csi_stems']:
        bvp_path = sorted(bvp_search_path.glob(f"{csi_stem}-*"))
        if len(bvp_path) > 1:
            raise ValueError("Number of matching BVP files is greater than one.")
        valid_bvps = True
        # Validate each BVP file
        if len(bvp_path) == 0:
            if not suppress_warnings:
                print(f"WARNING: No matching BVP file found for CSI stem: {csi_stem}")
                # You can also log this to a file for later review
                with open("missing_bvp_files.log", "a") as log_file:
                    log_file.write(f"No BVP file for CSI stem: {csi_stem}\n")
            valid_bvps = False
        for bvp_fp in bvp_path:
            bvp = loadmat(str(bvp_fp))["velocity_spectrum_ro"].astype(np.float32)
            if len(bvp.shape) != 3:
                if not suppress_warnings:
                    print(f"WARNING: Broken BVP file for CSI stem: {csi_stem}")
                valid_bvps = False
        if valid_bvps:
            bvp_paths.extend(bvp_path)
    return bvp_paths

def list_date_dirs(criteria: dict[str, any], csi_dir: Path) -> list[Path]:
    date_dirs = []
    for room_num in criteria["room_num"]:
        for date in ROOM_DATE_MAPPING[room_num]:
            date_dirs.append(csi_dir / date)
    return date_dirs


def create_glob_str(criteria: dict[str, any], user_id: str) -> str:
    """Creates the glob string for a search."""
    torso_str = "".join(criteria["torso_location"])
    gesture_str = "".join(criteria["gesture"])
    if "face_orientation" in criteria:
        face_str = "".join(criteria["face_orientation"])
        glob_str = f"user{user_id}/user{user_id}-[{gesture_str}]" \
                   f"-[{torso_str}]-[{face_str}]-[1234]-*"
    else:
        glob_str = f"user{user_id}/user{user_id}-[{gesture_str}]" \
                   f"-[{torso_str}]-[12345]-[1234]-*"

    return glob_str


def find_matching_in_date_dirs(date_dirs_to_search: list[Path],
                               split: dict[str, any],
                               progress_desc: str) -> dict[str, any]:
    """Finds the corresponding files in a dir.

    Args:
        date_dirs_to_search: Directories to search.
        split: The split selection criteria to filter from
            data_utils/__init__.py.
        progress_desc: Description to provide to tqdm
    """
    sample_records = {}
    glob_strs = [create_glob_str(split, user_id)
                 for user_id in split["user"]]
    for date_dir in tqdm(date_dirs_to_search, desc=progress_desc):
        matching_files = []
        for glob_str in glob_strs:
            matching_files += date_dir.glob(glob_str)
        for file in matching_files:
            file_data = file.stem[4:-3].split("-")[:-1]
            file_str = "".join(file_data)
            # user_dir = Path(f"user{file_data[0]}")

            if file_str in sample_records:
                file_stem = file.stem[:-3]
                if file_stem in sample_records[file_str]["csi_stems"]:
                    continue
                else:
                    sample_records[file_str]["csi_stems"].append(file_stem)
                # sample_records[file_str]["bvp_paths"].append(bvp_file)
            else:
                # csi_stems doesn't include the receiver number
                # This way, we have a list of unique repetition file stems.
                # We also turn all numeric values to ints and make them
                # 0-indexed
                sample_records[file_str] = {
                    "user": int(file_data[0]) - 1,
                    "room_num": DATE_ROOM_MAPPING[date_dir.stem],
                    "date": date_dir.stem,
                    "torso_location": int(file_data[2]) - 1,
                    "face_orientation": int(file_data[3]) - 1,
                    "gesture": int(file_data[1]) - 1,
                    "csi_stems": [file.stem[:-3]]
                }
    return sample_records


def process_samples(sample_records: dict[str, any],
                    progress_desc: str,
                    num_repetitions: int | None,
                    bvp_dir: Path,
                    csi_dir: Path) -> dict[str, any]:
    """Processes read samples."""
    keys_to_del = []
    total_samples = 0
    for key in tqdm(sample_records.keys(), desc=progress_desc):
        if num_repetitions is not None:
            chosen_files = random.sample(sample_records[key]["csi_stems"], num_repetitions)
        else:
            chosen_files = sample_records[key]["csi_stems"]

        sample_records[key]["csi_stems"] = sorted(chosen_files)
        bvp_paths = find_bvp_of_csi(bvp_dir, sample_records[key], suppress_warnings=True)

        if len(bvp_paths) != len(chosen_files):
            keys_to_del.append(key)
        else:
            user_str = f"user{sample_records[key]['user'] + 1}"
            file_dir = csi_dir / sample_records[key]["date"] / user_str
            repetitions = len(sample_records[key]["csi_stems"])
            sample_records[key]["repetitions"] = repetitions

            for rep in range(repetitions):
                total_samples += 1
                stem = sample_records[key]["csi_stems"][rep]
                sample_records[key][f"csi_paths_{rep}"] = [
                    file_dir / f'{stem}-r{i}.dat' for i in range(1, 7)
                ]
                sample_records[key][f"bvp_paths_{rep}"] = bvp_paths[rep]

    if len(keys_to_del) > 0:
        warn(f"No BVP found for {len(keys_to_del)} files.")

    for key in keys_to_del:
        del sample_records[key]

    index_to_sample_index = []
    samples = list(sample_records.values())

    for sample_idx, sample in enumerate(samples):
        for csi_idx in range(sample["repetitions"]):
            index_to_sample_index.append((sample_idx, csi_idx))

    output_data = {"samples": samples,
                   "num_total_samples": total_samples,
                   "index_to_csi_index": index_to_sample_index}

    # 👇 Fix: convert all Path objects to strings before pickling
    output_data = convert_paths_to_str(output_data)

    return output_data

# 👇 Update this function too
def parse_single_domain(widar_dir: Path):
    csi_dir = widar_dir / "CSI"
    bvp_dir = widar_dir / "BVP"

    criteria = SINGLE_DOMAIN_SELECTION
    date_dirs = list_date_dirs(criteria, csi_dir)

    sample_records = find_matching_in_date_dirs(date_dirs, criteria, "Reading dirs for single-domain")
    output_data = process_samples(sample_records, "Processing samples for single-domain", None, bvp_dir, csi_dir)

    train_data, validation_data = train_test_split(output_data["index_to_csi_index"], test_size=0.2, random_state=0)

    splits = {
        "train": {
            "samples": output_data["samples"],
            "num_total_samples": len(train_data),
            "index_to_csi_index": train_data,
        },
        "validation": {
            "samples": output_data["samples"],
            "num_total_samples": len(validation_data),
            "index_to_csi_index": validation_data,
        },
    }

    # 👇 Apply fix before saving each split
    for split in splits:
        safe_data = convert_paths_to_str(splits[split])
        with open(widar_dir / f"{split}_index_single_domain.pkl", "wb") as f:
            pickle.dump(safe_data, f)


def parse_files(widar_dir: Path, num_repetitions: int | None,
                splits: tuple, dataset_type: str):
    csi_dir = widar_dir / "CSI"
    bvp_dir = widar_dir / "BVP"
    for split_name, split in splits:
        date_dirs = list_date_dirs(split, csi_dir)
        sample_records = find_matching_in_date_dirs(date_dirs, split, f"Reading dirs for {split_name}")
        output_data = process_samples(sample_records, f"Processing samples for {split_name}", num_repetitions, bvp_dir, csi_dir)

        # 👇 Convert Path objects to strings before saving
        with open(widar_dir / f"{split_name}_index{dataset_type}.pkl", "wb") as f:
            pickle.dump(output_data, f, protocol=4)  # protocol=4 is cross-platform compatible


if __name__ == '__main__':
    random.seed(0)
    args = parse_args()

    if args.single_domain:
        parse_single_domain(args.DATA_FP)
    else:
        if args.single_user:
            select = (("train", TRAINING_SULO), ("validation", VALIDATION_SULO))
            suffix = "_single_user"
        else:
            if args.room:
                select = (("train", TRAINING_ROOM), ("validation", VALIDATION_ROOM))
                suffix = "_room"
            else:
                select = (
                    ("train", TRAINING_SELECTION),
                    ("validation", VALIDATION_SELECTION),
                    ("test_room", TEST_ROOM_SELECTION),
                    ("test_location", TEST_LOCATION_SELECTION)
                )
                suffix = ""

        suffix += "_small" if args.num_repetitions is not None else ""
        parse_files(args.DATA_FP, args.num_repetitions, select, suffix)
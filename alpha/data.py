import json5 as json, random
from collections import defaultdict
from copy import deepcopy
from typing import *

from torch.utils.data import ConcatDataset, Dataset, Subset

from alpha.utils import load_json_file


class DatasetManager:
    def __init__(self, config_path: str):
        self.config = load_json_file(config_path)
        self.default_class = None
        self.class_map = {}
        self.cache = {}

    def set_default_class(self, dataset_class: Type):
        """Set the default class for datasets."""
        self.default_class = dataset_class
        return self

    def register_class(self, class_type: Type, class_name: Optional[str] = None):
        """Register a dataset class with an optional name."""
        name = class_name or class_type.__name__
        self.class_map[name] = class_type

    def _load_dataset(self, name: str):
        """Lazy load a dataset by name."""
        if name in self.cache:
            return self.cache[name]

        dataset_config = self.config['datasets'].get(name)
        if not dataset_config:
            raise ValueError(f"Dataset {name} not found in configuration.")

        dataset_config = deepcopy(dataset_config)
        dataset_class = dataset_config.pop('class', self.default_class)
        if isinstance(dataset_class, str):
            dataset_class = self.class_map.get(dataset_class)
        if not dataset_class:
            raise ValueError(f"No class specified for dataset {name}, and no default class set.")

        # Instantiate the dataset
        dataset = dataset_class(**dataset_config)
        self.cache[name] = dataset
        return dataset

    def get_split(self, split_name: str, enable_weight: bool = False, verbose: bool = False) -> Union[Dataset, Tuple[Dataset, List[float]]]:
        """Get concatenated datasets for a specific split."""
        split_config = self.config['splits'].get(split_name)
        if not split_config:
            raise ValueError(f"Split {split_name} not found in configuration.")

        datasets = []
        weights: List[float] = []
        for entry in split_config:
            
            if verbose:
                print(f"Loading dataset: {entry['dataset']}")
                
            dataset = self._load_dataset(entry['dataset'])
            if 'starts' in entry or 'ends' in entry:
                starts = entry.get('starts', 0)
                ends = entry.get('ends', len(dataset))
                while starts < 0:
                    starts += len(dataset)
                while ends < 0:
                    ends += len(dataset)
                dataset = Subset(dataset, range(starts, ends))
                
            if 'repeat' in entry:
                # idx_lst: 0, 0, 0, 1, 1, 1, ... for e.g. repeat=3
                repeat = entry['repeat']
                indices = []
                for i in range(len(dataset)):
                    indices.extend([i] * repeat)
                dataset = Subset(dataset, indices)
                
            if 'sample' in entry:
                count = len(dataset)
                if 'ratio' in entry['sample']:
                    ratio = entry['sample'].get('ratio', 1.0)
                    count = min(count, int(len(dataset) * ratio))
                elif 'count' in entry['sample']:
                    cnt = entry['sample'].get('count', count)
                    count = min(count, cnt)
                seed = entry['sample'].get('seed', None)
                generator = random.Random(seed)
                indices = generator.sample(range(len(dataset)), count)
                dataset = Subset(dataset, indices)
            
            weight = entry.get('weight', 1.0)
            
            datasets.append(dataset)
            weights.extend([weight] * len(dataset))

        if enable_weight:
            return ConcatDataset(datasets), weights
        else:
            return ConcatDataset(datasets)
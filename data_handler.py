import os
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.transform import resize


def find_npz_files(folder):
    return [f for f in os.listdir(folder) if f.endswith(".npz")]


class Harvard_GF(Dataset):
    """
    Dataset for:
      - OCT 3D volumes
      - 52-point TDS regression
      - Race attribute (for AFF)
    """

    def __init__(
        self,
        data_path,
        modality_type="oct_bscans_3d",
        task="tds",
        resolution=224,
        attribute_type="race",
        transform=None,
    ):
        self.data_path = data_path
        self.modality_type = modality_type
        self.task = task
        self.transform = transform
        self.resolution = resolution

        self.race_mapping = {
            "Asian": 0,
            "Black or African American": 1,
            "White or Caucasian": 2,
        }

        self.data_files = find_npz_files(self.data_path)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_path, self.data_files[idx])
        raw = np.load(file_path, allow_pickle=True)

      
        # OCT 3D
      
        oct_volume = raw["oct_bscans"]  # shape [D, H, W]

        # Resize if needed
        if oct_volume.shape[1] != self.resolution:
            resized = []
            for img in oct_volume:
                resized.append(resize(img, (self.resolution, self.resolution)))
            oct_volume = np.stack(resized, axis=0)

        oct_volume = oct_volume.astype(np.float32)
        oct_volume = oct_volume[None, :, :, :]  # [1, D, H, W]

        if self.transform:
            oct_volume = self.transform(oct_volume)

     
        # Target
        
        if self.task == "tds":
            y = torch.tensor(raw["tds"], dtype=torch.float32)  # [52]
           
        else:
            raise NotImplementedError("Only TDS regression supported.")

      
        # Attribute (Race)
     
        race_str = raw["race"].item()
        race = self.race_mapping[race_str]
        attr = torch.tensor(race, dtype=torch.long)

        return oct_volume, y, attr

from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(
    "data/experiments/multifire_sample/Test/FU"
)

jpg_path = ROOT / "FramesV1_1020.jpg"
tif_path = ROOT / "FramesV1_1020.tif"


for path in [jpg_path, tif_path]:

    print("\n" + "=" * 60)
    print(path.name)
    print("=" * 60)

    image = Image.open(path)

    print("Format:", image.format)
    print("Mode:", image.mode)
    print("Size:", image.size)

    array = np.array(image)

    print("Array shape:", array.shape)
    print("Dtype:", array.dtype)

    unique = np.unique(array)

    print("Unique values:", len(unique))

    if len(unique) <= 30:
        print("Values:", unique)
    else:
        print(
            "Min:", array.min(),
            "Max:", array.max()
        )
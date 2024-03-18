import PixelEncoder
from pathlib import Path
import os

export_dir = Path(
    f"/Users/mary/Documents/School/Sketch Simplification/sketch_simplification_2/dataset"
)

blenderGeneratedJSON_dir = Path(
    f"/Users/mary/Documents/School/Sketch Simplification/sketch_simplification_2/disposable/blenderGeneratedJSON"
)

IMG_SIZE = 64

paths = blenderGeneratedJSON_dir.glob("*.json")

for path in paths:
    filename = os.path.basename(path)
    filename = os.path.splitext(filename)[0]
    PixelEncoder.exportData(path, export_dir / Path(filename), IMG_SIZE)

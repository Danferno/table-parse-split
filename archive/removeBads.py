# Import
import os, shutil
from pathlib import Path
from tqdm import tqdm

# Parameters
path_labels = Path(r"F:\ml-parsing-project\data\parse_activelearning2_png\labels_tableparse")

# List of baddies
baddies = set(["2016-02400322-p11_t0",
    "2012-64000120-p5_t0",
    "2011-16800034-p5_t0",
    "2010-11000569-p181_t0",
    "2010-11000569-p247_t0",
    "2010-11000569-p259_t1"])

# Loop over labels
labelFiles = os.scandir(path_labels)
for labelFileEntry in tqdm(labelFiles):
    if os.path.splitext(labelFileEntry.name)[0] in baddies:
        os.remove(labelFileEntry.path)
        tqdm.write(f'Removed {os.path.splitext(labelFileEntry.name)[0]}')
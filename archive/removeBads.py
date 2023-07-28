# Import
import os, shutil
from pathlib import Path
from tqdm import tqdm

# Parameters
path_labels = Path(r"F:\ml-parsing-project\data\parse_activelearning2_png\labels_tableparse")

# List of baddies
baddies = set([
    "2019-24200492-p3_t0.png",
    "2019-26400264-p18_t0.png",
    "2018-01000018-p8_t0.png",
    "2021-30300200-p16_t0.png",
    "2016-02400322-p11_t0.png",
    "2012-64000120-p5_t0.png",
    "2011-16800034-p5_t0.png",
    "2010-11000569-p181_t0.png",
    "2010-11000569-p247_t0.png",
    "2010-11000569-p259_t1.png",
    "2008-09000033-p4_t0.png",
    "2016-02400322-p11_t0",
    "2012-64000120-p5_t0",
    "2011-16800034-p5_t0",
    "2010-11000569-p181_t0",
    "2010-11000569-p247_t0",
    "2010-11000569-p259_t1",
    "2010-52600245-p14_t1",
    "2010-52600245-p16_t0",
    "2010-52600245-p19_t0",
    "2010-11000569-p96_t0",
    "2012-64000120-p13_t0",
    "2012-64000120-p49_t0",
    "2013-54600415-p10_t0",
    "2013-57700441-p12_t0",
    "2013-57700441-p8_t0",
    "2018-22900400-p13_t0",
    "2018-22900400-p15_t0",
    "2018-22900400-p19_t0",
    "2018-22900400-p23_t0",
    "2018-22900400-p30_t0",
    "2018-22900400-p38_t0",
    "2018-22900400-p5_t0",
    "2018-51400058-p5_t0"
])

# Loop over labels
labelFiles = os.scandir(path_labels)
for labelFileEntry in tqdm(labelFiles):
    if os.path.splitext(labelFileEntry.name)[0] in baddies:
        os.remove(labelFileEntry.path)
        tqdm.write(f'Removed {os.path.splitext(labelFileEntry.name)[0]}')
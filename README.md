# stios_utils

This repository contains utility functions for the Stereo Instance on Surfaces (STOIS) dataset, most importantly:

- Visualization
- Camera parameters
- Annotation generation

### Preliminaries

If you haven't already, please download the dataset from [the project website](https://www.dlr.de/rm/en/desktopdefault.aspx/tabid-17628/#gallery/36367).

To use this code it is suggested to create a new anaconda environment:

```
conda env create -f environment.yaml
```

This creates a new environment named `stios_utils`. All following steps assume this as the active environment.

### 1. Dataset visualization

Run `python visualize.py --root </path/to/stios> [--sensor SENSOR(S)] [--surface SURFACE(S)]`.

Run `python visualize.py --help` for further information.

For a list of _sensors_ see the `SENSORS` definition in [utils/utils.py](utils/utils.py).
For a list of _surfaces_ see the `SURFACES` definition in [utils/utils.py](utils/utils.py).

### 2. Camera parameters

Camera parameters are saved as yaml files in [params/rc_visard.yaml](params/rc_visard.yaml) and [params/zed.yaml](params/zed.yaml), respectively.
You can directly load them using `get_params(sensor)` defined in [utils/utils.py](utils/utils.py).

### 3. Annotation generation

In STIOS masks are saved as 8-bit png images, whereby the grayscale value denotes the instance class. 
This is possible since objects only appear once per image.
To derive corresponding bounding boxes and class names use `get_bboxes(mask)` with `mask` being a np.array:

```python
import cv2
from utils.utils import get_bboxes

mask = cv2.imread('/path/to/mask.png', cv2.IMREAD_GRAYSCALE)
bboxes = get_bboxes(mask)  # returns a list of dicts with keys 'class' and 'bbox' for each detected instance
```

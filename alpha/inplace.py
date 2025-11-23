import numpy as np
from typing import *

import transformers.image_utils

raw_infer_channel_dimension_format = transformers.image_utils.infer_channel_dimension_format
def custom_infer_channel_dimension_format(image: np.ndarray, num_channels: Optional[Union[int, tuple[int, ...]]] = None):
    return raw_infer_channel_dimension_format(image, num_channels or (1, 3, 4))
transformers.image_utils.infer_channel_dimension_format = custom_infer_channel_dimension_format
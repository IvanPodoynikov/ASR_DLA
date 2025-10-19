import math
from collections import defaultdict
from typing import List

import editdistance
import numpy as np
from torch import Tensor

# Based on seminar materials

# Don't forget to support cases when target_text == ''


def calc_cer(target_text, predicted_text) -> float:
    return editdistance.eval(target_text, predicted_text) / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    return editdistance.eval(target_text, predicted_text) / len(target_text)

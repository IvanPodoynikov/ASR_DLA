from typing import List

import editdistance
from torch import Tensor

# Don't forget to support cases when target_text == ''


def calc_cer(target_text, predicted_text) -> float:
    return editdistance.eval(target_text, predicted_text) / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    target_words = target_text.split()
    predicted_words = predicted_text.split()
    return editdistance.eval(target_words, predicted_words) / len(target_words)

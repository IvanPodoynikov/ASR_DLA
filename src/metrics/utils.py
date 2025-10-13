from typing import List

import editdistance
from torch import Tensor

# Don't forget to support cases when target_text == ''


def calc_cer(target_text, predicted_text) -> float:
    assert target_text != "", "Target text is empty"
    return editdistance.eval(target_text, predicted_text) / max(1, len(target_text))


def calc_wer(target_text, predicted_text) -> float:
    assert target_text != "", "Target text is empty"
    target_words = target_text.split()
    predicted_words = predicted_text.split()
    return editdistance.eval(target_words, predicted_words) / max(1, len(target_words))

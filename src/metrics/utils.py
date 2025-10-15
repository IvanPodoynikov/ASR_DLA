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


def ctc_beam_search(
    log_probs: Tensor, ctc_blank: int, beam_size: int = 10
) -> List[int]:
    """
    Performs CTC beam search decoding.

    Args:
        log_probs (Tensor): Log probabilities of shape (T, V) where T is the
            sequence length and V is the vocabulary size.

    Returns:
        best_path (List[int]): The most probable token indices.
    """
    T, V = log_probs.shape
    beams = [(("", ctc_blank), 0.0)]  # ((prefix, int), probability)

    for t in range(T):
        new_beams = {}
        for (prefix, prev_ind), prefix_proba in beams:
            for cur_ind in range(V):
                cur_proba = prefix_proba + log_probs[t, cur_ind].item()

                if cur_ind == ctc_blank:
                    cur_prefix = prefix
                elif cur_ind == prev_ind:
                    cur_prefix = prefix
                else:
                    cur_prefix = prefix + str(cur_ind)

                key = (cur_prefix, cur_ind)
                new_beams[key] = cur_proba

        # Keep only the top `beam_size` beams
        beams = sorted(new_beams.items(), key=lambda x: x[1], reverse=True)[:beam_size]
        beams = [((prefix, char), proba) for (prefix, char), proba in beams]

    best_path = max(beams, key=lambda x: x[1])[0][0]
    best_path = [int(ch) for ch in best_path]
    return best_path

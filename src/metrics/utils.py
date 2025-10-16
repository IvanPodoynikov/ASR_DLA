import math
from collections import defaultdict
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


def _log_add(a, b):
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a
    return math.logaddexp(a, b)


def _expand_and_merge_beams(dp, cur_log_prob, ctc_blank):
    V = cur_log_prob.shape[0]
    new_dp = defaultdict(lambda: float("-inf"))

    for (pref, prev_char), pref_logp in dp.items():
        for idx in range(V):
            logp = cur_log_prob[idx].item()
            new_logp = pref_logp + logp

            if idx == ctc_blank:
                new_pref = pref
            else:
                if idx == prev_char:
                    new_pref = pref
                else:
                    new_pref = pref + (idx,)

            key = (new_pref, idx)
            new_dp[key] = _log_add(new_dp[key], new_logp)

    return dict(new_dp)


def _truncate_beams(dp, beam_size):
    items = sorted(dp.items(), key=lambda kv: kv[1], reverse=True)[:beam_size]
    return dict(items)


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
    dp = {(tuple(), ctc_blank): 0.0}

    for t in range(T):
        cur_log_prob = log_probs[t]
        dp = _expand_and_merge_beams(dp, cur_log_prob, ctc_blank)
        dp = _truncate_beams(dp, beam_size)

    (best_prefix, _), best_logp = max(dp.items(), key=lambda kv: kv[1])
    return list(best_prefix)

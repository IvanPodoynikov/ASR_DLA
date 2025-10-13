import torch
from torch.nn.utils.rnn import pad_sequence


# оч странно, что функция к self не относится, не могу из baseline.yaml тянуть инфу
def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    def _strip_leading_channel(key: str, t: torch.Tensor):
        if key == "spectrogram":
            return t.squeeze(0).transpose(0, 1)  # (1, F, T) -> (T, F)
        elif key == "text_encoded":
            return t.squeeze(0)
        return t

    keys = list(dataset_items[0].keys())
    result_batch = {key: [] for key in keys}
    for key in keys:
        values = [_strip_leading_channel(key, item[key]) for item in dataset_items]
    if key in ["spectrogram", "text_encoded"]:
        result_batch[key] = pad_sequence(values, batch_first=True, padding_value=0)
        result_batch[f"{key}_length"] = torch.Tensor(
            [value.shape[0] for value in values]
        ).long()
        if key == "spectrogram":
            result_batch[key] = result_batch[key].transpose(1, 2)  # (B, F, T)
    else:
        result_batch[key] = values  # keep as list (e.g., strings, paths)
    return result_batch

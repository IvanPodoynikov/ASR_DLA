import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


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

    result_batch = {
        key: []
        for key in [
            "audio",
            "spectrogram",
            "text",
            "text_encoded",
            "audio_path",
            "spectrogram_length",
            "text_encoded_length",
        ]
    }
    if not dataset_items:
        return result_batch

    for element in dataset_items:
        result_batch["audio"].append(element["audio"])
        result_batch["audio_path"].append(element["audio_path"])

        result_batch["spectrogram"].append(
            element["spectrogram"].squeeze(0).transpose(0, 1)  # (1, F, T) -> (T, F)
        )
        result_batch["spectrogram_length"].append(element["spectrogram"].shape[-1])

        result_batch["text"].append(element["text"])
        result_batch["text_encoded"].append(element["text_encoded"].transpose(-1, -2))
        result_batch["text_encoded_length"].append(element["text_encoded"].shape[-1])

    result_batch["spectrogram"] = (
        pad_sequence(result_batch["spectrogram"], batch_first=True)
        .transpose(-1, -2)
        .contiguous()
    )
    result_batch["spectrogram_length"] = torch.tensor(
        result_batch["spectrogram_length"]
    )

    result_batch["text_encoded"] = (
        pad_sequence(result_batch["text_encoded"], batch_first=True)
        .squeeze(-1)
        .contiguous()
    )
    result_batch["text_encoded_length"] = torch.tensor(
        result_batch["text_encoded_length"]
    )
    return result_batch

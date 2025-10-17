import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


def collate_fn_levon(dataset_items: list[dict]):
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
    if not dataset_items:
        return {}

    batch = {}

    for sample in dataset_items:
        for key, val in sample.items():
            if key not in batch:
                batch[key] = []
            batch[key].append(val)

    def pad_tensors(tensors):
        # список тензоров. Нашли макс длину
        max_len = max(t.shape[-1] for t in tensors)
        padded_tensors = []
        lengths = []
        for t in tensors:
            # текущая длина
            lengths.append(t.shape[-1])
            # сколько допаддить
            padding_size = max_len - t.shape[-1]
            padded = nn.functional.pad(t, (0, padding_size))
            padded_tensors.append(padded)
        padded_tensors = torch.concat(padded_tensors)
        lengths = torch.tensor(lengths)
        return padded_tensors, lengths

    keys = list(batch.keys())
    for key in keys:
        if key in ["audio", "spectrogram", "text_encoded"]:
            padded, lengths = pad_tensors(batch[key])
            batch[key] = padded
            batch[key + "_length"] = lengths
        elif key in ["text", "audio_path"]:
            pass
        else:
            raise ValueError("Unexpected key `%s` encountered" % (key,))

    return batch


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

    # print(dataset_items[0]['spectrogram'].shape, dataset_items[1]['spectrogram'].shape)
    # assert 1 == 0

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
    levon_result = collate_fn_levon(dataset_items)
    print(
        result_batch["spectrogram_length"].shape,
        levon_result["spectrogram_length"].shape,
    )
    assert result_batch["spectrogram"].shape == levon_result["spectrogram"].shape
    assert (
        result_batch["spectrogram_length"].shape
        == levon_result["spectrogram_length"].shape
    )
    assert result_batch["text_encoded"].shape == levon_result["text_encoded"].shape

    assert torch.allclose(
        result_batch["spectrogram"], levon_result["spectrogram"]
    ), "NOT EQUAL COLLATE"
    assert torch.allclose(
        result_batch["spectrogram_length"], levon_result["spectrogram_length"]
    )
    assert torch.allclose(result_batch["text_encoded"], levon_result["text_encoded"])
    assert torch.allclose(
        result_batch["text_encoded_length"], levon_result["text_encoded_length"]
    )
    # print(f"My shape: {result_batch['spectrogram'][0].shape}. Levon shape: {levon_result['spectrogram'][0].shape}")
    torch.testing.assert_close(result_batch["spectrogram"], levon_result["spectrogram"])
    # assert result_batch['audio'] == levon_result['audio']

    return result_batch

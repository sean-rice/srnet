import copy

from detectron2.structures import ImageList


def split_image_list(il: "ImageList", i: int, mode: str = ">=") -> "ImageList":
    r"""
    Split an `ImageList` at a specified index, preserving the internal (padded)
    image sizes and using the *same* backing `Tensor`.

    Parameters:
        il (detectron2.structures.ImageList): The ImageList to split.
        i (int): The index to use to split the ImageList.
        mode (str): Which portion of the split to return; options are
            ">=", which will contain il[i:];
            "<=", which will contain il[:i+1];
            "<", which will contain il[:i].
    
    Returns:
        split_list (detectron2.structures.ImageList): The split list.
    """
    if mode == ">=":
        tensor = il.tensor[i:, ..., :, :]
        sizes = copy.deepcopy(il.image_sizes[i:])
    elif mode == "<=":
        tensor = il.tensor[: i + 1, ..., :, :]
        sizes = copy.deepcopy(il.image_sizes[: i + 1])
    elif mode == "<":
        tensor = il.tensor[:i, ..., :, :]
        sizes = copy.deepcopy(il.image_sizes[:i])
    else:
        raise ValueError(f'invalid split mode {mode}; use ">=", "<=", or "<".')

    assert (
        tensor.storage().data_ptr() == il.tensor.storage().data_ptr()
    ), "split_image_list: failed; data_ptrs mismatch"

    split_list = ImageList(tensor, sizes)
    return split_list


def repad_image_list(
    il: "ImageList", pad_value: float = 0.0, inplace: bool = True
) -> "ImageList":
    if inplace == False:
        il = ImageList(il.tensor.clone().detach(), copy.deepcopy(il.image_sizes))
    for i in range(len(il.image_sizes)):
        h, w = il.image_sizes[i]
        il.tensor[i, ..., h:, :] = pad_value
        il.tensor[i, ..., :, w:] = pad_value
    return il

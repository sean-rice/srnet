import unittest

from detectron2.structures import ImageList
import torch
import torch.nn.functional

from srnet.utils.image_list import repad_image_list


class TestImageListUtils(unittest.TestCase):
    def _test_repad_no_channels(self, test_with_inplace: bool):
        cpu = torch.device("cpu")
        t1: torch.Tensor = torch.rand((3, 3), device=cpu)
        t2: torch.Tensor = torch.rand((5, 5), device=cpu)
        il = ImageList.from_tensors([t1, t2], pad_value=0.0)
        true_answer: torch.Tensor = torch.stack(
            [
                torch.nn.functional.pad(t1, (0, 2, 0, 2), mode="constant", value=100.0),
                t2,
            ]
        )

        old_data_ptr = il.tensor.data_ptr
        il = repad_image_list(il, pad_value=100.0, inplace=test_with_inplace)

        # this is a case where "hard" equal (rather than allclose) is desirable
        self.assertTrue(torch.equal(il.tensor, true_answer))

        same_backing_tensor: bool = il.tensor.data_ptr == old_data_ptr
        if test_with_inplace == True:
            self.assertTrue(same_backing_tensor)
        else:
            self.assertFalse(same_backing_tensor)

    def test_repad_no_channels(self):
        self._test_repad_no_channels(test_with_inplace=False)

    def test_repad_no_channels_inplace(self):
        self._test_repad_no_channels(test_with_inplace=True)

    def _test_repad_c_channels(self, c: int, test_with_inplace: bool):
        cpu = torch.device("cpu")
        t1: torch.Tensor = torch.rand((c, 3, 3), device=cpu)
        t2: torch.Tensor = torch.rand((c, 5, 5), device=cpu)
        il = ImageList.from_tensors([t1, t2], pad_value=0.0)
        true_answer: torch.Tensor = torch.stack(
            [
                torch.nn.functional.pad(t1, (0, 2, 0, 2), mode="constant", value=100.0),
                t2,
            ]
        )

        old_data_ptr = il.tensor.data_ptr
        il = repad_image_list(il, pad_value=100.0, inplace=test_with_inplace)

        # this is a case where "hard" equal (rather than allclose) is desirable
        self.assertTrue(torch.equal(il.tensor, true_answer))

        same_backing_tensor: bool = il.tensor.data_ptr == old_data_ptr
        if test_with_inplace == True:
            self.assertTrue(same_backing_tensor)
        else:
            self.assertFalse(same_backing_tensor)

    def test_repad_1_channel(self):
        self._test_repad_c_channels(c=1, test_with_inplace=False)

    def test_repad_1_channel_inplace(self):
        self._test_repad_c_channels(c=1, test_with_inplace=True)

    def test_repad_2_channels(self):
        self._test_repad_c_channels(c=2, test_with_inplace=False)

    def test_repad_2_channels_inplace(self):
        self._test_repad_c_channels(c=2, test_with_inplace=True)

    def test_repad_3_channels(self):
        self._test_repad_c_channels(c=2, test_with_inplace=False)

    def test_repad_3_channels_inplace(self):
        self._test_repad_c_channels(c=2, test_with_inplace=True)

    def test_repad_many_channels(self):
        for c in range(4, 11):
            self._test_repad_c_channels(c=c, test_with_inplace=False)

    def test_repad_many_channels_inplace(self):
        for c in range(4, 11):
            self._test_repad_c_channels(c=c, test_with_inplace=True)

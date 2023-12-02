import unittest
from support_module.ImagesDataset import ImagesDataset
import pathlib
import numpy as np


class MyTestCase(unittest.TestCase):
    path = pathlib.Path("/Users/daniilsuhanov/DataspellProjects/ML-intensive-Yandex-Academy-autumn-2023/data")

    def test1(self):
        dataset = ImagesDataset(self.path.joinpath("train_images"), self.path.joinpath("train_answers.csv"))
        item = dataset[0]
        self.assertEqual(
            type(item[0]) is np.ndarray, type(item[1]) is float,
            msg=f"{type(item[0])=}\n{type(item[1])=}"
        )

    def test2(self):
        dataset = ImagesDataset(self.path.joinpath("train_images"), self.path.joinpath("train_lung_masks"))
        item = dataset[0]
        self.assertEqual(
            type(item[0]) is np.ndarray, type(item[1]) is np.ndarray,
            msg=f"{type(item[0])=}\n{type(item[1])=}"
        )

    def test3(self):
        dataset = ImagesDataset(self.path.joinpath("train_images"))
        item = dataset[0]
        self.assertEqual(
            type(item[0]) is np.ndarray, item[1] is None,
            msg=f"{type(item[0])=}\n{type(item[1])=}"
        )


if __name__ == '__main__':
    unittest.main()

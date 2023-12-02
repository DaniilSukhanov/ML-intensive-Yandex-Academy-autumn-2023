import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import pathlib
from typing import Union, Optional, TypeAlias
import csv


NPImage: TypeAlias = np.ndarray[np.ndarray[float]]


class ImagesDataset(Dataset):
    """
    Датасет для работы с изображениями в качестве данных для обучения.

    :param path_directory_images: Путь к папке с изображениями для обучения.
    :param path_answers: Путь к файлу с ответами (необязательный).
    """
    __path_images: Optional[list[pathlib.Path]]
    __answer: Optional[Union[tuple[float], list[pathlib.Path]]]

    def __init__(
            self, path_directory_images: Union[str, pathlib.Path],
            path_answers: Optional[Union[str, pathlib.Path]] = None
    ):
        if type(path_directory_images) is str:
            path_directory_images = pathlib.Path(path_directory_images)
        path_directory_images: pathlib.Path
        # Загрузка изображений и их сортировка
        self.__path_images = list(path_directory_images.glob("*.png"))
        self.__path_images.sort(key=ImagesDataset.__get_index)

        if path_answers is not None and type(path_answers) is str:
            path_answers = pathlib.Path(path_answers)
        path_answers: Optional[pathlib.Path]
        if path_answers is None:
            self.__answer = None
        elif path_answers.is_file():
            # Если есть файл с ответами, считываем их из CSV
            with open(path_answers) as file:
                reader = csv.reader(file, delimiter=",")
                next(reader)
                self.__answer = tuple(float(row[1]) for row in reader)
        else:
            # Если ответы - это изображения, сортируем их
            self.__answer = list(path_answers.glob("*.png"))
            self.__answer.sort(key=ImagesDataset.__get_index)

    def __len__(self) -> int:
        return len(self.__path_images)

    def __getitem__(
        self, index: int
    ) -> tuple[np.ndarray[np.ndarray[float]], Optional[Union[float, NPImage]]]:
        # Получаем изображение и соответствующий ответ по индексу
        images_path = self.__path_images[index]
        with Image.open(images_path) as image:
            # Преобразуем изображение в массив numpy и нормализуем его
            data_image = np.array(image, dtype=np.float64)
            data_image /= 256.0
        if self.__answer is None:
            answer = None
        else:
            answer = self.__answer[index]
            if type(answer) is pathlib.PosixPath:
                with Image.open(answer) as image:
                    # Преобразуем изображение в массив numpy и нормализуем его
                    answer = np.array(image, dtype=np.float64)
                    answer /= 256.0
        return data_image, answer

    @staticmethod
    def __get_index(filename: pathlib.Path) -> int:
        """Вспомогательный метод для извлечения индекса из имени файла"""
        return int(filename.name.split(".")[0].replace("img_", "", 1))


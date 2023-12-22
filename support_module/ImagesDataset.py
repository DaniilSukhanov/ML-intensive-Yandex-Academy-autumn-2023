import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision import transforms


class ImageToImageDataset(Dataset):
    """
    Создает набор данных для работы с парами изображений.

    Parameters:
        input_dir (str): Путь к папке с входными изображениями.
        target_dir (str, optional): Путь к папке с целевыми изображениями (по умолчанию: None).
        transform (callable, optional): Преобразование изображений (по умолчанию: None).
        mode (str): Режим работы ('train' или другой) (по умолчанию: 'train').

    Attributes:
        input_dir (str): Путь к папке с входными изображениями.
        target_dir (str): Путь к папке с целевыми изображениями (если указан).
        transform (callable): Функция преобразования изображений.
        mode (str): Режим работы.
        filenames (list): Список имен файлов изображений в папке с входными изображениями.
    """
    def __init__(self, input_dir, target_dir=None, transform=None, mode='train'):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        self.mode = mode

        # Получение списка имен файлов в папке с входными изображениями
        self.filenames = [f for f in os.listdir(input_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # Формирование пути к входному изображению и его открытие
        input_path = os.path.join(self.input_dir, self.filenames[idx])
        input_image = Image.open(input_path).convert('L')

        # Применение преобразования к входному изображению, если оно задано
        if self.transform:
            input_image = self.transform(input_image)

        # Если режим 'train', получаем путь к целевому изображению и его открытие
        if self.mode == 'train':
            target_path = os.path.join(self.target_dir, self.filenames[idx])
            target_image = Image.open(target_path).convert('L')
            # Возвращаем пару изображений (входное и выходной)
            return input_image, target_image
        else:
            # Возвращаем только входное изображение
            return input_image


class ImageToNumDataset(Dataset):
    """
    Создает набор данных изображений.

    Parameters:
        img_dir (str): Путь к папке с изображениями.
        transform (callable, optional): Преобразование изображений (по умолчанию: None).
        answers_file (str, optional): Путь к файлу с ответами (по умолчанию: None).

    Attributes:
        img_dir (str): Путь к папке с изображениями.
        transform (callable): Функция преобразования изображений.
        answers_file (str): Путь к файлу с ответами (если указан).
        img_labels (DataFrame): DataFrame с метками изображений или None, если метки отсутствуют.
        image_filenames (list): Список имен файлов изображений в папке.
    """
    def __init__(self, img_dir, transform=None, answers_file=None):
        # Инициализация класса ImageToNumDataset с указанием директории изображений,
        # возможности трансформации и файла с ответами
        self.img_dir = img_dir
        self.transform = transform
        self.answers_file = answers_file

        # Если указан файл с ответами, загружаем его в виде DataFrame,
        # иначе оставляем метки изображений пустыми
        if self.answers_file is not None:
            self.img_labels = pd.read_csv(answers_file)
        else:
            self.img_labels = None

        # Получение списка имен файлов изображений в указанной директории и их сортировка по номеру
        self.image_filenames = [file for file in os.listdir(img_dir) if file.endswith('.png')]
        self.image_filenames.sort(key=lambda x: int(x.replace("img_", "", 1).replace(".png", "", 1)))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Получение имени файла изображения по индексу
        img_name = self.image_filenames[idx]
        # Формирование пути к изображению
        img_path = os.path.join(self.img_dir, img_name)
        # Открытие изображения и преобразование в оттенки серого
        image = Image.open(img_path).convert("L")

        # Применение трансформации (если указана)
        if self.transform:
            image = self.transform(image)

        # Если имеются метки изображений, возвращаем изображение и соответствующую метку
        if self.img_labels is not None:
            label = self.img_labels.iloc[idx, 1]
            return image, label
        else:
            # Если меток нет, возвращаем только изображение
            return image

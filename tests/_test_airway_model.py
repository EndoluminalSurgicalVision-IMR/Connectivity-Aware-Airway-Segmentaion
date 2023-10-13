# -*- coding: utf-8 -*-

from models.airway_model import AirwayExtractionModel
from util.utils import load_itk_image, save_itk

airwayextractor = AirwayExtractionModel()
in_path = '../sample/image.nii.gz'
out_path = '../sample/airway.nii.gz'
image, origin, spacing, direction = load_itk_image(in_path)
airway = airwayextractor.predict(image)

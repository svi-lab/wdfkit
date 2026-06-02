# -*- coding: utf-8 -*-
"""PIL / white-light image helpers."""

from PIL import ImageFile

from . import constants as const

# Allow truncated JPEG blocks when decoding embedded camera images from WHTL.
ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_exif(img):
    """Recover exif data from a PIL image."""

    img_exif = dict()
    for tag, value in img._getexif().items():
        decoded_tag = const.EXIF_TAGS.get(tag, tag)
        img_exif[decoded_tag] = value
    dunit = img_exif["FocalPlaneResolutionUnit"]
    img_exif["FocalPlaneResolutionUnit"] = const.DATA_UNITS.get(dunit, dunit)
    return img_exif

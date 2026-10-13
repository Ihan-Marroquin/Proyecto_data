import unittest

import numpy as np
import pandas as pd
from PIL import Image

from src.modeling import (
    COLOR_FEATURE_LENGTH,
    crop_from_annotation,
    extract_feature_vector,
    select_feature_set,
)
from src.train_models import validate_training_config


class ModelingTests(unittest.TestCase):
    def test_out_of_bounds_bbox_is_clipped(self):
        image = Image.new("RGB", (100, 80), "white")
        row = pd.Series({
            "bbx_xtl": 10,
            "bbx_ytl": 5,
            "bbx_xbr": 140,
            "bbx_ybr": 95,
        })
        crop = crop_from_annotation(image, row, padding=0)
        self.assertEqual(crop.size, (90, 75))

    def test_feature_vector_is_finite_and_deterministic(self):
        array = np.zeros((120, 80, 3), dtype=np.uint8)
        array[20:100, 25:55] = [30, 140, 70]
        image = Image.fromarray(array)
        first = extract_feature_vector(image)
        second = extract_feature_vector(image)
        self.assertGreater(len(first), COLOR_FEATURE_LENGTH)
        self.assertTrue(np.isfinite(first).all())
        np.testing.assert_allclose(first, second)

    def test_feature_slices_match_expected_lengths(self):
        vector = extract_feature_vector(Image.new("RGB", (96, 96), "gray"))[None, :]
        color = select_feature_set(vector, "color")
        hog = select_feature_set(vector, "hog")
        self.assertEqual(color.shape[1], COLOR_FEATURE_LENGTH)
        self.assertEqual(hog.shape[1] + color.shape[1], vector.shape[1])

    def test_training_config_accepts_valid_values(self):
        validate_training_config(0.15, 0.15, 600, 96)

    def test_training_config_rejects_invalid_split(self):
        with self.assertRaisesRegex(ValueError, "suma"):
            validate_training_config(0.6, 0.4, 600, 96)

    def test_training_config_rejects_invalid_limits(self):
        with self.assertRaisesRegex(ValueError, "mayor que cero"):
            validate_training_config(0.15, 0.15, 0, 96)
        with self.assertRaisesRegex(ValueError, "al menos 32"):
            validate_training_config(0.15, 0.15, 600, 16)


if __name__ == "__main__":
    unittest.main()

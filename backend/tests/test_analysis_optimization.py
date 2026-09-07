"""Regression coverage for bounded analysis views and temporary result reuse."""

import time
import unittest

import cv2
import numpy as np

from app.analysis.image_optimizer import OCR_MAX_SIDE, scale_values, text_regions
from app.analysis.session_cache import AnalysisSessionCache
from app.schemas import RawDetection


class AnalysisOptimizationTests(unittest.TestCase):
    def test_fast_ocr_keeps_day16_small_text_recall_floor(self):
        self.assertGreaterEqual(OCR_MAX_SIDE["fast"], 640)

    def test_scaled_boxes_map_back_to_original_coordinates(self):
        raw = RawDetection(0, "person", 0.9, 10, 20, 50, 60)
        mapped = scale_values([raw], 2.0, 1.5)[0]
        self.assertEqual((mapped.x1, mapped.y1, mapped.x2, mapped.y2), (20, 30, 100, 90))

    def test_text_roi_finder_rejects_blank_scene_and_finds_rendered_text(self):
        blank = np.zeros((240, 640, 3), dtype=np.uint8)
        self.assertEqual(text_regions(blank), [])
        cv2.putText(blank, "PRIVATE 1234567890", (25, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        self.assertTrue(text_regions(blank))

    def test_session_cache_is_bounded_and_expires(self):
        class Copyable:
            def __init__(self, value): self.value = value
            def model_copy(self, deep=False): return Copyable(self.value)
        cache = AnalysisSessionCache(max_entries=1, ttl_seconds=1)
        first = cache.put(Copyable("one"), "hash-one")
        second = cache.put(Copyable("two"), "hash-two")
        self.assertIsNone(cache.get(first))
        self.assertEqual(cache.get(second).content_sha256, "hash-two")
        cache._entries[second] = cache._entries[second].__class__(cache._entries[second].analysis, "hash-two", time.time() - 1)
        self.assertIsNone(cache.get(second))


if __name__ == "__main__":
    unittest.main()

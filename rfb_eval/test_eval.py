import unittest

import evaluate as ev
from evaluate import IOU

import os
GUID = 'cpb-aacip-81-902z3f9j'
PRED_FILE = os.path.join(os.getcwd(), 'test_data', f"{GUID}.pred.mmif")
GOLD_FILE = os.path.join(os.getcwd(), 'test_data', f"{GUID}.gold.csv")


class TestLoading(unittest.TestCase):
    """Test loading of prediction and gold files"""

    def test_load_pred(self):
        """Test loading of prediction file"""
        preds = ev.load_pred(PRED_FILE)
        self.assertIsInstance(preds, dict)
        self.assertIsNotNone(preds)
        self.assertEqual(len(preds), 1)
        self.assertEqual(list(preds.keys())[0], GUID)
        pairs_sum = 0
        for _, bindings in preds[GUID].items():
            pairs_sum += len(bindings)
        self.assertEqual(pairs_sum, 17)

    def test_load_gold(self):
        """Test loading of gold file"""
        golds = ev.load_gold(GOLD_FILE)
        self.assertIsInstance(golds, dict)
        self.assertIsNotNone(golds)
        self.assertEqual(len(golds), 1)
        self.assertEqual(list(golds.keys())[0], GUID)
        num_anns = 0
        num_pairs = 0
        for span, bindings in golds[GUID].items():
            self.assertIsInstance(span, tuple)
            num_anns += 1
            num_pairs += len(bindings)
        self.assertEqual(num_pairs, 17)
        self.assertEqual(num_anns, 6)


class TestIOU(unittest.TestCase):
    """Test IOU calculation"""

    def test_iou(self):
        """Test IOU calculation"""
        iou_metric = IOU(ev.load_gold(GOLD_FILE), ev.load_pred(PRED_FILE))
        score = iou_metric.calculate()
        self.assertIsInstance(score, dict)
        print(score)


if __name__ == '__main__':
    unittest.main()
    
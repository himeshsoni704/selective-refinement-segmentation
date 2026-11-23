import os
import unittest

import torch

try:
    from selective_seg.yolo_regions import FrozenProposer, ProposalReader
except ImportError:
    ProposalReader = None


class FakeProposer:
    channels = 3

    def __init__(self, empty=False):
        self.empty = empty

    def __call__(self, view, origin, extent, image_hw):
        n = 0 if self.empty else 1
        y, x = origin
        h, w = extent
        return (torch.ones(n,3), torch.tensor([[x,y,x+w,y+h]])[:n].float(),
                torch.ones(n))


@unittest.skipIf(ProposalReader is None, "optional torchvision unavailable")
class ProposalTests(unittest.TestCase):
    def test_scene_cache_and_training(self):
        reader = ProposalReader(FakeProposer(), 6)
        cache = {}
        image = torch.rand(3,80,96)
        for x in (0,32):
            local = torch.randn(1,6,32,32)
            summary = reader(image,(0,x,32,32),("scene",),local,cache)
            summary.sum().backward()
        self.assertEqual(cache["yolo_calls"],3)
        self.assertEqual(cache["yolo_rois"],3)
        self.assertIsNotNone(reader.relevance[-1].weight.grad)

    def test_empty_roi(self):
        reader = ProposalReader(FakeProposer(True),6)
        cache = {}
        out = reader(torch.rand(3,64,64),(0,0,32,32),("large",),torch.rand(1,6,32,32),cache)
        torch.testing.assert_close(out,torch.zeros(1,24))
        self.assertEqual(cache["yolo_empty"],2)

    @unittest.skipUnless(os.environ.get("YOLO_TEST_WEIGHTS"),"requires YOLOv8 checkpoint")
    def test_real_detect_head(self):
        detector = FrozenProposer(os.environ["YOLO_TEST_WEIGHTS"],torch.device("cpu"),0,3)
        features, boxes, scores = detector(torch.rand(1,3,160,160),(-16,-20),(160,160),(120,130))
        self.assertGreater(len(scores),0)
        self.assertLessEqual(len(scores),3)
        self.assertEqual(features.shape,(len(scores),detector.channels))
        self.assertTrue((boxes[:,0::2]>=0).all() and (boxes[:,0::2]<=130).all())
        self.assertTrue((boxes[:,1::2]>=0).all() and (boxes[:,1::2]<=120).all())
        self.assertFalse(features.requires_grad)
        detector.confidence=1
        features, _, _ = detector(torch.zeros(1,3,160,160),(0,0),(160,160),(160,160))
        self.assertEqual(features.shape,(0,detector.channels))

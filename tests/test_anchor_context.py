import itertools
import unittest

import numpy as np
import torch

from selective_seg.anchor_context import ContextReader, allocate, crop, window


class AnchorTests(unittest.TestCase):
    def test_centres_and_reconstruction(self):
        x = torch.arange(37*43,dtype=torch.float32).reshape(1,1,37,43)
        for tile in [(4,5,16,16),(30,39,7,4),(0,0,15,13)]:
            for side in (16,24,64):
                region = crop(x,tile,side)
                y,x0 = window(tile, side)
                ty,tx,h,w = tile
                self.assertEqual(region.shape[-2:], (side,side))
                torch.testing.assert_close(region[...,ty-y:ty-y+h,tx-x0:tx-x0+w], x[...,ty:ty+h,tx:tx+w])

    def test_allocator_with_negative_gain(self):
        gains = np.array([[0,2,5],[0,-2,5],[0,0,8.]])
        costs = np.array([0,2.1,5.2])
        for cap in range(16):
            options = allocate(gains,costs,cap,[True,False,True])
            objective = lambda a: sum(gains[i,j] for i,j in enumerate(a))
            feasible = [a for a in itertools.product(range(3), repeat=3)
                        if a[1]==0 and sum(np.ceil(costs[j]) for j in a)<=cap]
            self.assertEqual(objective(options),max(map(objective,feasible)))

    def test_scene_encoded_once_and_keep_geometry(self):
        reader = ContextReader(6)
        calls = []
        hook = reader.encoder.register_forward_hook(lambda *args: calls.append(1))
        image, cache = torch.rand(3,37,43), {}
        for tile in [(0,0,16,16),(30,39,7,4)]:
            logits = torch.randn(1,6,*tile[2:])
            torch.testing.assert_close(reader(logits,image,tile,"scene",cache),logits)
        hook.remove()
        self.assertEqual(len(calls),1)

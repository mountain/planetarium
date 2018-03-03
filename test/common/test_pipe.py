import unittest

import flare.pipe as p
import flare.dataset.decorators as d


class TestPipe(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_roll(self):
        elevens = list(range(11))
        self.assertListEqual([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]], list(p.roll(elevens, window_size=1)))
        self.assertListEqual([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10]], list(p.roll(elevens, window_size=2)))
        self.assertListEqual([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10]], list(p.roll(elevens, window_size=3)))
        self.assertListEqual([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9], [7, 8, 9, 10]], list(p.roll(elevens, window_size=4)))
        self.assertListEqual([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7], [4, 5, 6, 7, 8], [5, 6, 7, 8, 9], [6, 7, 8, 9, 10]], list(p.roll(elevens, window_size=5)))

    def test_batches(self):
        elevens = list(range(11))
        self.assertListEqual([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]], list(p.batches(elevens, batch_size=1)))
        self.assertListEqual([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], list(p.batches(elevens, batch_size=2)))
        self.assertListEqual([[0, 1, 2], [3, 4, 5], [6, 7, 8]], list(p.batches(elevens, batch_size=3)))
        self.assertListEqual([[0, 1, 2, 3], [4, 5, 6, 7]], list(p.batches(elevens, batch_size=4)))
        self.assertListEqual([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], list(p.batches(elevens, batch_size=5)))

    def test_rolled_batches(self):
        elevens = list(range(11))
        self.assertListEqual([[[0, 1, 2], [1, 2, 3], [2, 3, 4]], [[3, 4, 5], [4, 5, 6], [5, 6, 7]], [[6, 7, 8], [7, 8, 9], [8, 9, 10]]], list(p.batches(p.roll(elevens, window_size=3), batch_size=3)))

    def test_batched_roll(self):
        elevens = list(range(11))
        self.assertListEqual([[[0, 1, 2], [3, 4, 5]], [[3, 4, 5], [6, 7, 8]]], list(p.roll(p.batches(elevens, batch_size=3), window_size=2)))

    def test_attributes(self):

        @d.attributes('x1', 'x2', 'x3')
        def datagen():
            elevens = list(range(11))
            return p.batches(elevens, batch_size=3)

        self.assertListEqual([{'x1': 0, 'x2': 1, 'x3': 2}, {'x1': 3, 'x2': 4, 'x3': 5}, {'x1': 6, 'x2': 7, 'x3': 8}], [d for d in datagen()])

    def test_feature(self):

        def add(x1=0, x2=0):
            return [x1 + x2]

        @d.feature(add, ['x1', 'x2'], ['a'])
        @d.attributes('x1', 'x2', 'x3')
        def datagen():
            elevens = list(range(11))
            return p.batches(elevens, batch_size=3)

        self.assertListEqual([{'x1': 0, 'x2': 1, 'x3': 2, 'a': 1}, {'x1': 3, 'x2': 4, 'x3': 5, 'a': 7}, {'x1': 6, 'x2': 7, 'x3': 8, 'a': 13}], [d for d in datagen()])

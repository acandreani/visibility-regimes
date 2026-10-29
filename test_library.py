import unittest
import numpy as np
from visibility_regimes import augment, metrics


class LibraryTests(unittest.TestCase):
    def test_support(self):
        for w in (4, 6, 8, 24, 144):
            x = np.random.default_rng(w).normal(size=(3,w,2))
            z = augment(x)
            self.assertEqual(z.shape, (3,w+4,2))
            np.testing.assert_array_equal(z[:,:w], x)
            changed = x.copy()
            if w > 8:
                changed[:,:-8] += 100
            np.testing.assert_array_equal(augment(changed)[:,-4:], z[:,-4:])

    def test_signed_strengths(self):
        x = np.array([[[1.], [2.], [4.], [1.]]])
        np.testing.assert_array_equal(augment(x)[0,-4:,0], [-3,2,3,4])

    def test_metrics(self):
        self.assertEqual(metrics([0,1], [0,1])['macro_f1'], 1)
        with self.assertRaises(ValueError):
            augment(np.zeros((2,3,1)))


if __name__ == '__main__':
    unittest.main()

import unittest

from umt.umt_utils import plot_colors


class test_utils(unittest.TestCase):
    def test_colors(self):
        colors = plot_colors()
        self.assertEqual(colors.shape[0], 32)
        self.assertEqual(colors.shape[1], 3)

if __name__ == "__main__":
    unittest.main()

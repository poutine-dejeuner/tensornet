import unittest as ut
from tensornet.utils import tensor_tree_node_init

class Test_tensor_tree_node_init(ut.TestCase):
    def test_shape(self):
        tensor = tensor_tree_node_init((4,4,4,4), std = 1e-8)
        self.assertTrue(tensor.shape == (4,4,4,4))

if __name__ == '__main__':
    ut.main()
import unittest
import torch
from transformer import PositonalEncoder, MultiHeadAttention, FeedForward


class TestPositonalEncoder(unittest.TestCase):
    def setUp(self):
        self.d_model = 512
        self.seq_len = 80
        self.batch_size = 64

        self.positional_encoder = PositonalEncoder(self.d_model, self.seq_len)
        self.input_data = torch.rand(self.batch_size, self.seq_len, self.d_model)

    def test_positional_encoder(self):
        output = self.positional_encoder(self.input_data)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))

class TestMultiHeadAttention(unittest.TestCase):
    def setUp(self):
        self.d_model = 512
        self.seq_len = 80
        self.batch_size = 64
        self.heads = 8
        self.dropout = 0.1

        self.multi_head_attention = MultiHeadAttention(self.heads, self.d_model, self.dropout)
        self.input_data = torch.rand(self.batch_size, self.seq_len, self.d_model)

    def test_multi_head_attention(self):
        output = self.multi_head_attention(self.input_data, self.input_data, self.input_data)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))

class TestFeedForward(unittest.TestCase):
    def setUp(self):
        self.d_model = 512
        self.d_ff = 2048
        self.batch_size = 64
        self.seq_len = 80
        self.dropout = 0.1

        self.feed_forward = FeedForward(self.d_model, self.d_ff, self.dropout)
        self.input_data = torch.rand(self.batch_size, self.seq_len, self.d_model)

    def test_feed_forward(self):
        output = self.feed_forward(self.input_data)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))


if __name__ == '__main__':
    unittest.main()
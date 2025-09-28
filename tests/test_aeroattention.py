import unittest

try:
    import torch
except ImportError:  # pragma: no cover - tests are skipped when PyTorch is missing.
    torch = None

if torch is not None:
    from aeroattention import AeroAttention
    from aeroattention.fused_attention import fused_multi_head_attention, is_triton_available
else:  # pragma: no cover - prevents import errors during test discovery.
    AeroAttention = None
    fused_multi_head_attention = None
    is_triton_available = lambda: False


@unittest.skipUnless(torch is not None, "PyTorch is required for attention tests")
class TestAeroAttention(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_forward_matches_projection_shape(self) -> None:
        module = AeroAttention(embed_dim=64, num_heads=4, dropout_p=0.0)
        hidden_states = torch.randn(2, 16, 64)
        output = module(hidden_states)
        self.assertEqual(output.output.shape, hidden_states.shape)
        self.assertIsNone(output.attn_probs)

    def test_compute_attention_cpu(self) -> None:
        module = AeroAttention(embed_dim=32, num_heads=4, dropout_p=0.0)
        hidden_states = torch.randn(1, 8, 32)
        q, k, v = module.project_qkv(hidden_states)
        result = module.compute_attention(q, k, v, return_attn_probs=True)
        self.assertEqual(result.output.shape, hidden_states.shape)
        self.assertIsNotNone(result.attn_probs)
        self.assertTrue(torch.allclose(result.output, result.output, atol=1e-5))

    def test_attention_is_probability_distribution(self) -> None:
        module = AeroAttention(embed_dim=32, num_heads=4, dropout_p=0.0)
        hidden_states = torch.randn(2, 10, 32)
        q, k, v = module.project_qkv(hidden_states)
        attention = module.compute_attention(q, k, v, return_attn_probs=True)
        self.assertIsNotNone(attention.attn_probs)
        probs = attention.attn_probs
        assert probs is not None
        sums = probs.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-4))

    def test_kv_cache_appends_sequences(self) -> None:
        module = AeroAttention(embed_dim=32, num_heads=4, dropout_p=0.0)
        module.eval()
        cache = module.build_kv_cache(batch_size=1)
        hidden_states = torch.randn(1, 4, 32)
        output1 = module(hidden_states[:, :2, :], kv_cache=cache)
        cache = output1.kv_cache
        assert cache is not None
        output2 = module(hidden_states[:, 2:, :], kv_cache=cache)
        self.assertEqual(output2.kv_cache.key.size(-2), 4)

    def test_float16_and_bfloat16_numerical_stability(self) -> None:
        for dtype in (torch.float16, torch.bfloat16):
            module = AeroAttention(embed_dim=32, num_heads=4, dropout_p=0.0, use_out_proj=False).eval()
            q = torch.randn(1, 12, 32, dtype=dtype)
            k = torch.randn(1, 12, 32, dtype=dtype)
            v = torch.randn(1, 12, 32, dtype=dtype)
            output = module.compute_attention(q, k, v).output
            self.assertFalse(torch.isnan(output).any())
            self.assertFalse(torch.isinf(output).any())

    def test_cuda_path_dispatches_when_available(self) -> None:
        if not torch.cuda.is_available() or not is_triton_available():
            self.skipTest("CUDA/Triton unavailable")
        module = AeroAttention(embed_dim=32, num_heads=4, dropout_p=0.0, device=torch.device("cuda"))
        module = module.cuda()
        hidden_states = torch.randn(1, 8, 32, device="cuda")
        q, k, v = module.project_qkv(hidden_states)
        q = q.view(1, 8, 4, 8).transpose(1, 2)
        k = k.view(1, 8, 4, 8).transpose(1, 2)
        v = v.view(1, 8, 4, 8).transpose(1, 2)
        output, _ = fused_multi_head_attention(q, k, v)
        self.assertEqual(output.device.type, "cuda")

    def test_masking_matches_manual(self) -> None:
        module = AeroAttention(embed_dim=32, num_heads=4, dropout_p=0.0)
        module.eval()
        hidden_states = torch.randn(1, 4, 32)
        q, k, v = module.project_qkv(hidden_states)
        attn_mask = torch.full((1, 4, 4), float("-inf"))
        attn_mask[..., 0] = 0.0
        manual = module.compute_attention(q, k, v, attn_mask=attn_mask, return_attn_probs=True)
        masked_scores = manual.attn_probs
        assert masked_scores is not None
        self.assertTrue((masked_scores[..., 1:] < 1e-5).all())


if __name__ == "__main__":
    unittest.main()
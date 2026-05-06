"""Unit tests for DraftArtifacts storage and per-request persistence."""

import unittest

import torch

from sglang.srt.speculative.eagle_info import DraftArtifacts


class TestDraftArtifacts(unittest.TestCase):
    """Test DraftArtifacts dataclass creation and per-request slicing."""

    def test_create_draft_artifacts(self):
        """DraftArtifacts can be created with valid tensors."""
        topk = 4
        num_draft_tokens = 5  # speculative_num_draft_tokens
        hidden_size = 16
        num_steps = 3

        # Simulate output from draft_forward for a single request
        artifacts = DraftArtifacts(
            parent_list=torch.arange((num_steps - 1) * topk),
            top_scores_index=torch.arange(num_draft_tokens - 1),
            draft_tokens=torch.randint(0, 1000, (num_draft_tokens - 1,)),
            verified_id=torch.tensor(42, dtype=torch.int32),
            topk_p=torch.rand(topk),
            topk_index=torch.randint(0, 32000, (topk,)),
            hidden_states=torch.rand(hidden_size),
        )

        self.assertEqual(artifacts.parent_list.shape, ((num_steps - 1) * topk,))
        self.assertEqual(artifacts.top_scores_index.shape, (num_draft_tokens - 1,))
        self.assertEqual(artifacts.draft_tokens.shape, (num_draft_tokens - 1,))
        self.assertEqual(artifacts.verified_id.shape, ())
        self.assertEqual(artifacts.topk_p.shape, (topk,))
        self.assertEqual(artifacts.topk_index.shape, (topk,))
        self.assertEqual(artifacts.hidden_states.shape, (hidden_size,))

    def test_batch_slicing(self):
        """Simulate slicing batch-level draft outputs to per-request artifacts."""
        batch_size = 3
        topk = 4
        num_draft_tokens = 5
        hidden_size = 16
        num_steps = 3

        # Simulate batch-level outputs from draft_forward
        parent_list = torch.rand(batch_size, (num_steps - 1) * topk)
        top_scores_index = torch.randint(0, 100, (batch_size, num_draft_tokens - 1))
        draft_tokens = torch.randint(0, 32000, (batch_size, num_draft_tokens - 1))
        verified_id = torch.randint(0, 32000, (batch_size,), dtype=torch.int32)
        topk_p = torch.rand(batch_size, topk)
        topk_index = torch.randint(0, 32000, (batch_size, topk))
        hidden_states = torch.rand(batch_size, hidden_size)

        # Slice per-request (same logic as _store_draft_artifacts)
        for i in range(batch_size):
            artifacts = DraftArtifacts(
                parent_list=parent_list[i],
                top_scores_index=top_scores_index[i],
                draft_tokens=draft_tokens[i],
                verified_id=verified_id[i],
                topk_p=topk_p[i],
                topk_index=topk_index[i],
                hidden_states=hidden_states[i],
            )
            self.assertEqual(artifacts.parent_list.shape, ((num_steps - 1) * topk,))
            self.assertEqual(artifacts.verified_id.shape, ())
            self.assertEqual(artifacts.topk_p.shape, (topk,))
            self.assertEqual(artifacts.hidden_states.shape, (hidden_size,))

    def test_empty_parent_list(self):
        """Handle single-step draft where parent_list is empty (shape (0,))."""
        batch_size = 2
        topk = 4
        num_draft_tokens = 5
        hidden_size = 8

        # Single step: parent_list is (batch_size, 0)
        parent_list = torch.empty(batch_size, 0)

        for i in range(batch_size):
            artifacts = DraftArtifacts(
                parent_list=parent_list[i],
                top_scores_index=torch.arange(num_draft_tokens - 1),
                draft_tokens=torch.randint(0, 1000, (num_draft_tokens - 1,)),
                verified_id=torch.tensor(42, dtype=torch.int32),
                topk_p=torch.rand(topk),
                topk_index=torch.randint(0, 32000, (topk,)),
                hidden_states=torch.rand(hidden_size),
            )
            self.assertEqual(artifacts.parent_list.shape, (0,))

    def test_round_trip_consistency(self):
        """Verify that sliced artifacts can be re-assembled to batch-level tensors."""
        batch_size = 4
        topk = 4
        num_draft_tokens = 5
        hidden_size = 16
        num_steps = 3

        # Original batch-level tensors
        orig_parent_list = torch.rand(batch_size, (num_steps - 1) * topk)
        orig_top_scores_index = torch.randint(0, 100, (batch_size, num_draft_tokens - 1))
        orig_draft_tokens = torch.randint(0, 32000, (batch_size, num_draft_tokens - 1))
        orig_verified_id = torch.randint(0, 32000, (batch_size,), dtype=torch.int32)
        orig_topk_p = torch.rand(batch_size, topk)
        orig_topk_index = torch.randint(0, 32000, (batch_size, topk))
        orig_hidden_states = torch.rand(batch_size, hidden_size)

        # Slice to per-request
        artifacts_list = []
        for i in range(batch_size):
            artifacts_list.append(
                DraftArtifacts(
                    parent_list=orig_parent_list[i].clone(),
                    top_scores_index=orig_top_scores_index[i].clone(),
                    draft_tokens=orig_draft_tokens[i].clone(),
                    verified_id=orig_verified_id[i].clone(),
                    topk_p=orig_topk_p[i].clone(),
                    topk_index=orig_topk_index[i].clone(),
                    hidden_states=orig_hidden_states[i].clone(),
                )
            )

        # Re-assemble to batch-level
        reassembled_parent_list = torch.stack([a.parent_list for a in artifacts_list])
        reassembled_top_scores_index = torch.stack([a.top_scores_index for a in artifacts_list])
        reassembled_draft_tokens = torch.stack([a.draft_tokens for a in artifacts_list])
        reassembled_verified_id = torch.stack([a.verified_id for a in artifacts_list])
        reassembled_topk_p = torch.stack([a.topk_p for a in artifacts_list])
        reassembled_topk_index = torch.stack([a.topk_index for a in artifacts_list])
        reassembled_hidden_states = torch.stack([a.hidden_states for a in artifacts_list])

        torch.testing.assert_close(reassembled_parent_list, orig_parent_list)
        torch.testing.assert_close(reassembled_top_scores_index, orig_top_scores_index)
        torch.testing.assert_close(reassembled_draft_tokens, orig_draft_tokens)
        torch.testing.assert_close(reassembled_verified_id, orig_verified_id)
        torch.testing.assert_close(reassembled_topk_p, orig_topk_p)
        torch.testing.assert_close(reassembled_topk_index, orig_topk_index)
        torch.testing.assert_close(reassembled_hidden_states, orig_hidden_states)


if __name__ == "__main__":
    unittest.main()

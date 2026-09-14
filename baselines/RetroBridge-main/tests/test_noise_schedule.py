import unittest

import torch

from src.frameworks.noise_schedule import PredefinedNoiseScheduleDiscrete


class PredefinedNoiseScheduleDiscreteTest(unittest.TestCase):
    def test_derived_schedules_are_non_persistent_buffers(self):
        schedule = PredefinedNoiseScheduleDiscrete('cosine', timesteps=20)

        buffers = dict(schedule.named_buffers())
        self.assertIn('alphas', buffers)
        self.assertIn('alphas_bar', buffers)
        self.assertEqual(set(schedule.state_dict()), {'betas'})

        # Legacy checkpoints contain only betas and must still load strictly.
        schedule.load_state_dict({'betas': schedule.betas.clone()}, strict=True)
        times = torch.tensor([[0.0], [0.5], [1.0]])
        values = schedule.get_alpha_bar(t_normalized=times)
        self.assertEqual(values.device, times.device)
        self.assertTrue(torch.isfinite(values).all())


if __name__ == '__main__':
    unittest.main()

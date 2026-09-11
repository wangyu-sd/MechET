from collections import Counter

from mechet.mixed_horizon import CurriculumController, MixedHorizonSampler


def test_sampler_is_reproducible_and_prefix_is_valid():
    first = MixedHorizonSampler(seed=17, phase="A")
    second = MixedHorizonSampler(seed=17, phase="A")
    left = [
        first.sample(identifier="rxn", n_events=8, draw_index=index)
        for index in range(100)
    ]
    right = [
        second.sample(identifier="rxn", n_events=8, draw_index=index)
        for index in range(100)
    ]
    assert left == right
    assert all(0 <= item.prefix_events < 8 for item in left)
    assert all(item.remaining_gold_events == 8 - item.prefix_events for item in left)


def test_phase_a_has_all_horizons_and_expected_ordering():
    sampler = MixedHorizonSampler(seed=17, phase="A")
    counts = Counter(
        sampler.sample(identifier="rxn", n_events=8, draw_index=index).horizon
        for index in range(2000)
    )
    assert counts["near_end"] > counts["middle"] > counts["product"]


def test_curriculum_promotions_do_not_reset_sampler_state():
    controller = CurriculumController()
    assert controller.phase == "A"
    assert controller.observe(near_end_exact_rate=0.59) == "A"
    assert controller.observe(near_end_exact_rate=0.60) == "B"
    assert controller.observe(product_monitor_exact_rate=0.01) == "B"
    assert controller.observe(product_monitor_exact_rate=0.01) == "C"

from scripts.analyze_system_one_target_focus_pilot import trajectory_signature


def test_trajectory_comparison_ignores_prompt_token_and_logit_changes():
    old = {"actions": [{"action": "apply_electron_flow", "accepted": True,
                        "selected_pairs": [4], "state_after": "C.O",
                        "route_input_tokens": 100, "route_logits": [1.0, 0.0]}]}
    new = {"actions": [{"action": "apply_electron_flow", "accepted": True,
                        "selected_pairs": [4], "state_after": "C.O",
                        "route_input_tokens": 90, "route_logits": [0.8, 0.2]}]}
    assert trajectory_signature(old) == trajectory_signature(new)
    new["actions"][0]["selected_pairs"] = [5]
    assert trajectory_signature(old) != trajectory_signature(new)

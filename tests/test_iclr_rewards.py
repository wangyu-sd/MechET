from mechet.iclr_rewards import compute_core_proof_reward

PROOF = '''<proof>
MECH_PROOF v1
TARGET_SMILES "[CH3:1][OH:2]"
ROOT s0
  IMPORT "[Br-:3]"
  IMPORT "[Na+:4]"
PRECURSOR_STATE s1
EDGE s0 s1
  BOND 1 2 -1
  BOND 1 3 +1
  LP 2 +2
  LP 3 -2
  CHARGE 2 0 -1
  CHARGE 3 -1 0
</proof>'''


def test_core_reward_ignores_free_spectator():
    row = {
        "messages": [
            {"role": "user", "content": "TARGET: [CH3:1][OH:2]"},
            {"role": "assistant", "content": PROOF},
        ],
        "metadata": {
            "derived_precursor": "[CH3:1][Br:3].[OH-:2].[Na+:4]",
        },
    }
    score = compute_core_proof_reward(row, PROOF)
    assert score["execute_ok"]
    assert score["endpoint_core_exact"]
    assert score["composition_match"]
    assert "[Na+:4]" not in score["derived_core_precursor"]

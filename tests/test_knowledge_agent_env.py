import hashlib
import json

from mechet.knowledge_agent_env import KnowledgeAgentConfig, KnowledgeAugmentedAgentEnv
from mechet.textbook_store import TextbookPassage, TextbookStore


def write_corpus(path):
    text = (
        "A nucleophile donates an electron pair to the electrophilic carbonyl carbon. "
        "The carbon oxygen pi bond shifts toward oxygen, while steric and electronic effects may alter competing pathways."
    )
    store = TextbookStore(
        [
            TextbookPassage(
                passage_id="carbonyl-1",
                title="Nucleophilic carbonyl addition",
                text=text,
                source_id="open_textbook",
                locator="chapter/carbonyl",
                revision="r1",
                license="CC-BY-4.0",
                source_url="https://example.org/carbonyl",
                evidence_sha256=hashlib.sha256(text.encode()).hexdigest(),
                topics=("carbonyl", "addition"),
            )
        ]
    )
    store.save(path)


def test_textbook_retrieval_is_soft_logged_evidence(tmp_path):
    corpus = tmp_path / "passages.jsonl"
    write_corpus(corpus)
    config = KnowledgeAgentConfig(
        textbook_corpus_path=str(corpus),
        require_textbook_corpus=True,
        max_tool_calls=8,
    )
    env = KnowledgeAugmentedAgentEnv(config=config)
    observation = json.loads(
        env.reset(target_smiles="[CH3:1][C:2](=[O:3])[CH3:4]")
    )
    assert observation["knowledge"]["textbook_enabled"]
    reward_before = env.reward
    result = json.loads(env.retrieve_textbook_guidance("carbonyl nucleophilic attack"))
    assert result["ok"]
    assert result["matches"][0]["passage"]["passage_id"] == "carbonyl-1"
    assert result["direct_reward"] is False
    assert env.reward == reward_before
    assert env.state_dict()["textbook_retrievals"]


def test_auto_retrieval_is_reproducible_and_bounded(tmp_path):
    corpus = tmp_path / "passages.jsonl"
    write_corpus(corpus)
    config = KnowledgeAgentConfig(
        textbook_corpus_path=str(corpus),
        require_textbook_corpus=True,
        auto_textbook_on_reset=True,
        textbook_max_characters=800,
    )
    first = KnowledgeAugmentedAgentEnv(config=config)
    second = KnowledgeAugmentedAgentEnv(config=config)
    first_obs = json.loads(first.reset(target_smiles="[CH3:1][C:2](=[O:3])[CH3:4]"))
    second_obs = json.loads(second.reset(target_smiles="[CH3:1][C:2](=[O:3])[CH3:4]"))
    left = first_obs["initial_textbook_context"]
    right = second_obs["initial_textbook_context"]
    assert left["context_sha256"] == right["context_sha256"]
    assert left["n_characters"] <= 800


def test_missing_corpus_has_actionable_failure_when_optional(tmp_path):
    config = KnowledgeAgentConfig(
        textbook_corpus_path=str(tmp_path / "missing.jsonl"),
        require_textbook_corpus=False,
    )
    env = KnowledgeAugmentedAgentEnv(config=config)
    env.reset(target_smiles="[CH3:1][OH:2]")
    result = json.loads(env.retrieve_textbook_guidance())
    assert result["code"] == "TEXTBOOK_CORPUS_UNAVAILABLE"


def test_structured_anchor_tool_is_explicitly_disabled_by_default(tmp_path):
    corpus = tmp_path / "passages.jsonl"
    write_corpus(corpus)
    env = KnowledgeAugmentedAgentEnv(
        config=KnowledgeAgentConfig(
            textbook_corpus_path=str(corpus),
            require_textbook_corpus=True,
        )
    )
    env.reset(target_smiles="[CH3:1][OH:2]")
    result = json.loads(env.retrieve_primitives())
    assert result["code"] == "STRUCTURED_PRIMITIVES_DISABLED"


def test_free_form_proof_remains_disabled(tmp_path):
    corpus = tmp_path / "passages.jsonl"
    write_corpus(corpus)
    env = KnowledgeAugmentedAgentEnv(
        config=KnowledgeAgentConfig(textbook_corpus_path=str(corpus))
    )
    env.reset(target_smiles="[CH3:1][OH:2]")
    result = json.loads(env.submit_proof("invented"))
    assert result["code"] == "FREE_FORM_PROOF_DISABLED"

"""Fixed VALID next-action and genuinely autonomous main-product-only K1.

Only public messages cross the backend boundary. GT actions construct teacher
states and score outputs; they never seed closed-loop imports or stopping.
"""
from collections import Counter
import json

from mechet.agent_inference import parse_tool_calls
from mechet.endpoints import split_precursor_endpoints
from mechet.in_place_grounded_flow import mapped_atom_numbers
from scripts.build_mech31k_main_product_only_sft import same_chemical_state
from scripts.build_natural_language_event_sft import SYSTEM, TOOLS
from scripts.eval_natural_language_event_local import score_prediction
from scripts.run_natural_language_value_search import (
    Action, Node, execute, policy_prompt, product_only_private_state, visible,
)


def public_root(reference):
    target = visible(reference["target_smiles"])
    if "." in target:
        raise ValueError("PR96_NON_MAIN_ONLY_ROOT")
    state = product_only_private_state(target)
    return Node(target=target, state=state, next_map=max(mapped_atom_numbers(state)) + 1,
                visited={target})


def messages_for(node, compact_history):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": policy_prompt(
                node.target, node.state, include_inventory=True, actions=node.actions,
                compact_history=compact_history)}]


def parsed(output):
    if output.get("truncated"):
        return "", {}, "OUTPUT_TRUNCATED"
    try:
        calls = parse_tool_calls(output.get("raw", ""))
        if len(calls) != 1:
            return "", {}, f"EXPECTED_ONE_TOOL_CALL:{len(calls)}"
        return calls[0].name, calls[0].arguments, ""
    except Exception as exc:
        return "", {}, f"PARSE_ERROR:{type(exc).__name__}:{exc}"


def evaluate_reactions(references, state_rows, backend, *, compact_history,
                       max_decisions=64, journal=None):
    references = list(references)
    tasks, ground_states = [], {}
    for reference in references:
        rid = str(reference["source_id"])
        node = public_root(reference)
        ground_states[rid] = []
        for depth, row in enumerate(state_rows[rid]):
            if messages_for(node, False) != row["messages"][:2]:
                raise ValueError(f"PR96_GOLD_PUBLIC_PREFIX_MISMATCH:{rid}:{depth}")
            fn = row["messages"][2]["tool_calls"][0]["function"]
            child, error = execute(node, Action(fn["name"], fn["arguments"], "", 0, 0), max_imports=32)
            if child is None:
                raise ValueError(f"PR96_GOLD_NOT_EXECUTABLE:{rid}:{depth}:{error}")
            tasks.append({"reaction_id": rid, "depth": depth,
                          "decision_type": row["metadata"]["decision_type"],
                          "messages": messages_for(node, compact_history),
                          "private_state": node.state, "reference_successor": child.state,
                          "gold_name": fn["name"], "gold_arguments": fn["arguments"]})
            ground_states[rid].append((fn["name"], child.state))
            node = child
        if not node.terminal or not same_chemical_state(node.state, reference["full_precursor_state"]):
            raise ValueError(f"PR96_GOLD_ENDPOINT_MISMATCH:{rid}")

    counts, teacher, episodes, calls_journal = Counter(), [], [], []
    def record(value):
        calls_journal.append(value)
        if journal:
            journal(value)
    outputs = backend.generate([t["messages"] for t in tasks])
    if len(outputs) != len(tasks):
        raise ValueError("PR96_BACKEND_DROPPED_DECISIONS")
    for task, output in zip(tasks, outputs):
        name, arguments, error = parsed(output)
        counts["parse_total"] += 1
        counts["parse_correct"] += not bool(error)
        counts["truncated"] += bool(output.get("truncated"))
        score = score_prediction(task, predicted_name=name, predicted_arguments=arguments)
        tf_node = Node(target="", state=task["private_state"],
                       next_map=max(mapped_atom_numbers(task["private_state"])) + 1)
        tf_child, tf_error = (execute(tf_node, Action(name, arguments, "", 0, 0), max_imports=32)
                              if not error else (None, error))
        score["formal_execute"] = tf_child is not None
        if tf_error:
            score["execution_error"] = tf_error
        # Use exact component-wise stereochemical equivalence, not only one
        # canonical SMILES spelling of a symmetric stereoring.
        if task["decision_type"] == "event" and score["formal_execute"]:
            score["successor_chemical_exact"] = same_chemical_state(tf_child.state, task["reference_successor"])
        dtype = task["decision_type"]
        if dtype == "event":
            counts["flow_total"] += 1
            counts["flow_correct"] += score["successor_chemical_exact"]
        if dtype == "finish":
            counts["finish_total"] += 1
            counts["finish_correct"] += score["finish_exact"]
        if dtype == "import" and task["depth"] == 0:
            counts["root_import_total"] += 1
            counts["root_import_tool_correct"] += score["correct_tool"]
            counts["root_import_fragment_correct"] += score["import_fragment_exact"]
        value = {"mode": "teacher_forced", "reaction_id": task["reaction_id"],
                 "depth": task["depth"], "decision_type": dtype,
                 "input_messages": task["messages"], "output": output,
                 "predicted_name": name, "predicted_arguments": arguments,
                 "parse_error": error, "score": score,
                 "gt_name": task["gold_name"], "gt_arguments": task["gold_arguments"],
                 "state_before": task["private_state"], "reference_successor": task["reference_successor"]}
        teacher.append(value)
        record(value)

    nodes = {str(r["source_id"]): public_root(r) for r in references}
    errors, first_errors, first_names, rejected = {}, {}, {}, Counter()
    for depth in range(max_decisions):
        active = [(rid, n) for rid,n in nodes.items() if not n.terminal and rid not in errors]
        if not active:
            break
        prompts = [messages_for(n, compact_history) for _,n in active]
        outputs = backend.generate(prompts)
        if len(outputs) != len(active):
            raise ValueError("PR96_BACKEND_DROPPED_EPISODES")
        for (rid, node), prompt, output in zip(active, prompts, outputs):
            name, arguments, error = parsed(output)
            if depth == 0:
                first_names[rid] = name or "PARSE_ERROR"
            counts["parse_total"] += 1
            counts["parse_correct"] += not bool(error)
            counts["truncated"] += bool(output.get("truncated"))
            child = None
            if not error:
                child, error = execute(node, Action(name, arguments, output.get("raw", ""), 0,
                                                    len(output.get("generated_token_ids", []))), max_imports=32)
            if child is None:
                errors[rid] = error
                rejected[error] += 1
            else:
                nodes[rid] = child
            gt = ground_states[rid]
            differs = (depth >= len(gt) or name != gt[depth][0] or child is None
                       or not same_chemical_state(child.state, gt[depth][1]))
            if differs and rid not in first_errors:
                first_errors[rid] = depth
            record({"mode": "closed_loop", "reaction_id": rid, "depth": depth,
                    "input_messages": prompt, "output": output, "predicted_name": name,
                    "predicted_arguments": arguments, "state_before": node.state,
                    "state_after": child.state if child else node.state, "error": error,
                    "gt_used_for_generation_or_stopping": False})
    for reference in references:
        rid = str(reference["source_id"])
        node = nodes[rid]
        expected = reference["full_precursor_state"]
        core_gt = split_precursor_endpoints(expected, reference["target_smiles"]).structural
        core_pred = split_precursor_endpoints(node.state, public_root(reference).state).structural
        structural = node.terminal and same_chemical_state(core_pred, core_gt)
        full = node.terminal and same_chemical_state(node.state, expected)
        episode = {"reaction_id": rid, "target": node.target, "terminal": node.terminal,
                   "structural_exact": structural, "full_exact": full,
                   "predicted_endpoint": visible(node.state), "gt_endpoint": visible(expected),
                   "predicted_core": visible(core_pred) if core_pred else "",
                   "gt_core": visible(core_gt) if core_gt else "",
                   "first_error_depth": first_errors.get(rid),
                   "error_depth_definition": "first reference tool/chemical-state divergence, not proof of chemical implausibility",
                   "first_tool": first_names.get(rid, "NONE"), "actions": node.actions,
                   "error": errors.get(rid, "" if node.terminal else "DECISION_BUDGET_EXCEEDED")}
        episodes.append(episode)
        record({"mode": "episode_complete", **episode})
    ratio = lambda a,b: {"correct": int(counts[a]), "total": int(counts[b])}
    metrics = {"closed_loop_completed": len(episodes), "closed_loop_expected": len(references),
               "structural_hits": sum(e["structural_exact"] for e in episodes),
               "full_hits": sum(e["full_exact"] for e in episodes),
               "parse": ratio("parse_correct", "parse_total"),
               "root_import_tool": ratio("root_import_tool_correct", "root_import_total"),
               "root_import_fragment": ratio("root_import_fragment_correct", "root_import_total"),
               "tf_flow_successor": ratio("flow_correct", "flow_total"),
               "expected_tf_flow_decisions": counts["flow_total"],
               "tf_finish": ratio("finish_correct", "finish_total"),
               "root_predicted_tools": dict(Counter(first_names.values())),
               "truncated_calls": counts["truncated"], "gt_future_leak_detected": False,
               "rejected": dict(rejected), "max_decisions": max_decisions}
    return {"metrics": metrics, "teacher_forced": teacher, "episodes": episodes,
            "raw_calls": calls_journal}

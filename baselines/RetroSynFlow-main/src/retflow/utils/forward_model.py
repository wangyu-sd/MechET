from __future__ import print_function

import argparse
import codecs
import re

import onmt.inputters as inputters
import onmt.model_builder
import onmt.opts as opts
import torch
from onmt.translate.translator import Translator
from onmt.utils.misc import tile

from retflow import config


def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    https://github.com/pschwllr/MolecularTransformer/tree/master#pre-processing
    """
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    # assert smi == ''.join(tokens)
    if smi != "".join(tokens):
        print(smi, "".join(tokens))
    return " ".join(tokens)


def get_forward_model(n_best):
    model_path = config.get_models_directory() / "mol_former.pt"

    args = argparse.Namespace(
        models=[str(model_path)],
        n_best=n_best,
        src="input.txt",
        output="pred.txt",
        batch_size=128,
        replace_unk=True,
        max_length=200,
        fast=True,
        data_type="text",
        alpha=0.0,
        beta=-0.0,
        block_ngram_repeat=0,
        ignore_when_blocking=[],
        length_penalty="none",
        coverage_penalty="none",
        stepwise_penalty=False,
        beam_size=5,
        min_length=0,
        dump_beam="",
        verbose=False,
        report_bleu=False,
        gpu=0,
        sample_rate=16000,
        window_size=0.02,
        window_stride=0.01,
        window="hamming",
        image_channel_size=3,
        attn_debug=False,
    )
    return modified_build_translator(args, report_score=False)


def modified_build_translator(opt, report_score=True, logger=None, out_file=None):
    if out_file is None:
        out_file = codecs.open(opt.output, 'w+', 'utf-8')

    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]


    if len(opt.models) > 1:
        # use ensemble decoding if more than one model is specified
        fields, model, model_opt = \
            onmt.decoders.ensemble.load_test_model(opt, dummy_opt.__dict__)
    else:
        fields, model, model_opt = \
            onmt.model_builder.load_test_model(opt, dummy_opt.__dict__)

    scorer = onmt.translate.GNMTGlobalScorer(opt.alpha,
                                             opt.beta,
                                             opt.coverage_penalty,
                                             opt.length_penalty)

    kwargs = {k: getattr(opt, k)
              for k in ["beam_size", "n_best", "max_length", "min_length",
                        "stepwise_penalty", "block_ngram_repeat",
                        "ignore_when_blocking", "dump_beam", "report_bleu",
                        "data_type", "replace_unk", "gpu", "verbose", "fast",
                        "sample_rate", "window_size", "window_stride",
                        "window", "image_channel_size"]}

    translator = ModifiedTranslator(model, fields, global_scorer=scorer,
                            out_file=out_file, report_score=report_score,
                            copy_attn=model_opt.copy_attn, logger=logger,
                            **kwargs)
    return translator

class ModifiedTranslator(Translator):
    def _fast_translate_batch(self,
                batch,
                data,
                max_length,
                min_length=0,
                n_best=1,
                return_attention=False):
        # TODO: faster code path for beam_size == 1.

        # TODO: support these blacklisted features.
        assert data.data_type == 'text'
        assert not self.copy_attn
        assert not self.dump_beam
        assert not self.use_filter_pred
        assert self.block_ngram_repeat == 0
        assert self.global_scorer.beta == 0

        beam_size = self.beam_size
        batch_size = batch.batch_size
        vocab = self.fields["tgt"].vocab
        start_token = vocab.stoi[inputters.BOS_WORD]
        end_token = vocab.stoi[inputters.EOS_WORD]

        # Encoder forward.
        src = inputters.make_features(batch, 'src', data.data_type)
        _, src_lengths = batch.src
        enc_states, memory_bank, src_lengths \
            = self.model.encoder(src, src_lengths)
        dec_states = self.model.decoder.init_decoder_state(
            src, memory_bank, enc_states, with_cache=True)

        # Tile states and memory beam_size times.
        dec_states.map_batch_fn(
            lambda state, dim: tile(state, beam_size, dim=dim))
        memory_bank = tile(memory_bank, beam_size, dim=1)
        memory_lengths = tile(src_lengths, beam_size)

        batch_offset = torch.arange(
            batch_size, dtype=torch.long, device=memory_bank.device)
        beam_offset = torch.arange(
            0,
            batch_size * beam_size,
            step=beam_size,
            dtype=torch.long,
            device=memory_bank.device)
        alive_seq = torch.full(
            [batch_size * beam_size, 1],
            start_token,
            dtype=torch.long,
            device=memory_bank.device)
        alive_attn = None

        # Give full probability to the first beam on the first step.
        topk_log_probs = (
            torch.tensor([0.0] + [float("-inf")] * (beam_size - 1),
                        device=memory_bank.device).repeat(batch_size))

        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]  # noqa: F812

        results = {}
        results["predictions"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["scores"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["attention"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["gold_score"] = [0] * batch_size
        results["batch"] = batch

        for step in range(max_length):
            decoder_input = alive_seq[:, -1].view(1, -1, 1)

            # Decoder forward.
            dec_out, dec_states, attn = self.model.decoder(
                decoder_input,
                memory_bank,
                dec_states,
                memory_lengths=memory_lengths,
                step=step)

            # Generator forward.
            log_probs = self.model.generator.forward(dec_out.squeeze(0))
            vocab_size = log_probs.size(-1)

            if step < min_length:
                log_probs[:, end_token] = -1e20

            # Multiply probs by the beam probability.
            log_probs += topk_log_probs.view(-1).unsqueeze(1)

            alpha = self.global_scorer.alpha
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha

            # Flatten probs into a list of possibilities.
            curr_scores = log_probs / length_penalty
            curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
            topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)

            # Recover log probs.
            topk_log_probs = topk_scores * length_penalty

            # Resolve beam origin and true word ids.
            topk_beam_index = topk_ids.div(vocab_size)
            topk_ids = topk_ids.fmod(vocab_size)

            # Map beam_index to batch_index in the flat representation.
            batch_index = (
                    topk_beam_index
                    + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
            batch_index = batch_index.to(torch.long)
            select_indices = batch_index.view(-1).to(torch.long)

            # Append last prediction.
            alive_seq = torch.cat(
                [alive_seq.index_select(0, select_indices),
                topk_ids.view(-1, 1)], -1)
            if return_attention:
                current_attn = attn["std"].index_select(1, select_indices)
                if alive_attn is None:
                    alive_attn = current_attn
                else:
                    alive_attn = alive_attn.index_select(1, select_indices)
                    alive_attn = torch.cat([alive_attn, current_attn], 0)

            is_finished = topk_ids.eq(end_token)
            if step + 1 == max_length:
                is_finished.fill_(1)
            # End condition is top beam is finished.
            end_condition = is_finished[:, 0].eq(1)

            # Save finished hypotheses.
            if is_finished.any():
                predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                attention = (
                    alive_attn.view(
                        alive_attn.size(0), -1, beam_size, alive_attn.size(-1))
                    if alive_attn is not None else None)
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished[i].fill_(1)
                    finished_hyp = is_finished[i].nonzero().view(-1)
                    # Store finished hypotheses for this batch.
                    for j in finished_hyp:
                        hypotheses[b].append((
                            topk_scores[i, j],
                            predictions[i, j, 1:],  # Ignore start_token.
                            attention[:, i, j, :memory_lengths[i]]
                            if attention is not None else None))
                    # If the batch reached the end, save the n_best hypotheses.
                    if end_condition[i]:
                        best_hyp = sorted(
                            hypotheses[b], key=lambda x: x[0], reverse=True)
                        for n, (score, pred, attn) in enumerate(best_hyp):
                            if n >= n_best:
                                break
                            results["scores"][b].append(score)
                            results["predictions"][b].append(pred)
                            results["attention"][b].append(
                                attn if attn is not None else [])
                non_finished = end_condition.eq(0).nonzero().view(-1)
                # If all sentences are translated, no need to go further.
                if len(non_finished) == 0:
                    break
                # Remove finished batches for the next step.
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished) \
                    .view(-1, alive_seq.size(-1))
                if alive_attn is not None:
                    alive_attn = attention.index_select(1, non_finished) \
                        .view(alive_attn.size(0),
                            -1, alive_attn.size(-1))

            # Reorder states.
            select_indices = batch_index.view(-1)
            memory_bank = memory_bank.index_select(1, select_indices)
            memory_lengths = memory_lengths.index_select(0, select_indices)
            dec_states.map_batch_fn(
                lambda state, dim: state.index_select(dim, select_indices))

        return results
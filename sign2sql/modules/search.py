# coding: utf-8
import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np


# from onmt
def tile(x: Tensor, count: int, dim=0) -> Tensor:
    """
    Tiles x on dimension dim count times. From OpenNMT. Used for beam search.
    :param x: tensor to tile
    :param count: number of tiles
    :param dim: dimension along which the tensor is tiled
    :return: tiled tensor
    """
    if isinstance(x, tuple):
        h, c = x
        return tile(h, count, dim=dim), tile(c, count, dim=dim)

    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = (
        x.view(batch, -1)
        .transpose(0, 1)
        .repeat(count, 1)
        .transpose(0, 1)
        .contiguous()
        .view(*out_size)
    )
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


# pylint: disable=too-many-statements,too-many-branches
def beam_search(
    embed,
    pos_encoder,
    decoder,
    output_linear,
    size: int,
    bos_index: int,
    eos_index: int,
    pad_index: int,
    encoder_output,
    encoder_hidden,
    src_mask,
    max_output_length: int,
    alpha: float,
    output_size: int,
    n_best: int = 1,
) -> (np.array, np.array):
    """
    Beam search with size k.
    Inspired by OpenNMT-py, adapted for Transformer.
    In each decoding step, find the k most likely partial hypotheses.
    :param decoder:
    :param size: size of the beam
    :param bos_index:
    :param eos_index:
    :param pad_index:
    :param encoder_output:
    :param encoder_hidden:
    :param src_mask:
    :param max_output_length:
    :param alpha: `alpha` factor for length penalty
    :param embed:
    :param n_best: return this many hypotheses, <= beam (currently only 1)
    :return:
        - stacked_output: output hypotheses (2d array of indices),
        - stacked_attention_scores: attention scores (3d array)
    """
    assert size > 0, "Beam size must be >0."
    assert n_best <= size, "Can only return {} best hypotheses.".format(size)

    # init
    transformer = True
    batch_size = src_mask.size(0)
    att_vectors = None  # not used for Transformer

    # Recurrent models only: initialize RNN hidden state
    # pylint: disable=protected-access
    if not transformer:
        hidden = decoder._init_hidden(encoder_hidden)
    else:
        hidden = None

    # tile encoder states and decoder initial states beam_size times
    if hidden is not None:
        hidden = tile(hidden, size, dim=1)  # layers x batch*k x dec_hidden_size

    encoder_output = tile(
        encoder_output.contiguous(), size, dim=0
    )  # batch*k x src_len x enc_hidden_size
    src_mask = tile(src_mask, size, dim=0)  # batch*k x 1 x src_len

    # Transformer only: create target mask
    if transformer:
        trg_mask = src_mask.new_ones([1, 1, 1])  # transformer only
    else:
        trg_mask = None

    # numbering elements in the batch
    batch_offset = torch.arange(
        batch_size, dtype=torch.long, device=encoder_output.device
    )

    # numbering elements in the extended batch, i.e. beam size copies of each
    # batch element
    beam_offset = torch.arange(
        0, batch_size * size, step=size, dtype=torch.long, device=encoder_output.device
    )

    # keeps track of the top beam size hypotheses to expand for each element
    # in the batch to be further decoded (that are still "alive")
    alive_seq = torch.full(
        [batch_size * size, 1],
        bos_index,
        dtype=torch.long,
        device=encoder_output.device,
    )

    # Give full probability to the first beam on the first step.
    topk_log_probs = torch.zeros(batch_size, size, device=encoder_output.device)
    topk_log_probs[:, 1:] = float("-inf")

    # Structure that holds finished hypotheses.
    hypotheses = [[] for _ in range(batch_size)]

    results = {
        "predictions": [[] for _ in range(batch_size)],
        "scores": [[] for _ in range(batch_size)],
        "gold_score": [0] * batch_size,
    }

    for step in range(max_output_length):

        # This decides which part of the predicted sentence we feed to the
        # decoder to make the next prediction.
        # For Transformer, we feed the complete predicted sentence so far.
        # For Recurrent models, only feed the previous target word prediction
        if transformer:  # Transformer
            decoder_input = alive_seq  # complete prediction so far
        else:  # Recurrent
            decoder_input = alive_seq[:, -1].view(-1, 1)  # only the last word

        # expand current hypotheses
        # decode one single step
        # logits: logits for final softmax
        # pylint: disable=unused-variable
        trg_embed = pos_encoder(embed(decoder_input))
        # x, memory, src_mask, tgt_mask
        decoder_hidden = decoder(
            memory=encoder_output,
            # encoder_hidden=encoder_hidden,
            src_mask=src_mask.unsqueeze(-2),
            x=trg_embed,
            # hidden=hidden,
            # prev_att_vector=att_vectors,
            # unroll_steps=1,
            tgt_mask=None,  # subsequent mask for Transformer only
        )
        logits = output_linear(decoder_hidden)

        # For the Transformer we made predictions for all time steps up to
        # this point, so we only want to know about the last time step.
        if transformer:
            logits = logits[:, -1]  # keep only the last time step
            hidden = None  # we don't need to keep it for transformer

        # batch*k x trg_vocab
        log_probs = F.log_softmax(logits, dim=-1).squeeze(1)

        # multiply probs by the beam probability (=add logprobs)
        log_probs += topk_log_probs.view(-1).unsqueeze(1)
        curr_scores = log_probs.clone()

        # compute length penalty
        if alpha > -1:
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha
            curr_scores /= length_penalty

        # flatten log_probs into a list of possibilities
        curr_scores = curr_scores.reshape(-1, size * output_size)

        # pick currently best top k hypotheses (flattened order)
        topk_scores, topk_ids = curr_scores.topk(size, dim=-1)

        if alpha > -1:
            # recover original log probs
            topk_log_probs = topk_scores * length_penalty
        else:
            topk_log_probs = topk_scores.clone()

        # reconstruct beam origin and true word ids from flattened order
        topk_beam_index = topk_ids.div(output_size)
        topk_ids = topk_ids.fmod(output_size)

        # map beam_index to batch_index in the flat representation
        batch_index = topk_beam_index + beam_offset[
            : topk_beam_index.size(0)
        ].unsqueeze(1)
        select_indices = batch_index.view(-1).long()

        # append latest prediction
        alive_seq = torch.cat(
            [alive_seq.index_select(0, select_indices), topk_ids.view(-1, 1)], -1
        )  # batch_size*k x hyp_len

        is_finished = topk_ids.eq(eos_index)
        if step + 1 == max_output_length:
            is_finished.fill_(True)
        # end condition is whether the top beam is finished
        end_condition = is_finished[:, 0].eq(True)

        # save finished hypotheses
        if is_finished.any():
            predictions = alive_seq.view(-1, size, alive_seq.size(-1))
            for i in range(is_finished.size(0)):
                b = batch_offset[i]
                if end_condition[i]:
                    is_finished[i].fill_(True)
                finished_hyp = is_finished[i].nonzero().view(-1)
                # store finished hypotheses for this batch
                for j in finished_hyp:
                    # Check if the prediction has more than one EOS.
                    # If it has more than one EOS, it means that the prediction should have already
                    # been added to the hypotheses, so you don't have to add them again.
                    if (predictions[i, j, 1:] == eos_index).nonzero().numel() < 2:
                        hypotheses[b].append(
                            (
                                topk_scores[i, j],
                                predictions[i, j, 1:],
                            )  # ignore start_token
                        )
                # if the batch reached the end, save the n_best hypotheses
                if end_condition[i]:
                    best_hyp = sorted(hypotheses[b], key=lambda x: x[0], reverse=True)
                    for n, (score, pred) in enumerate(best_hyp):
                        if n >= n_best:
                            break
                        results["scores"][b].append(score)
                        results["predictions"][b].append(pred)
            non_finished = end_condition.eq(False).nonzero().view(-1)
            # if all sentences are translated, no need to go further
            # pylint: disable=len-as-condition
            if len(non_finished) == 0:
                break
            # remove finished batches for the next step
            topk_log_probs = topk_log_probs.index_select(0, non_finished)
            batch_index = batch_index.index_select(0, non_finished)
            batch_offset = batch_offset.index_select(0, non_finished)
            alive_seq = predictions.index_select(0, non_finished).view(
                -1, alive_seq.size(-1)
            )

        # reorder indices, outputs and masks
        select_indices = batch_index.view(-1).long()
        encoder_output = encoder_output.index_select(0, select_indices)
        src_mask = src_mask.index_select(0, select_indices)

        if hidden is not None and not transformer:
            if isinstance(hidden, tuple):
                # for LSTMs, states are tuples of tensors
                h, c = hidden
                h = h.index_select(1, select_indices)
                c = c.index_select(1, select_indices)
                hidden = (h, c)
            else:
                # for GRUs, states are single tensors
                hidden = hidden.index_select(1, select_indices)

        if att_vectors is not None:
            att_vectors = att_vectors.index_select(0, select_indices)

    def pad_and_stack_hyps(hyps, pad_value):
        filled = (
            np.ones((len(hyps), max([h.shape[0] for h in hyps])), dtype=int) * pad_value
        )
        for j, h in enumerate(hyps):
            for k, i in enumerate(h):
                filled[j, k] = i
        return filled

    # from results to stacked outputs
    assert n_best == 1
    # only works for n_best=1 for now
    final_outputs = pad_and_stack_hyps(
        [r[0].cpu().numpy() for r in results["predictions"]], pad_value=pad_index
    )

    return final_outputs, None


# pylint: disable=too-many-statements,too-many-branches
def beam_search_disentangled_decoder(
    embed,
    pos_encoder,
    decoder,
    output_linear,
    size: int,
    bos_index: int,
    eos_index: int,
    pad_index: int,
    encoder_output,
    encoder_hidden,
    src_mask,
    encoder_output2,
    encoder_hidden2,
    src_mask2,
    max_output_length: int,
    alpha: float,
    output_size: int,
    n_best: int = 1,
) -> (np.array, np.array):
    """
    Beam search with size k.
    Inspired by OpenNMT-py, adapted for Transformer.
    In each decoding step, find the k most likely partial hypotheses.
    :param decoder:
    :param size: size of the beam
    :param bos_index:
    :param eos_index:
    :param pad_index:
    :param encoder_output:
    :param encoder_hidden:
    :param src_mask:
    :param encoder_output2:
    :param encoder_hidden2:
    :param src_mask2:
    :param max_output_length:
    :param alpha: `alpha` factor for length penalty
    :param embed:
    :param n_best: return this many hypotheses, <= beam (currently only 1)
    :return:
        - stacked_output: output hypotheses (2d array of indices),
        - stacked_attention_scores: attention scores (3d array)
    """
    assert size > 0, "Beam size must be >0."
    assert n_best <= size, "Can only return {} best hypotheses.".format(size)

    # init
    transformer = True
    batch_size = src_mask.size(0) if src_mask is not None else src_mask2.size(0)
    att_vectors = None  # not used for Transformer
    device = encoder_output.device if encoder_output is not None else encoder_output2.device

    # Recurrent models only: initialize RNN hidden state
    # pylint: disable=protected-access
    if not transformer:  # not implement
        hidden = decoder._init_hidden(encoder_hidden)
    else:
        hidden = None

    # tile encoder states and decoder initial states beam_size times
    if hidden is not None:
        hidden = tile(hidden, size, dim=1)  # layers x batch*k x dec_hidden_size

    encoder_output = tile(
        encoder_output.contiguous(), size, dim=0
    ) if encoder_output is not None else None  # batch*k x src_len x enc_hidden_size
    src_mask = tile(src_mask, size, dim=0) if src_mask is not None else None  # batch*k x 1 x src_len

    encoder_output2 = tile(
        encoder_output2.contiguous(), size, dim=0
    ) if encoder_output2 is not None else None  # batch*k x src_len x enc_hidden_size
    src_mask2 = tile(src_mask2, size, dim=0) if src_mask2 is not None else None  # batch*k x 1 x src_len

    # Transformer only: create target mask
    if transformer:
        trg_mask = src_mask.new_ones([1, 1, 1]) if src_mask is not None else src_mask2.new_ones([1, 1, 1])  # transformer only
    else:
        trg_mask = None

    # numbering elements in the batch
    batch_offset = torch.arange(
        batch_size, dtype=torch.long, device=device
    )

    # numbering elements in the extended batch, i.e. beam size copies of each
    # batch element
    beam_offset = torch.arange(
        0, batch_size * size, step=size, dtype=torch.long, device=device
    )

    # keeps track of the top beam size hypotheses to expand for each element
    # in the batch to be further decoded (that are still "alive")
    alive_seq = torch.full(
        [batch_size * size, 1],
        bos_index,
        dtype=torch.long,
        device=device,
    )

    # Give full probability to the first beam on the first step.
    topk_log_probs = torch.zeros(batch_size, size, device=device)
    topk_log_probs[:, 1:] = float("-inf")

    # Structure that holds finished hypotheses.
    hypotheses = [[] for _ in range(batch_size)]

    results = {
        "predictions": [[] for _ in range(batch_size)],
        "scores": [[] for _ in range(batch_size)],
        "gold_score": [0] * batch_size,
    }

    for step in range(max_output_length):

        # This decides which part of the predicted sentence we feed to the
        # decoder to make the next prediction.
        # For Transformer, we feed the complete predicted sentence so far.
        # For Recurrent models, only feed the previous target word prediction
        if transformer:  # Transformer
            decoder_input = alive_seq  # complete prediction so far
        else:  # Recurrent
            decoder_input = alive_seq[:, -1].view(-1, 1)  # only the last word

        # expand current hypotheses
        # decode one single step
        # logits: logits for final softmax
        # pylint: disable=unused-variable
        trg_embed = pos_encoder(embed(decoder_input))
        # x, memory1, memory2, src_mask1, src_mask2, tgt_mask -> x, hidden_pair
        decoder_hidden, _ = decoder(
            memory1=encoder_output,
            # encoder_hidden=encoder_hidden,
            src_mask1=src_mask.unsqueeze(-2) if src_mask is not None else None,
            memory2=encoder_output2,
            # encoder_hidden=encoder_hidden,
            src_mask2=src_mask2.unsqueeze(-2) if src_mask2 is not None else None,
            x=trg_embed,
            # hidden=hidden,
            # prev_att_vector=att_vectors,
            # unroll_steps=1,
            tgt_mask=None,  # subsequent mask for Transformer only
        )
        logits = output_linear(decoder_hidden)

        # For the Transformer we made predictions for all time steps up to
        # this point, so we only want to know about the last time step.
        if transformer:
            logits = logits[:, -1]  # keep only the last time step
            hidden = None  # we don't need to keep it for transformer

        # batch*k x trg_vocab
        log_probs = F.log_softmax(logits, dim=-1).squeeze(1)

        # multiply probs by the beam probability (=add logprobs)
        log_probs += topk_log_probs.view(-1).unsqueeze(1)
        curr_scores = log_probs.clone()

        # compute length penalty
        if alpha > -1:
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha
            curr_scores /= length_penalty

        # flatten log_probs into a list of possibilities
        curr_scores = curr_scores.reshape(-1, size * output_size)

        # pick currently best top k hypotheses (flattened order)
        topk_scores, topk_ids = curr_scores.topk(size, dim=-1)

        if alpha > -1:
            # recover original log probs
            topk_log_probs = topk_scores * length_penalty
        else:
            topk_log_probs = topk_scores.clone()

        # reconstruct beam origin and true word ids from flattened order
        topk_beam_index = topk_ids.div(output_size)
        topk_ids = topk_ids.fmod(output_size)

        # map beam_index to batch_index in the flat representation
        batch_index = topk_beam_index + beam_offset[
            : topk_beam_index.size(0)
        ].unsqueeze(1)
        select_indices = batch_index.view(-1).long()

        # append latest prediction
        alive_seq = torch.cat(
            [alive_seq.index_select(0, select_indices), topk_ids.view(-1, 1)], -1
        )  # batch_size*k x hyp_len

        is_finished = topk_ids.eq(eos_index)
        if step + 1 == max_output_length:
            is_finished.fill_(True)
        # end condition is whether the top beam is finished
        end_condition = is_finished[:, 0].eq(True)

        # save finished hypotheses
        if is_finished.any():
            predictions = alive_seq.view(-1, size, alive_seq.size(-1))
            for i in range(is_finished.size(0)):
                b = batch_offset[i]
                if end_condition[i]:
                    is_finished[i].fill_(True)
                finished_hyp = is_finished[i].nonzero().view(-1)
                # store finished hypotheses for this batch
                for j in finished_hyp:
                    # Check if the prediction has more than one EOS.
                    # If it has more than one EOS, it means that the prediction should have already
                    # been added to the hypotheses, so you don't have to add them again.
                    if (predictions[i, j, 1:] == eos_index).nonzero().numel() < 2:
                        hypotheses[b].append(
                            (
                                topk_scores[i, j],
                                predictions[i, j, 1:],
                            )  # ignore start_token
                        )
                # if the batch reached the end, save the n_best hypotheses
                if end_condition[i]:
                    best_hyp = sorted(hypotheses[b], key=lambda x: x[0], reverse=True)
                    for n, (score, pred) in enumerate(best_hyp):
                        if n >= n_best:
                            break
                        results["scores"][b].append(score)
                        results["predictions"][b].append(pred)
            non_finished = end_condition.eq(False).nonzero().view(-1)
            # if all sentences are translated, no need to go further
            # pylint: disable=len-as-condition
            if len(non_finished) == 0:
                break
            # remove finished batches for the next step
            topk_log_probs = topk_log_probs.index_select(0, non_finished)
            batch_index = batch_index.index_select(0, non_finished)
            batch_offset = batch_offset.index_select(0, non_finished)
            alive_seq = predictions.index_select(0, non_finished).view(
                -1, alive_seq.size(-1)
            )

        # reorder indices, outputs and masks
        select_indices = batch_index.view(-1).long()
        encoder_output = encoder_output.index_select(0, select_indices) if encoder_output is not None else None
        src_mask = src_mask.index_select(0, select_indices) if src_mask is not None else None

        encoder_output2 = encoder_output2.index_select(0, select_indices) if encoder_output2 is not None else None
        src_mask2 = src_mask2.index_select(0, select_indices) if src_mask2 is not None else None

        if hidden is not None and not transformer:
            if isinstance(hidden, tuple):
                # for LSTMs, states are tuples of tensors
                h, c = hidden
                h = h.index_select(1, select_indices)
                c = c.index_select(1, select_indices)
                hidden = (h, c)
            else:
                # for GRUs, states are single tensors
                hidden = hidden.index_select(1, select_indices)

        if att_vectors is not None:
            att_vectors = att_vectors.index_select(0, select_indices)

    def pad_and_stack_hyps(hyps, pad_value):
        filled = (
            np.ones((len(hyps), max([h.shape[0] for h in hyps])), dtype=int) * pad_value
        )
        for j, h in enumerate(hyps):
            for k, i in enumerate(h):
                filled[j, k] = i
        return filled

    # from results to stacked outputs
    assert n_best == 1
    # only works for n_best=1 for now
    final_outputs = pad_and_stack_hyps(
        [r[0].cpu().numpy() for r in results["predictions"]], pad_value=pad_index
    )

    return final_outputs, None

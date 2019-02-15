#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from parlai.agents.transformer.modules import TransformerGeneratorModel


def universal_sentence_embedding(sentences, mask, sqrt=True):
    """
    Perform Universal Sentence Encoder averaging (https://arxiv.org/abs/1803.11175).

    This is really just sum / sqrt(len).

    :param Tensor sentences: an N x T x D of Transformer outputs. Note this is
        the exact output of TransformerEncoder, but has the time axis first
    :param ByteTensor: an N x T binary matrix of paddings

    :return: an N x D matrix of sentence embeddings
    :rtype Tensor:
    """
    # need to mask out the padded chars
    sentence_sums = th.bmm(
        sentences.permute(0, 2, 1),
        mask.float().unsqueeze(-1)
    ).squeeze(-1)
    divisor = mask.sum(dim=1).view(-1, 1).float()
    if sqrt:
        divisor = divisor.sqrt()
    sentence_sums /= divisor
    return sentence_sums


class EndToEndModel(TransformerGeneratorModel):
    def __init__(self, opt, dictionary):
        super().__init__(opt, dictionary)
        self.encoder = ContextKnowledgeEncoder(self.encoder)
        self.decoder = ContextKnowledgeDecoder(self.decoder)

    def reorder_encoder_states(self, encoder_out, new_order):
        raise NotImplementedError


class ContextKnowledgeEncoder(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        # The transformer_enc takes care of most of the work, but other modules
        # expect us to have an embeddings available
        self.embeddings = transformer.embeddings
        self.embed_dim = transformer.embeddings.embedding_dim
        self.transformer = transformer

    def forward(self, src_tokens, know_tokens, ck_mask, cs_ids, use_cs_ids=True):
        # encode the context, pretty basic
        context_encoded, context_mask = self.transformer(src_tokens)

        # make all the knowledge into a 2D matrix to encode
        N, K, Tk = know_tokens.size()
        know_flat = know_tokens.reshape(-1, Tk)
        know_encoded, know_mask = self.transformer(know_flat)

        # compute our sentence embeddings for context and knowledge
        context_use = universal_sentence_embedding(context_encoded, context_mask)
        know_use = universal_sentence_embedding(know_encoded, know_mask)

        # remash it back into the shape we need
        know_use = know_use.reshape(N, know_tokens.size(1), self.embed_dim)
        context_use /= np.sqrt(self.embed_dim)
        know_use /= np.sqrt(self.embed_dim)

        ck_attn = th.bmm(know_use, context_use.unsqueeze(-1)).squeeze(-1)
        # fill with near -inf
        ck_attn.masked_fill_(~ck_mask, -65504)

        if not use_cs_ids:
            # if we're not given the true chosen_sentence (test time), pick our
            # best guess
            _, cs_ids = ck_attn.max(1)

        # pick the true chosen sentence. remember that TransformerEncoder outputs
        #   (batch, time, embed)
        # but because know_encoded is a flattened, it's really
        #   (N * K, T, D)
        # We need to compute the offsets of the chosen_sentences
        cs_offsets = th.arange(N, device=cs_ids.device) * K + cs_ids
        cs_encoded = know_encoded[cs_offsets]
        # but padding is (N * K, T)
        cs_mask = know_mask[cs_offsets]

        # finally, concatenate it all
        full_enc = th.cat([context_encoded, cs_encoded], dim=1)
        full_mask = th.cat([context_mask, cs_mask], dim=1)

        # also return the knowledge selection mask for the loss
        return full_enc, full_mask, ck_attn


class ContextKnowledgeDecoder(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, input, encoder_state, incr_state=None):
        # our CK Encoder returns an extra output which the Transformer decoder
        # doesn't expect (the knowledge selection mask). Just chop it off
        encoder_output, encoder_mask, _ = encoder_state
        return self.transformer(input, (encoder_output, encoder_mask), incr_state)


class EndToEndCriterion(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.knowledge_alpha = opt['knowledge_alpha']

    def forward(self, model, sample):
        """Compute losses for a given sample."""
        # TODO: enable model to skip computations if knowledge or ranking loss are
        # not included
        net_output = model(**sample['net_input'])

        # maybe normalize gradients by # sentences instead of # tokens (default false)
        # TODO: ask myle & alex about this again
        nsentences = sample['target'].size(0)
        sample_size = nsentences if self.args.sentence_avg else sample['ntokens']

        # generative loss, copied over from fairseq's cross_entropy criterion
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        gen_loss = F.nll_loss(
            lprobs, target, size_average=False, ignore_index=self.padding_idx,
            reduce=True
        )
        nll_loss = gen_loss / sample['ntokens'] / np.log(2)

        # knowledge loss, cross entropy over the attn
        ctx_know_attn = net_output[-1]
        ctx_know_targets = sample['net_input']['cs_ids']
        know_loss = F.cross_entropy(
            ctx_know_attn.float(),  # already logits
            ctx_know_targets,
            reduce=True,
            size_average=True,
        )

        _, know_pred = ctx_know_attn.max(1)
        # for just reporting
        know_acc = (know_pred == ctx_know_targets).float().mean().item()
        know_chance = (sample['net_input']['ck_mask']
                       .sum(1).float().reciprocal().mean().item())

        # aggregate all the losses together
        if self.knowledge_alpha == 0.0:
            loss = gen_loss
        elif self.knowledge_alpha == 1.0:
            loss = know_loss
        else:
            loss = (
                (1 - self.knowledge_alpha) * gen_loss +
                self.knowledge_alpha * know_loss
            )

        logging_output = {
            'loss': loss.item(),
            # nll loss is always per token
            'nll_loss': nll_loss.item(),
            'know_loss': know_loss.item(),
            'ntokens': sample['ntokens'],
            'nsentences': nsentences,
            'sample_size': sample_size,
            'know_acc': know_acc,
            'know_chance': know_chance,
            'know_pred': know_pred,
        }

        return loss, sample_size, logging_output

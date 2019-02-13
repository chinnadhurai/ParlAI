import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from fairseq.utils import fill_with_neg_inf

from fairseq.models import (
    FairseqModel,
    FairseqEncoder,
    FairseqDecoder,
)

from fairseq.criterions import (
    FairseqCriterion,
    register_criterion,
)
from fairseq.models.transformer import (
    TransformerEncoder,
    TransformerDecoderLayer,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    Embedding,
    base_architecture,
)


def build_embedding(dictionary, embed_dim):
    """
    Create and randomly initialize an embedding matrix.

    Warning: this uses special initalization from the fairseq module!
    """
    return Embedding(len(dictionary), embed_dim, dictionary.pad())


def maybe(pydict, key):
    """
    Return pydict[key] if pydict is not None, or just return None
    """
    if pydict is None:
        return None
    return pydict[key]


class TransformerFFN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.fc1 = nn.Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = nn.Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.ln = nn.LayerNorm(self.embed_dim)
        # do the LN process the same
        self.normalize_before = args.encoder_normalize_before
        self.relu_dropout = args.relu_dropout
        self.dropout = args.dropout

    def forward(self, x):
        # keep the residual around
        # flatten to apply same transformation everywhere
        shape = x.size()
        x = x.view(-1, self.embed_dim)
        residual = x
        # apply transformation
        if self.normalize_before:
            x = self.ln(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.ln(x)
        # project back to the original shape
        x = x.view(shape)
        return x


def universal_sentence_embedding(sentences, padding_mask, cosine=False, sqrt=True):
    """
    Perform Universal Sentence Encoder averaging (https://arxiv.org/abs/1803.11175).

    This is really just sum / sqrt(len).

    :param Tensor sentences: an T x N x D of Transformer outputs. Note this is
        the exact output of TransformerEncoder, but has the time axis first
    :param ByteTensor: an N x T binary matrix of paddings

    :return: an D x N matrix of sentence embeddings
    :rtype Tensor:
    """
    # a None padding mask is equivalent to everything being available
    if padding_mask is None:
        T, N, D = sentences.shape
        mask = sentences.new(N, T).fill_(1)
    else:
        mask = (~padding_mask).type_as(sentences)
    # need to mask out the padded chars
    sentence_sums = th.bmm(sentences.permute(1, 2, 0), mask.unsqueeze(-1)).squeeze(-1)
    divisor = mask.sum(dim=1).view(-1, 1)
    if sqrt:
        divisor = divisor.sqrt()
    sentence_sums /= divisor
    if cosine:
        return sentence_sums.renorm(2, -1, 1)
    return sentence_sums


class _MyTransformerEncoder(TransformerEncoder):
    """Annoying wrapper necessary to get around a weird quirk of serializing."""
    def upgrade_state_dict(self, state_dict):
        # Overrides the normal TransformerEncoder upgrade_state_dict to
        # keep it from adding an extra key, which frustrates torch.load.
        return state_dict


class ContextKnowledgeEncoder(FairseqEncoder):
    def __init__(self, args, dictionary, embed_tokens, left_pad=True):
        # As a base we want a regular model
        super().__init__(dictionary)
        self.args = args

        # TODO: make this a command line argument
        self.use_addl_resid = args.use_addl_resid

        # The transformer_enc takes care of most of the work, but other modules
        # expect us to have an embed_tokens available
        self.embed_tokens = embed_tokens
        self.embed_dim = args.encoder_embed_dim

        # build the transformer
        self.transformer_enc = _MyTransformerEncoder(
            args, dictionary, embed_tokens, left_pad=left_pad
        )
        if self.args.shared_encoder:
            self.knowledge_enc = self.transformer_enc
        else:
            self.knowledge_enc = _MyTransformerEncoder(
                args, dictionary, embed_tokens, left_pad=left_pad
            )

        if self.use_addl_resid:
            # also need our knowledge transformation layer
            self.know_ffn = TransformerFFN(args)

    def forward(self, src_tokens, src_lengths, know_tokens, know_lengths,
                ck_mask, cs_ids, use_cs_ids=True):
        # encode the context, pretty basic
        context_encoded = self.transformer_enc(src_tokens, src_lengths)

        # make all the knowledge into a 2D matrix to encode
        N, K, Tk = know_tokens.size()
        know_flat = know_tokens.reshape(-1, Tk)
        know_encoded = self.knowledge_enc(know_flat, know_lengths.view(-1, 1))

        # apply the special layer for the knowledge. This is the same as the FFN
        # in the standard TransformerLayer
        if self.use_addl_resid:
            # pull out the result
            know_encoded['encoder_out'] = self.know_ffn(know_encoded['encoder_out'])

        # compute our sentence embeddings for context and knowledge
        context_use = universal_sentence_embedding(
            context_encoded['encoder_out'],
            context_encoded['encoder_padding_mask'],
            cosine=self.args.use_cosine,
            sqrt=self.args.use_sqrt,
        )
        know_use = universal_sentence_embedding(
            know_encoded['encoder_out'],
            know_encoded['encoder_padding_mask'],
            cosine=self.args.use_cosine,
            sqrt=self.args.use_sqrt,
        )

        # remash it back into the shape we need
        know_use = know_use.reshape(N, know_tokens.size(1), self.embed_dim)
        if self.args.use_embed_dim:
            context_use /= np.sqrt(self.embed_dim)
            know_use /= np.sqrt(self.embed_dim)

        ck_attn = (
            th.bmm(know_use, context_use.unsqueeze(-1)).squeeze(-1)
        )
        ck_attn.masked_fill_(~ck_mask, -65504 if self.args.fp16 else -1e20)

        if not use_cs_ids:
            # if we're not given the true chosen_sentence (test time), pick our
            # best guess
            _, cs_ids = ck_attn.max(1)

        # pick the true chosen sentence. remember that TransformerEncoder outputs
        #   (time, batch, embed)
        # but because know_encoded is a flattened, it's really
        #   (T, N * K, D)
        # We need to compute the offsets of the chosen_sentences
        cs_offsets = th.arange(N, device=cs_ids.device) * K + cs_ids
        cs_encoded = know_encoded['encoder_out'][:, cs_offsets]
        # but padding is (N * K, T)
        if know_encoded['encoder_padding_mask'] is not None:
            cs_padding_mask = know_encoded['encoder_padding_mask'][cs_offsets]
        else:
            # Transformers return None if the mask is completely full. This saves
            # compute, but we can't use that here. Produce a full (N, T) mask
            cs_padding_mask = ck_mask.new(N, cs_encoded.size(0)).fill_(0)

        ctx_encoded = context_encoded['encoder_out']
        ctx_padding_mask = context_encoded['encoder_padding_mask']

        # also need to check for the padding mask is None like above
        if ctx_padding_mask is None:
            ctx_padding_mask = ck_mask.new(N, ctx_encoded.size(0)).fill_(0)

        # merge the two items
        return {
            'encoder_out': ctx_encoded,
            'encoder_padding_mask': ctx_padding_mask,
            'cs_out': cs_encoded,
            'cs_padding_mask': cs_padding_mask,
            'ck_attn': ck_attn,
        }

    def max_positions(self):
        raise NotImplementedError()

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        if encoder_out['cs_out'] is not None:
            encoder_out['cs_out'] = \
                encoder_out['cs_out'].index_select(1, new_order)
        if encoder_out['cs_padding_mask'] is not None:
            encoder_out['cs_padding_mask'] = \
                encoder_out['cs_padding_mask'].index_select(0, new_order)
        if encoder_out['ck_attn'] is not None:
            encoder_out['ck_attn'] = \
                encoder_out['ck_attn'].index_select(0, new_order)
        return encoder_out

    def upgrade_state_dict(self, state_dict):
        if isinstance(self.transformer_enc.embed_positions,
                      SinusoidalPositionalEmbedding):
            if 'encoder.transformer_enc.embed_positions.weights' in state_dict:
                del state_dict['encoder.transformer_enc.embed_positions.weights']
            state_dict['encoder.transformer_enc.embed_positions._float_tensor'] = (
                th.FloatTensor(1)
            )
        return state_dict


class ContextKnowledgeDecoder(FairseqDecoder):
    def __init__(self, args, dictionary, embed_tokens, left_pad=False):
        super().__init__(dictionary)
        self.dropout = args.dropout

        # TODO: set this properly
        self.share_input_output_embed = True
        self.shared_encoder = args.shared_encoder

        embed_dim = embed_tokens.embedding_dim
        padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, embed_dim, padding_idx,
            left_pad=left_pad,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(args, no_encoder_attn=False)
            for _ in range(args.decoder_layers)
        ])

        if not self.share_input_output_embed:
            self.embed_out = nn.Parameter(th.Tensor(len(dictionary), embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=embed_dim ** -0.5)

    def forward(self,
                prev_output_tokens,
                encoder_out=None,
                incremental_state=None):

        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        joint_out = th.cat(
            [encoder_out['cs_out'], encoder_out['encoder_out']],
        )
        joint_pad = th.cat(
            [encoder_out['cs_padding_mask'], encoder_out['encoder_padding_mask']],
            dim=1,
        )

        # decoder layers from context
        for layer in self.layers:
            self_attn_mask = None
            if incremental_state is None:
                self_attn_mask = self.buffered_future_mask(x)
            x, attn = layer(
                x,
                joint_out,
                joint_pad,
                incremental_state,
                self_attn_mask=self_attn_mask
            )

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # project back to size of vocabulary
        if self.share_input_output_embed:
            x = F.linear(x, self.embed_tokens.weight)
        else:
            x = F.linear(x, self.embed_out)

        # fairseq doesn't like it when attn isn't the same length as src_tokens,
        # so we need to break this into two separate attns
        time_know = encoder_out['cs_out'].size(0)
        if attn is not None:
            attn2 = attn[:, :, :time_know]  # just the knowledge
            attn1 = attn[:, :, time_know:]  # after the knowledge
        else:
            attn1 = attn2 = None

        return x, attn1, attn2

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (not hasattr(self, '_future_mask') or
                self._future_mask is None or
                self._future_mask.device != tensor.device):
            self._future_mask = th.triu(fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._future_mask.size(0) < dim:
            self._future_mask = th.triu(
                fill_with_neg_inf(self._future_mask.resize_(dim, dim)),
                1
            )
        return self._future_mask[:dim, :dim]

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions())

    def upgrade_state_dict(self, state_dict):
        return state_dict


class KnowledgeTransformerModel(FairseqModel):
    """
    Transformer with special knowledge
    """
    @classmethod
    def build_model(cls, args, task):
        # make sure we're setting the defaults properly
        base_architecture(args)

        # model assumptions
        assert task.source_dictionary is task.target_dictionary, (
            "Only shared embeddings are allowed"
        )
        src_dict = task.source_dictionary
        # force all embeddings to be the same
        assert args.decoder_embed_dim == args.encoder_embed_dim
        if args.share_all_embeddings:
            args.share_decoder_input_output_embed = True
        embed_tokens = build_embedding(src_dict, args.encoder_embed_dim)

        # make the actual embeddings
        encoder = ContextKnowledgeEncoder(args, src_dict, embed_tokens)
        decoder = ContextKnowledgeDecoder(args, src_dict, embed_tokens)
        return KnowledgeTransformerModel(encoder, decoder)

    def forward(self, prev_output_tokens, **encoder_inputs):
        encoder_out = self.encoder(**encoder_inputs)
        decoder_out = self.decoder(prev_output_tokens, encoder_out)

        final_out = (
            decoder_out[0],
            None,
            encoder_out['ck_attn'],
        )
        return final_out


@register_criterion('wizard')
class WizardCriterion(FairseqCriterion):
    @staticmethod
    def add_args(argparser):
        # get the "parent" arguments in there
        argparser.add_argument(
            '--knowledge-alpha', type=float, default=0.95,
            help='Weight on the knowledge-attn loss'
        )

    def __init__(self, args, task):
        super().__init__(args, task)
        self.knowledge_alpha = args.knowledge_alpha

    def forward(self, model, sample, reduce=True):
        """Compute losses for a given sample."""
        assert reduce, "You need to double check this works reasonably"

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
            reduce=reduce
        )
        nll_loss = gen_loss / sample['ntokens'] / np.log(2)

        # knowledge loss, cross entropy over the attn
        ctx_know_attn = net_output[-1]
        ctx_know_targets = sample['net_input']['cs_ids']
        know_loss = F.cross_entropy(
            ctx_know_attn.float(),  # already logits
            ctx_know_targets,
            reduce=reduce,
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
            'loss': loss.item() if reduce else loss.data,
            # nll loss is always per token
            'nll_loss': nll_loss.item() if reduce else nll_loss.data,
            'know_loss': know_loss.item() if reduce else know_loss.data,
            # 'rank_loss': rank_loss.item() if reduce else rank_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': nsentences,
            'sample_size': sample_size,
            'know_acc': know_acc,
            'know_chance': know_chance,
            'know_pred': know_pred,
        }

        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        if len(logging_outputs) > 1:
            assert False, "This is all assuming there's just one host"
        output = logging_outputs[0]
        output['loss'] /= output['sample_size'] / np.log(2)
        return output

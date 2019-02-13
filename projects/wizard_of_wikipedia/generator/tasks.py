"""
All the generation tasks in a single place.
"""

import os
import json
from collections import defaultdict
import numpy as np

from parlai.core.teachers import FixedDialogTeacher
from parlai.tasks.squad.agents import get_sentence_tokenizer, build as build_squad
from parlai_internal.tasks.wizard_of_perzona.agents import WizardDialogKnowledgeTeacher
from parlai_internal.tasks.wizard_of_perzona.agents import TOKEN_NOCHOSEN
from parlai_internal.tasks.wizard_of_perzona.agents import TOKEN_KNOWLEDGE
from parlai_internal.tasks.funpedia.agents import FunpediaTeacher


def _zero_punct(string):
    return string.replace('.', '').replace('?', '').replace('!', '')


class FunpediaTeacher(FunpediaTeacher):
    """
    Returns messages similar to that of Wizard
    """
    def _build_action(self, entry):
        return {
            # Mirror the usage from ConvAI2
            'text': entry['title'] + '\n',
            'labels': [entry['label']],
            'chosen_topic': entry['title'],
            'knowledge': (
                entry['title'] + ' ' + TOKEN_KNOWLEDGE + ' ' + entry['passage']
            ),
            'title': entry['title'],
            'checked_sentence': entry['passage'],
            'reward': 0,
            'episode_done': True,
        }


class SquadTeacher(FixedDialogTeacher):
    """
    Squad teacher with:
        Title: the article title
        Knowledge: the sentence with the answer in it
        Text: the question
        Label: the extracted answer
    """
    def __init__(self, opt, shared=None):
        # download the data etc
        build_squad(opt)
        super().__init__(opt, shared)

        if not shared:
            suffix = 'train' if self.datatype.startswith('train') else 'dev'
            datapath = os.path.join(opt['datapath'], 'SQuAD', suffix + '-v1.1.json')
            self._setup_data(datapath)
        else:
            self.episodes = shared['episodes']

        self.id = 'ezsquad'
        self.reset()

    def share(self):
        shared = super().share()
        shared['episodes'] = self.episodes
        return shared

    def num_episodes(self):
        return self.num_examples()

    def num_examples(self):
        return len(self.episodes)

    def _setup_data(self, path):
        self.sent_tok = get_sentence_tokenizer()
        with open(path) as data_file:
            self.squad = json.load(data_file)['data']

        self.episodes = []
        num_skipped = 0
        for article in self.squad:
            title = article['title'].replace('_', ' ')

            for paragraph in article['paragraphs']:
                full_context = paragraph['context'].replace('\n', ' ')
                context = self.sent_tok.tokenize(full_context)
                zpctx = [_zero_punct(s) for s in context]

                for qa in paragraph['qas']:
                    question = qa['question']
                    answers = [a['text'] for a in qa['answers']]
                    zpanswers = [_zero_punct(a) for a in answers]

                    ctx_ans = defaultdict(set)
                    for c, zpc in zip(context, zpctx):
                        for a, zpa in zip(answers, zpanswers):
                            if zpa in c:
                                ctx_ans[c].add(a)

                    if not ctx_ans:
                        # couldn't find the correct answer
                        num_skipped += 1
                        continue

                    best_ctx = max(ctx_ans.keys(), key=lambda c: len(ctx_ans[c]))

                    self.episodes.append({
                        'chosen_topic': title,
                        'title': title,
                        'checked_sentence': best_ctx,
                        'knowledge': (
                            title + ' ' + TOKEN_KNOWLEDGE + ' ' + best_ctx
                        ),
                        'text': title + '\n' + question,
                        'labels': list(ctx_ans[best_ctx]),
                        'episode_done': True,
                    })
        if num_skipped:
            import warnings
            warnings.warn("Skipped {} squad questions in {}".format(num_skipped, path))

    def get(self, episode_idx, entry_idx=None):
        return self.episodes[episode_idx]


class WizardTeacher(WizardDialogKnowledgeTeacher):
    def __init__(self, opt, shared=None):
        opt['label_type'] = 'response'
        opt['include_checked_sentence'] = True
        opt['task'] = 'foobar:' + opt['task']  # make sure the topic_split logic works
        super().__init__(opt, shared)
        self.knowledge_separator = True
        self.only_checked_knowledge = opt.get('only_checked_knowledge', False)

    @staticmethod
    def add_cmdline_args(argparser):
        WizardDialogKnowledgeTeacher.add_cmdline_args(argparser)
        argparser.add_argument(
            '--only-checked-knowledge', type='bool', default=False,
            help='If true, only the checked sentence is provided'
        )

    def getID(self):
        return "WizTeacher"

    def get(self, episode_idx, entry_idx=0):
        a = super().get(episode_idx, entry_idx)
        # zero out the label candidates?
        if 'knowledge' not in a:
            # just a batch padding item
            return a
        # save some memory, we don't need label_candidates
        a['label_candidates'] = []
        if not a['knowledge'].startswith(TOKEN_NOCHOSEN):
            # make sure the token is appearing
            a['knowledge'] = (
                TOKEN_NOCHOSEN + ' ' + TOKEN_KNOWLEDGE + ' ' + TOKEN_NOCHOSEN +
                '\n' + a['knowledge']
            )
        if a['checked_sentence'] not in a['knowledge']:
            # just default to no passage chosen
            a['checked_sentence'] = TOKEN_NOCHOSEN
            a['title'] = TOKEN_NOCHOSEN
            #  but hard error if this isn't training time
            if 'train' not in self.datatype:
                raise ValueError(
                    "Our checked_sentence exception *cannot* be allowed in test/valid."
                )
        if self.only_checked_knowledge:
            # useful for test time evaluation, where it's only ever trained on true
            # knowledge
            a['knowledge'] = (
                a['title'] + ' ' + TOKEN_KNOWLEDGE + ' ' + a['checked_sentence']
            )
        return a


class GoldWizardTeacher(WizardTeacher):
    def __init__(self, opt, shared=None):
        opt['only_checked_knowledge'] = True
        super().__init__(opt, shared)

    def getID(self):
        return "GoldWizardTeacher"


class IgnorantWizardTeacher(WizardTeacher):
    def getID(self):
        return "IgnorantWizardTeacher"

    def get(self, episode_idx, entry_idx=0):
        a = super().get(episode_idx, entry_idx)
        a['title'] = TOKEN_NOCHOSEN
        a['checked_sentence'] = TOKEN_NOCHOSEN
        a['knowledge'] = (
            TOKEN_NOCHOSEN + ' ' + TOKEN_KNOWLEDGE + ' ' + TOKEN_NOCHOSEN
        )
        return a


class DropoutWizardTeacher(WizardTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.dropout = opt.get('ignorant_dropout', 0.0)

    @staticmethod
    def add_cmdline_args(argparser):
        agent = argparser.add_argument_group('Dropout Wizard Arguments')
        agent.add_argument('--ignorant-dropout', type=float, default=0.0,
                           help='Eliminate all knowledge with this probability.')

    def getID(self):
        return "DropoutWizardTeacher"

    def get(self, episode_idx, entry_idx=0):
        a = super().get(episode_idx, entry_idx)
        if self.dropout and np.random.rand() < self.dropout:
            a['title'] = TOKEN_NOCHOSEN
            a['checked_sentence'] = TOKEN_NOCHOSEN
            a['knowledge'] = (
                TOKEN_NOCHOSEN + ' ' + TOKEN_KNOWLEDGE + ' ' + TOKEN_NOCHOSEN
            )
        return a

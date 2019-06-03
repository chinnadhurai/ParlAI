This document contains run instructions for our ACL submission.

Please download and install SpaCy along with the associated English models to run perturbations relating to noun and verb dropping - https://spacy.io/.

Please look at parlai/core/perturb_utils.py for implementation details of our perturbations.

## Training Models

bash run.sh train <transformer/s2s/s2s_att_general> <dailydialog/personachat/dialog_babi:Task:5> <1/2/3/4/5>

1. The first argument specifies the type of model to train - transformers, seq2seq with lstms or seq2seq with lstms and attention.
2. The second argument specifies the dataset to train on
3. The third argument specifies which run to save models to (In our paper we averaged resutls across 3 runs).

## Testing with perturbations

bash run.sh perturb <transformer/s2s/s2s_att_general> <dailydialog/personachat/dialog_babi:Task:5> <1/2/3/4/5>

This runs all perturbations indicated in run.sh using a saved model. The result of running this should look like:

MODELTYPE: transformer
MODE : perturb
---------------------
CONFIG : dialog_babi:Task:5_transformer_test_NoPerturb
FINAL_REPORT: {'exs': 1000, 'accuracy': 0.876, 'f1': 0.9739, 'bleu': 0.8938, 'num_updates': 0, 'token_acc': 0.9862, 'loss': 0.07079, 'ppl': 1.073}
---------------------
CONFIG : dialog_babi:Task:5_transformer_test_only_last
FINAL_REPORT: {'exs': 1000, 'accuracy': 0.032, 'f1': 0.1929, 'bleu': 0.1325, 'num_updates': 0, 'token_acc': 0.7235, 'loss': 1.542, 'ppl': 4.674}
---------------------
CONFIG : dialog_babi:Task:5_transformer_test_shuffle
FINAL_REPORT: {'exs': 1000, 'accuracy': 0.534, 'f1': 0.6504, 'bleu': 0.5308, 'num_updates': 0, 'token_acc': 0.9347, 'loss': 0.3598, 'ppl': 1.433}
---------------------
CONFIG : dialog_babi:Task:5_transformer_test_reverse_utr_order
FINAL_REPORT: {'exs': 1000, 'accuracy': 0.428, 'f1': 0.5304, 'bleu': 0.3874, 'num_updates': 0, 'token_acc': 0.9106, 'loss': 0.5269, 'ppl': 1.694}
---------------------
CONFIG : dialog_babi:Task:5_transformer_test_drop_first
FINAL_REPORT: {'exs': 1000, 'accuracy': 0.875, 'f1': 0.9739, 'bleu': 0.8934, 'num_updates': 0, 'token_acc': 0.986, 'loss': 0.07157, 'ppl': 1.074}
---------------------
CONFIG : dialog_babi:Task:5_transformer_test_drop_last
FINAL_REPORT: {'exs': 1000, 'accuracy': 0.63, 'f1': 0.7523, 'bleu': 0.6465, 'num_updates': 0, 'token_acc': 0.9584, 'loss': 0.2582, 'ppl': 1.295}
---------------------
CONFIG : dialog_babi:Task:5_transformer_test_worddrop_random
FINAL_REPORT: {'exs': 1000, 'accuracy': 0.537, 'f1': 0.6823, 'bleu': 0.5764, 'num_updates': 0, 'token_acc': 0.9344, 'loss': 0.3451, 'ppl': 1.412}
---------------------
CONFIG : dialog_babi:Task:5_transformer_test_verbdrop_random
FINAL_REPORT: {'exs': 1000, 'accuracy': 0.642, 'f1': 0.7994, 'bleu': 0.7254, 'num_updates': 0, 'token_acc': 0.9565, 'loss': 0.2306, 'ppl': 1.259}
---------------------
CONFIG : dialog_babi:Task:5_transformer_test_noundrop_random
FINAL_REPORT: {'exs': 1000, 'accuracy': 0.519, 'f1': 0.6382, 'bleu': 0.5155, 'num_updates': 0, 'token_acc': 0.9221, 'loss': 0.4255, 'ppl': 1.53}
---------------------
CONFIG : dialog_babi:Task:5_transformer_test_wordshuf_random
FINAL_REPORT: {'exs': 1000, 'accuracy': 0.868, 'f1': 0.9717, 'bleu': 0.8873, 'num_updates': 0, 'token_acc': 0.9853, 'loss': 0.07352, 'ppl': 1.076}
---------------------
CONFIG : dialog_babi:Task:5_transformer_test_wordreverse_random
FINAL_REPORT: {'exs': 1000, 'accuracy': 0.863, 'f1': 0.9707, 'bleu': 0.8828, 'num_updates': 0, 'token_acc': 0.9845, 'loss': 0.07589, 'ppl': 1.079}
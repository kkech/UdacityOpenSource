import copy

GOOGLE_W2V = "./data/GoogleNews-vectors-negative300.bin.gz"

NEGATIONS_DICT = {
                "isn't":"is not", "aren't":"are not", "wasn't":"was not", 
                "weren't": "were not", "haven't":"have not","hasn't":"has not",
                "hadn't":"had not","won't":"will not", "wouldn't":"would not", 
                "don't":"do not", "doesn't":"does not", "didn't":"did not",
                "can't":"can not","couldn't":"could not",
                "shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"
                }

CLEANING_REGEX = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

HISTORY = lambda param: {
                "train_acc": copy.deepcopy(param),
                "topk_train_acc": copy.deepcopy(param),
                "train_loss": copy.deepcopy(param), 
                "val_acc": copy.deepcopy(param), 
                "topk_val_acc": copy.deepcopy(param),
                "val_loss": copy.deepcopy(param),
                }




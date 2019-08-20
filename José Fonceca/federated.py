import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import syft as sy
import torch as th

from utils.constants import HISTORY
from utils.processors import extend_data


def get_federated_dataset(data, users, context_size, hook):
    users_data = []
    workers = []
    for user in users:
        user_worker = sy.VirtualWorker(hook, id = user)
        cur_data = data[data.user == user]
        X, Y = extend_data(cur_data.X, cur_data.Y, context_size)
        X = th.tensor(X)
        Y = th.tensor(Y)
        users_data.append(sy.BaseDataset(X, Y).send(user_worker))
        workers.append(user_worker)
    return sy.FederatedDataset(users_data), workers


def update_history(history, new_history, iteration):
    for key, value in new_history.items():    
        padded = np.zeros(history[key].shape[0])
        value = np.array(value)    
        padded[:value.shape[0]] = value        
        history[key] += (padded - history[key]) / iteration
    return history


def update_best_metrics(best_history, history, iteration):
    for key, value in history.items():
        best_history[key] += (value[-1] - best_history[key]) / iteration
    return best_history
    

def update_params(model, new_model, updates_dict, iteration):
    for key, value in new_model.state_dict().items():
        if key in model.state_dict().keys():
            delta = model.state_dict()[key] - value                
            updates_dict[key] += (delta - updates_dict[key]) / iteration            
    return updates_dict    


def plot_metrics(history, plotting_file = "data/federated_evolution.pdf",
                data_file = "data/federated_evolution.csv"):
    fig, (ax1, ax2, ax3) = plt.subplots(nrows = 3, sharex = True)

    ax1.plot(history["train_loss"], "b--", label = "Train loss")
    ax1.plot(history["val_loss"], "k", label = "Validation loss")
    ax1.set_ylabel("Loss")
    ax1.grid()
    ax1.legend(loc = "upper center", bbox_to_anchor = (0.5, 1.3),
                ncol = 2, fancybox = True, shadow = True)
    
    ax2.plot(history["train_acc"], "b--", label = "Train accuracy") 
    ax2.plot(history["val_acc"], "k", label = "Validation accuracy")
    ax2.set_ylabel("Accuracy")
    ax2.grid()
    ax2.legend(loc = "upper center", bbox_to_anchor = (0.5, 1.3),
                ncol = 2, fancybox = True, shadow = True)

    ax3.plot(history["topk_train_acc"], "b--", label = "Top-3 train accuracy")
    ax3.plot(history["topk_val_acc"], "k", label = "Top-3 validation accuracy")
    ax3.set_ylabel("Top-3 Accuracy")
    ax3.set_xlabel(r"Round")
    ax3.grid()
    ax3.legend(loc = "upper center", bbox_to_anchor = (0.5, 1.3),
                ncol = 2, fancybox = True, shadow = True)

    fig.savefig(plotting_file, bbox_inches = "tight", format = "pdf")
    plt.close()

    data = pd.DataFrame(history)
    data.to_csv(data_file)


def train_federated(model, dataset, p_string, batch_size = 16,
                    epochs = 50, validation_split = 0.2):
    optimizer = th.optim.RMSprop    
    updates_dict = {key: 0 for key, value in model.state_dict().items()}
    history = HISTORY(np.zeros(epochs))
    best_history = HISTORY(0)    
    for idx, (_, basedataset) in enumerate(dataset.datasets.items()):
        user_model = copy.deepcopy(model)
        X = basedataset.data
        Y = basedataset.targets
        user_model = user_model.send(X.location)
        user_history = user_model.fit(X, Y, optimizer, batch_size, epochs, 
                        local = False, validation_split = validation_split, 
                        verbose = False, topk_pred = 3)
        best_history = update_best_metrics(best_history, user_history, idx + 1)
        history = update_history(history, user_history, idx + 1)                        
        user_model = user_model.get()        
        updates_dict = update_params(model, user_model, updates_dict, idx + 1)
        print(p_string(idx + 1), end = "\r")
        del user_model
    params_dict = {key: value + updates_dict[key] 
                    for key, value in model.state_dict().items()}
    model.load_state_dict(params_dict, strict = False)
    return model, best_history


def train_multiple_federated(model, data, model_file, V, user_batch_size = 20, 
                            batch_size = 16, context_size = 5, epochs = 30,
                            validation_split = 0.2):
    unique_users = data.user.unique()
    n_user_batches = len(unique_users) // user_batch_size
    federated_model = copy.deepcopy(model)   
    best_history = HISTORY([])
    hook = sy.TorchHook(th)
    p_string = "Users round {}/{}\t Users processed current round: {}/{}".format
    for i in range(n_user_batches):
        cur_users = unique_users[i*user_batch_size:(i + 1)*user_batch_size]
        federated_dataset, workers = get_federated_dataset(data, cur_users, 
                                                            context_size, hook)
        p_str = p_string(i + 1, n_user_batches, "{}", user_batch_size).format
        federated_model, split_history = train_federated(federated_model, 
                                            federated_dataset, p_str,
                                            batch_size, epochs, 
                                            validation_split)
        for key, value in split_history.items():
            best_history[key].append(value)
        plot_metrics(best_history)
        th.save(federated_model.state_dict(), model_file)
        del federated_dataset
        for worker in workers:
            del worker        
    print("")
        



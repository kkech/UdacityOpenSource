from federated import train_federated, train_multiple_federated
from local import train_local_model
from utils.processors import get_local_and_remote_data, merge_and_index_data
from utils.models import bidirectional_LSTM


if __name__ == "__main__":
    data_file = "data/kaggle_twitter.csv"
    model_file = "data/LSTM_model_local.pth"
    federated_file = "data/LSTM_model_federated.pth"
    dump_file = "data/pandas_df.pkl"
    word2idx_file = "data/tokenizer_keys.pkl"
    min_tweets = 20
    local_share = 0.2
    context_size = 5
    epochs = 5
    D = 300
    n_nodes = 128
    data, word2idx = merge_and_index_data(data_file, dump_file, word2idx_file, min_tweets)
    local_data, remote_data = get_local_and_remote_data(data, local_share)
    model = train_local_model(bidirectional_LSTM, word2idx, D, n_nodes,
                    local_data, model_file, context_size = context_size, 
                    epochs = epochs)
    train_multiple_federated(model, remote_data, federated_file, len(word2idx))

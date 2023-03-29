import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, mean_absolute_error
#from sklearn.metrics import mean_absolute_percentage_error
import math




def get_norm(labels, datelist, nodelist):
    avg_norm = labels.query('date in @datelist & channelId in @nodelist')['target'].to_numpy()
    avg_norm_mu = np.mean(avg_norm, axis=0)
    avg_norm_sigma = np.std(avg_norm, axis=0)
    #norm_df = labels.query('date in @datelist')['target']

    norm_dict = dict()
    for node in nodelist:
        norm = labels.query('date in @datelist & channelId == @node')['target'].to_numpy()
        if norm.shape[0] != 0:
            norm_mu = np.mean(norm, axis=0)
            norm_sigma = np.std(norm, axis=0)
        else:
            norm_mu = avg_norm_mu
            norm_sigma = avg_norm_sigma

        norm_dict[node] = [norm_mu, norm_sigma]
        
    '''
    norm_mu = np.mean(norm, axis=0)
    norm_sigma = np.std(norm, axis=0)
    '''
    return norm_dict #norm_mu, norm_sigma


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()

    return correct / len(labels)


def evaluation(output, labels, norm_dict, nodes):
    norm = np.array([norm_dict[x] for x in nodes])      # [0]: mu, [1]: sigma
    #print(norm[-10:])
    #print(np.isnan(norm).any())
    output = output.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    #print(output.shape, labels.shape, norm[:, 1].shape)
    output = output * np.expand_dims(norm[:, 1], axis=1) + np.expand_dims(norm[:, 0], axis=1)
    #output = abs(output * norm_sigma + norm_mu)
    #print(output.shape, labels.shape)
    #labels = (labels - np.mean(norm, axis=0)) / np.std(norm, axis=0)
    #print(output[:10], labels[:10])
    #print(output.shape)
    #print(norm_sigma, norm_mu)
    #print(output[:10])
    #print(labels[:10])
    rmse = math.sqrt(mean_squared_error(output, labels))
    mape = mean_absolute_percentage_error(output, labels)
    r2 = r2_score(output, labels)
    mae = mean_absolute_error(output, labels)

    return rmse, mape, r2, mae


class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-10):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1

        self.epoch_count += 1

        return self.num_round >= self.max_round

import numpy as np
import torch
import argparse
import time
import logging
import sys
from pathlib import Path
from readData import readData
from MLN import MLN
from utils import EarlyStopMonitor, evaluation, get_norm
import matplotlib.pyplot as plt
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

# pd.set_option('display.max_columns', None)


parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('--start', type=str, default="2021-04", help='Start date(e.g. 2021-04)')
parser.add_argument('--period', type=str, default="m", choices=[
    "d", "w", "m"], help='Period of data separation(day, week, month)')
parser.add_argument('--epochs', type=int, default=10,
                    help='Number of epochs to train.')          # straight_5_18  attn_3_29
parser.add_argument('--prefix', type=str, default='straight_5_18', help='Prefix to name the checkpoints')
parser.add_argument('--coldstart', type=int, default=8, help='Number of data for pretraining')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-2,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden embedding dimension.')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='Hyper-parameter for graph structure learning.')
parser.add_argument('--la', type=float, default=0.01,
                    help='Hyper-parameter for GSL constraints.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--perf', action='store_true', default=True,
                    help='Percentage label.')
parser.add_argument('--gsl', action='store_true', default=False,
                    help='Using graph structure learning.')


try:
    #print(parser.parse_known_args())
    args, unknown = parser.parse_known_args()
except:
    parser.print_help()
    sys.exit(0)

PERIOD = args.period
START = args.start
args.cuda = 0 if (not args.no_cuda) and (torch.cuda.is_available()) else -1
COLDSTART = args.coldstart

# pd.set_option('display.max_columns', None)
np.set_printoptions(threshold=np.inf)
torch.autograd.set_detect_anomaly(True)

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler("log/{}.log".format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f"./saved_models/{args.prefix}-{PERIOD}" + "\
  node-classification.pth"
get_checkpoint_path = lambda \
        epoch: f"./saved_checkpoints/{args.prefix}-{PERIOD}-{epoch}" + "\
  node-classification.pth"

torch.cuda.empty_cache()

# Set device
device_string = "cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu"
device = torch.device(device_string)

datelist, node_feature, adj_viewer, adj_period, adj_description, labels, nodes, nodelist = readData(PERIOD)




#datelist = datelist[:-5]

#cs_list = datelist[:COLDSTART]
#norm_dict = get_norm(labels, datelist[:COLDSTART], nodelist, args.perf)
#norm_mu, norm_sigma = get_norm(labels, datelist[:COLDSTART], nodelist, args.perf)

if 0 >= COLDSTART or COLDSTART >= len(datelist):
    logger.error('Invalid COLDSTART')
    sys.exit(0)

model = MLN(datelist, node_feature, adj_viewer, adj_period, adj_description,
            labels, nodes, nodelist, args.hidden, device, args.dropout, args.perf, args.gsl)
model = model.to(device)

logger.debug("Num of dates: {}".format(len(datelist)))
logger.debug("Num of total nodes: {}".format(len(nodelist)))

logger.info("Start task")

early_stopper = EarlyStopMonitor(max_round=args.patience)

params = list(model.parameters())
optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
criterion = torch.nn.MSELoss()


early_stopper = EarlyStopMonitor(max_round=args.patience, higher_better=False)
for epoch in range(args.epochs):
    logger.info('Epoch {:04d} start!'.format(epoch+1))
    t = time.time()

    model.train()
    loss = 0
    train_auc = 0
    train_pre = 0

    optimizer.zero_grad()

    for k in range(len(datelist)-2) :
        #print(datelist[k])
        #print(len(nodes[datelist[k]]))

        node_embedding, output, y_true, adj_v, adj_p = model.get_embedding(datelist[k])

        #norm = np.array([norm_dict[x] for x in nodes[datelist[k]]])  # [0]: mu, [1]: sigma
        #y_true_n = (y_true - torch.from_numpy(np.expand_dims(norm[:, 0], axis=1)).to(device)) /\
        #           torch.from_numpy(np.expand_dims(norm[:, 1], axis=1)).to(device)


        y_pred = output
        #print(output[:5], y_pred[:5], y_true[:5])
        #print(output[:10], y_pred[:10], y_true[:10])

        #print(y_pred.dtype, y_true.dtype)
        loss_train = criterion(y_pred.to(torch.float32), y_true.to(torch.float32)) \
                     + args.la * criterion(adj_v, torch.zeros(adj_v.shape).to(device))\
                     + args.la * criterion(adj_p, torch.zeros(adj_p.shape).to(device))
        loss += loss_train
        if k >= COLDSTART:
            with torch.no_grad():
                rmse, mape, r2, mae = evaluation(y_pred, y_true)
            logger.info('Date: {}'.format(datelist[k]))
            logger.info('Loss: {:.4f}'.format(loss_train.item()))
            logger.info('RMSE: {:.4f}, MAPE: {:.4f}, R2_score: {:.4f}, MAE: {:.4f}'.format(rmse, mape, r2, mae))
            torch.cuda.empty_cache()

    loss.backward()
    optimizer.step()

    model.eval()
    node_embedding, output, y_true, adj_v, adj_p = model.get_embedding(datelist[-2])


    #norm = np.array([norm_dict[x] for x in nodes[datelist[-2]]])  # [0]: mu, [1]: sigma
    #y_true_n = (y_true - torch.from_numpy(np.expand_dims(norm[:, 0], axis=1)).to(device)) / \
    #           torch.from_numpy(np.expand_dims(norm[:, 1], axis=1)).to(device)


    y_pred = output
    loss_val = criterion(output, y_true) \
                     + args.la * criterion(adj_v, torch.zeros(adj_v.shape).to(device))\
                     + args.la * criterion(adj_p, torch.zeros(adj_p.shape).to(device))
    #print(output[:5], y_pred[:5], y_true[:5])

    with torch.no_grad():
        rmse, mape, r2, mae = evaluation(y_pred, y_true)
    logger.info('Validation-----------------'.format(datelist[-2]))
    logger.info('Date: {}'.format(datelist[-2]))
    logger.info('Loss: {:.4f}'.format(loss_val.item()))
    logger.info('RMSE: {:.4f}, MAPE: {:.4f}, R2_score: {:.4f}, MAE: {:.4f}'.format(rmse, mape, r2, mae))

    logger.info("Optimization Finished!")
    logger.info('Epoch {:04d} time: {:.4f}s'.format(epoch + 1, time.time() - t))

    '''
    x = np.arange(y_pred.shape[0])
    l1 = plt.plot(x, y_pred.detach().cpu().numpy() - y_true.detach().cpu().numpy(), 'r--', label='y_pred')
    #l1 = plt.plot(x, , 'g--', label='y_true')
    plt.plot(x, y_pred.detach().cpu().numpy(), 'ro-')
    plt.xlabel('samples')
    plt.ylabel('result')
    plt.legend()
    plt.show()
    '''

    if early_stopper.early_stop_check(loss.item()):
        logger.info("No improvement over {} epochs, stop training".format(early_stopper.max_round))
        break
    else:
        torch.save(model.state_dict(), get_checkpoint_path(epoch))

    torch.cuda.empty_cache()


logger.info(f"Loading the best model at epoch {early_stopper.best_epoch}")
best_model_path = get_checkpoint_path(early_stopper.best_epoch)
model.load_state_dict(torch.load(best_model_path))
logger.info(f"Loaded the best model at epoch {early_stopper.best_epoch} for inference")

model.eval()
node_embedding, output, y_true, adj_v, adj_p = model.get_embedding(datelist[-1])

#norm = np.array([norm_dict[x] for x in nodes[datelist[-1]]])  # [0]: mu, [1]: sigma
#y_true_n = (y_true - torch.from_numpy(np.expand_dims(norm[:, 0], axis=1)).to(device)) /\
#                   torch.from_numpy(np.expand_dims(norm[:, 1], axis=1)).to(device)

y_pred = output
loss_test = criterion(y_pred, y_true)\
                     + args.la * criterion(adj_v, torch.zeros(adj_v.shape).to(device))\
                     + args.la * criterion(adj_p, torch.zeros(adj_p.shape).to(device))

#print(output[:5], y_pred[:5], y_true[:5])

with torch.no_grad():
    rmse, mape, r2, mae = evaluation(y_pred, y_true)
logger.info('Test-----------------'.format(datelist[-1]))
logger.info('Date: {}'.format(datelist[-1]))
logger.info('Loss: {:.4f}'.format(loss_val.item()))
logger.info('RMSE: {:.4f}, MAPE: {:.4f}, R2_score: {:.4f}, MAE: {:.4f}'.format(rmse, mape, r2, mae))

'''
x = np.arange(y_pred.shape[0])
l1 = plt.plot(x, y_pred.detach().cpu().numpy() - y_true.detach().cpu().numpy(), 'r--', label='y_pred')
#l1 = plt.plot(x, , 'g--', label='y_true')
plt.plot(x, y_pred.detach().cpu().numpy(), 'ro-')
plt.xlabel('samples')
plt.ylabel('result')
plt.legend()
plt.show()
'''

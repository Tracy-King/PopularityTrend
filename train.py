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

# pd.set_option('display.max_columns', None)


parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('--start', type=str, default="2021-04", help='Start date(e.g. 2021-04)')
parser.add_argument('--period', type=str, default="w", choices=[
    "d", "w", "m"], help='Period of data separation(day, week, month)')
parser.add_argument('--year', type=str, default="2022", choices=["2021", "2022"],
                    help='Period of data separation(day, week, month)')
parser.add_argument('--epochs', type=int, default=10,
                    help='Number of epochs to train.')
parser.add_argument('--coldstart', type=int, default=10, help='Number of data for pretraining')
parser.add_argument('--lr', type=float, default=0.1,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden embedding dimension.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

PERIOD = args.period
START = args.start
YEAR = args.year
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

torch.cuda.empty_cache()

# Set device
device_string = "cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu"
device = torch.device(device_string)

datelist, node_feature, adj_viewer, adj_period, adj_description, labels, nodes, nodelist = readData(PERIOD)

#cs_list = datelist[:COLDSTART]
norm_mu, norm_sigma = get_norm(labels, datelist[:COLDSTART], nodelist)

if 0 >= COLDSTART or COLDSTART >= len(datelist):
    logger.error('Invalid COLDSTART')
    sys.exit(0)

model = MLN(datelist, node_feature, adj_viewer, adj_period, adj_description,
            labels, nodes, nodelist, args.hidden, device, args.dropout)
model = model.to(device)

logger.debug("Num of dates: {}".format(len(datelist)))
logger.debug("Num of total nodes: {}".format(len(nodelist)))

logger.info("Start task")

early_stopper = EarlyStopMonitor(max_round=args.patience)

params = list(model.parameters())
optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
criterion = torch.nn.MSELoss()

for epoch in range(args.epochs):
    logger.info('Epoch {:04d} start!'.format(epoch+1))
    t = time.time()

    model = model.train()
    loss = 0
    train_auc = 0
    train_pre = 0

    optimizer.zero_grad()

    for k in range(len(datelist)):
        #print(datelist[k])
        #print(len(nodes[datelist[k]]))

        node_embedding, y_pred, y_true = model.get_embedding(datelist[k])

        loss_train = criterion(y_pred, y_true)
        loss += loss_train
        if k >= COLDSTART:
            with torch.no_grad():
                rmse, mape = evaluation(y_pred, y_true, norm_mu, norm_sigma, nodes[datelist[k]])
            logger.info('Date: {}'.format(datelist[k]))
            logger.info('Loss: {:.4f}'.format(loss_train.item()))
            logger.info('RMSE: {:.4f}, MAPE: {:.4f}'.format(rmse, mape))

        if k == len(datelist)-1:
            with torch.no_grad():
                rmse, mape = evaluation(y_pred, y_true, norm_mu, norm_sigma, nodes[datelist[k]])
            logger.info('Test-----------------'.format(datelist[k]))
            logger.info('Date: {}'.format(datelist[k]))
            logger.info('Loss: {:.4f}'.format(loss_train.item()))
            logger.info('RMSE: {:.4f}, MAPE: {:.4f}'.format(rmse, mape))

    loss.backward()
    optimizer.step()

    logger.info("Optimization Finished!")
    logger.info('Epoch {:04d} time: {:.4f}s'.format(epoch + 1, time.time() - t))

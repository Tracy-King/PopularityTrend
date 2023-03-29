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
parser.add_argument('--period', type=str, default="m", choices=[
    "d", "w", "m"], help='Period of data separation(day, week, month)')
parser.add_argument('--year', type=str, default="2022", choices=["2021", "2022"],
                    help='Period of data separation(day, week, month)')
parser.add_argument('--epochs', type=int, default=10,
                    help='Number of epochs to train.')
parser.add_argument('--prefix', type=str, default='attn_3_29', help='Prefix to name the checkpoints')
parser.add_argument('--coldstart', type=int, default=10, help='Number of data for pretraining')
parser.add_argument('--lr', type=float, default=0.1,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-2,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden embedding dimension.')
parser.add_argument('--dropout', type=float, default=0.2,
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
norm_dict = get_norm(labels, datelist[:COLDSTART], nodelist)
#norm_mu, norm_sigma = get_norm(labels, datelist[:COLDSTART], nodelist)

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


early_stopper = EarlyStopMonitor(max_round=args.patience, higher_better=False)
for epoch in range(args.epochs):
    logger.info('Epoch {:04d} start!'.format(epoch+1))
    t = time.time()

    model = model.train()
    loss = 0
    train_auc = 0
    train_pre = 0

    optimizer.zero_grad()

    for k in range(len(datelist)-2) :
        #print(datelist[k])
        #print(len(nodes[datelist[k]]))

        node_embedding, y_pred, y_true = model.get_embedding(datelist[k])

        loss_train = criterion(y_pred, y_true)
        loss += loss_train
        if k >= COLDSTART:
            with torch.no_grad():
                rmse, mape, r2, mae = evaluation(y_pred, y_true, norm_dict, nodes[datelist[k]])
            logger.info('Date: {}'.format(datelist[k]))
            logger.info('Loss: {:.4f}'.format(loss_train.item()))
            logger.info('RMSE: {:.4f}, MAPE: {:.4f}, R2_score: {:.4f}, MAE: {:.4f}'.format(rmse, mape, r2, mae))
            torch.cuda.empty_cache()

    loss.backward()
    optimizer.step()

    model.eval()
    node_embedding, y_pred, y_true = model.get_embedding(datelist[-2])
    loss_val = criterion(y_pred, y_true)

    with torch.no_grad():
        rmse, mape, r2, mae = evaluation(y_pred, y_true, norm_dict, nodes[datelist[-2]])
    logger.info('Validation-----------------'.format(datelist[-2]))
    logger.info('Date: {}'.format(datelist[-2]))
    logger.info('Loss: {:.4f}'.format(loss_val.item()))
    logger.info('RMSE: {:.4f}, MAPE: {:.4f}, R2_score: {:.4f}, MAE: {:.4f}'.format(rmse, mape, r2, mae))

    logger.info("Optimization Finished!")
    logger.info('Epoch {:04d} time: {:.4f}s'.format(epoch + 1, time.time() - t))

    if early_stopper.early_stop_check(mape):
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
node_embedding, y_pred, y_true = model.get_embedding(datelist[-1])
loss_test = criterion(y_pred, y_true)

with torch.no_grad():
    rmse, mape, r2, mae = evaluation(y_pred, y_true, norm_dict, nodes[datelist[-1]])
logger.info('Test-----------------'.format(datelist[-1]))
logger.info('Date: {}'.format(datelist[-1]))
logger.info('Loss: {:.4f}'.format(loss_val.item()))
logger.info('RMSE: {:.4f}, MAPE: {:.4f}, R2_score: {:.4f}, MAE: {:.4f}'.format(rmse, mape, r2, mae))


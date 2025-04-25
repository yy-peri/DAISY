import argparse
import random
import time
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tools import EarlyStopping, time_since, random_seed
import pandas as pd
from DAISY import get_interaction_map, get_global_attributes_diff, resnet_danet


parser = argparse.ArgumentParser(description='train the TCR_pMHC prediction model')
# file dir
parser.add_argument('--input', type=str, default='./data/tcr_pmhc_train.csv', help='path to input data,\
includes the following two columns:peptide, cdr3')
parser.add_argument('--healty_tcr', type=str, default='./data/small_healthy_tcr.csv', help='tcr of healthy people')
parser.add_argument('--model_dir', type=str, default='./model/DAISY.pt', help='where to save model')
# Hyperparameters
parser.add_argument('--batch_size', type=int, default=256, help='batch_size for tcr_pmhc prediction model')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--max_epoch', type=int, default=500, help='max epoch')
parser.add_argument('--seed', type=int, help='random seed')
args = parser.parse_args()


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# set the random seed
random_seed(args.seed)



# Load data and process data
df = pd.read_csv(args.input)
cdr3s = df['cdr3'].tolist()
peptides = df['peptide'].tolist()
healthy_tcrs = pd.read_csv(args.healty_tcr, nrows=10000)['cdr3'].tolist()


# generate dataloader for training the tcr_pmhc prediction model
def make_data(tcrs, peptides, healthy_tcrs, mode = 'healthy'):
    print(f"Creating data with mode: {mode}")
    pos_map_data = get_interaction_map(tcrs, peptides)     # [batch_size, 5, tcr, pep]
    pos_global_data = get_global_attributes_diff(tcrs, peptides)
    neg_map_data = []
    neg_global_data = []
    pos = 0
    for index, peptide in enumerate(peptides):
        num = 0
        while num < 1:
            if mode == 'healthy':
                tcr = healthy_tcrs[random.randint(0, len(healthy_tcrs)-1)]
            else:
                tcr = tcrs[random.randint(0, len(tcrs)-1)]   # 用途????？?

            if tcr == tcrs[index]:
                continue

            neg_map_data.extend(get_interaction_map([tcr], [peptide]))     # [1, 5, tcr, pep]
            neg_global_data.extend(get_global_attributes_diff([tcr], [peptide]))
            num += 1
            pos += 1
    total_map_data = torch.cat((torch.tensor(pos_map_data), torch.tensor(neg_map_data)), dim=0)
    total_global_data = torch.cat((torch.tensor(pos_global_data), torch.tensor(neg_global_data)), dim=0)  # [total_samples, global_feature_dim]
    labels = [1] * len(pos_map_data) + [0] * len(neg_map_data)
    dataset = torch.utils.data.TensorDataset(total_map_data, total_global_data, torch.tensor(labels, dtype=torch.int64))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


# Split the training set and validation set
train_tcrs, val_tcrs, train_peptides, val_peptides = train_test_split(cdr3s, peptides, test_size=0.1, random_state=0)
# Create the Dataloader for the training set and the validation set
train_dataloader = make_data(train_tcrs, train_peptides, healthy_tcrs)
val_dataloader = make_data(val_tcrs, val_peptides, healthy_tcrs)


# Initialize tcr-pmhc prediction model
model = resnet_danet()
model.to(DEVICE)
if torch.cuda.is_available() & torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))


# Initialize the loss function and the optimizer
Loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)

# Initialize the learning rate decay strategy and early stop strategy
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2, verbose=True)
early_stopping = EarlyStopping(patience=6, verbose=True, save_path=args.model_dir)

# model training
def train(epoch):
    model.train()
    train_loss = 0.0
    for tra_step, (maps, diffs, labels) in enumerate(train_dataloader, 1):
        maps, diffs, labels = maps.to(DEVICE), diffs.to(DEVICE), labels.to(DEVICE)
        outputs = model(maps, diffs)
        loss = Loss(outputs, labels)
        # calculate the accuracy
        probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
        preds = [1 if prob > 0.5 else 0 for prob in probs]
        accuracy = accuracy_score(labels.detach().cpu().numpy(), preds)
        train_loss += loss.detach()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if tra_step % 20 == 0:
            print(f'epoch: {epoch}, tra_step: {tra_step}, train_loss: {train_loss/tra_step}, loss: {loss.detach():.3f}, accuracy: {accuracy:.3f}')


def validation(epoch):
    model.eval()
    val_loss = 0.0
    all_step = 0
    with torch.no_grad():
        for val_step, (maps, diffs, labels) in enumerate(val_dataloader, 1):
            maps, diffs, labels = maps.to(DEVICE), diffs.to(DEVICE), labels.to(DEVICE)
            outputs = model(maps, diffs)
            loss = Loss(outputs, labels)
            val_loss += loss.detach()
            all_step = val_step
            # calculate the accuracy
            probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
            preds = [1 if prob > 0.5 else 0 for prob in probs]
            accuracy = accuracy_score(labels.detach().cpu().numpy(), preds)
            if val_step % 2 == 0:
                print(f'epoch: {epoch}, val_step: {val_step}, val_loss: {val_loss/val_step}, loss: {loss.detach():.3f}, accuracy: {accuracy:.3f}')
    return val_loss/all_step


if __name__ == '__main__':
    start_time = time.time()
    print(f'model training starts at: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))}')
    for epoch in range(1, args.max_epoch+1):
        train(epoch)
        val_loss = validation(epoch)
        # Determine whether the learning rate needs to be decayed
        lr_scheduler.step(val_loss)
        # Determine whether to stop training
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            break
        # save the model
    torch.save(model.state_dict(), args.model_dir)
    end_time = time.time()
    print(f'model training finished at: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))}')
    print(time_since(start_time, end_time))


















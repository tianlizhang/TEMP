import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN
from tqdm import trange
from dataset_2 import load_data_aminer, load_data_V13, load_data_V11


class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, periods):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN(in_channels=node_features,  out_channels=32, periods=periods)
        self.linear = torch.nn.Linear(32, periods)
        self.train_year = 5
        self.relation = ['P1P','P1A','P1V','P1K','self']
        self.support = len(self.relation)

    def forward(self, x, edge_index):
        h = self.tgnn(x, edge_index) # x [b, 207, 2, 12]  returns h [b, 207, 12]
        h = F.relu(h)
        h = self.linear(h)

        return h


def train():
    epochs = 100
    bar = trange(epochs)
    train_loss, total = 0, 0
    for epoch in bar:
        model.train()

        edge_index = (edge_list[0][0]).to(device)
        feat = feature.to(device)

        out = model(feat, edge_index)
        # print(out.shape, out[rank].shape, labels.shape)

        loss = loss_fn(out[rank_train], labels_train.to(device))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # loss_list.append(loss.item())
        train_loss += loss.item()
        total += 1
        
        bar.set_postfix(train_loss = train_loss/total)
        # print("Epoch {} train RMSE: {:.4f}".format(epoch, sum(loss_list)/len(loss_list)))


def eval_set():
    model.eval()
    edge_index = (edge_list[0][0]).to(device)
    feat = feature.to(device)
    out = model(feat, edge_index)
    loss = loss_fn(out[rank_test], labels_test.to(device))
    print('valid: ', loss)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    model = TemporalGNN(node_features=4, periods=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    # edge_list, feature, labels, label_max, label_min, rank = load_data_V13(2010, 'train')
    # edge_list, feature, labels_train, labels_test, rank_train, rank_test = load_data_V13(2010, 'train')
    # edge_list, feature_list, labels_train, labels_test, rank_train, rank_test =  load_data_V13(2010, 'train')
    # edge_list, feature, labels_train, labels_test, rank_train, rank_test = load_data_V13(2010, 'train')

    # edge_list, feature, labels_train, labels_test, rank_train, rank_test = load_data_V11(2010, 'train')
    edge_list, feature, labels_train, labels_test, rank_train, rank_test = load_data_V13(2010, 'train')
    train()

    eval_set()
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric_temporal.nn.recurrent import A3TGCN, TGCN
from torch_geometric_temporal.nn.attention import  MSTGCN
# from torch_geometric_temporal.nn
from tqdm import trange
from dataset import load_data_aminer, load_data_V13, load_data_V11


class mygnn(torch.nn.Module):
    def __init__(self, node_features):
        super(mygnn, self).__init__()
        self.train_year = 5
        self.relation = ['P1P','P1A','P1V','P1K','self']
        self.support = len(self.relation)
        self.tgnn_list = []
        self.hidden = 4
        for i in range(self.support):
            # self.tgnn_list.append( TGCN(in_channels=node_features,  out_channels=4).to(device) )
            self.tgnn_list.append( MSTGCN(nb_block=1, in_channels=node_features, K=1, nb_chev_filter=1, nb_time_filter=1,\
             time_strides=1, num_for_predict=self.hidden, len_input=1).to(device)  )

    def forward(self, feature, edge):
        out = []
        for i in range(self.support ):
            ff = torch.from_numpy(feature.reshape(1, -1, 4, 1)).to(torch.float).to(device)
            ee = edge[i].to(device)
            # print(ff.shape, ee.shape)
            tt = self.tgnn_list[i](ff, ee)
            out.append(tt)
        x1 = torch.hstack(out) # [n, 20]
        return x1


class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, periods):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.train_year = 5
        self.relation = ['P1P','P1A','P1V','P1K','self']
        self.support = len(self.relation)

        self.gnn_list = []
        for i in range(self.train_year):
            self.gnn_list.append( mygnn(node_features).to(device) )
        
        # node_features=2, periods=12
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(32, periods)
        self.lstm = nn.LSTM(20, 32, 1, batch_first=True)

    def forward(self, feature_list, edge_list, rank):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        out = []
        for i in range(self.train_year):
            x1 = self.gnn_list[i](feature_list[i], edge_list[i])
            out.append(x1[rank, :]) # [n, 20]
        x2 = torch.stack(out, dim=1) # [p, 5, 20]
        # print('x2:', x2.shape)

        _, (h, _) = self.lstm(x2)
        # h = self.tgnn(x2, edge_index) # x [b, 207, 2, 12]  returns h [b, 207, 12]
        # print('h', h.shape, 'oo:', oo.shape, 'cc:', cc.shape) # [1, 1200, 32]
        h = F.relu(h.squeeze(0))
        h = self.linear(h)
        # print('h', h.shape)
        return h


def train():
    epochs = 1000
    bar = trange(epochs)
    train_loss, total = 0, 0
    for epoch in bar:
        model.train()

        out = model(feature_list, edge_list, rank_train)
        # print('out:', out.shape)
        # print(out.shape, out[rank].shape, labels.shape)
        
        labels_train_ = labels_train.to(device)
        # print('labels_train:', labels_train_.shape)
        # print('out[rank_train]:', out[rank_train].shape)
        loss = loss_fn(out, labels_train_)
        # print('loss:', loss.shape)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # loss_list.append(loss.item())
        train_loss += loss.item()
        total += 1
        
        bar.set_postfix(train_loss = train_loss/total)


def eval_set():
    model.eval()
    out = model(feature_list, edge_list, rank_test)
    loss = loss_fn(out, labels_test.to(device))
    print('valid: ', loss)

torch.cuda.set_device(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    model = TemporalGNN(node_features=4, periods=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    # edge_list, feature, labels, label_max, label_min, rank = load_data_V13(2010, 'train')
    # edge_list, feature, labels_train, labels_test, rank_train, rank_test = load_data_V13(2010, 'train')

    # edge_list, feature_list, labels_train, labels_test, rank_train, rank_test =  load_data_V13(2010, 'train')
    edge_list, feature_list, labels_train, labels_test, rank_train, rank_test =  load_data_V11(2010, 'train')
    train()

    eval_set()
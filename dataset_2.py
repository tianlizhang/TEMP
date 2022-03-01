import pickle as pkl
import numpy as np
import scipy.sparse as sp
import torch

def lable_normorlization(labels):
    maximum = labels.max()
    minimum = labels.min()
    new_value = (labels-minimum)/(maximum-minimum)
    return new_value,maximum,minimum


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1)) # [n, 1]
    r_inv = np.power(rowsum, -1).flatten() #[n]
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv) # [n, n]
    mx = r_mat_inv.dot(mx) # [n, n]
    return mx


def adj2edge_index(adj):
    adj_coo = adj.tocoo()
    row, col = adj_coo.row, adj_coo.col
    edge_index = torch.zeros((2, len(row))).to(torch.long)
    edge_index[0] = torch.from_numpy(row)
    edge_index[1] = torch.from_numpy(col)
    return edge_index


# def load_data_dblpV13(year,flag):
def load_data_aminer(year,flag):
    # DPATH = '../07_HINTS_code-main/fanxing'
    DPATH = '../07_HINTS_code-main/aminer'
    with open(f'{DPATH}/individual_data/graph_' + str(year - 5) + '_nf.pkl', 'rb') as f:
        graph_1 = pkl.load(f)
        adj_ = [normalize(adj_)for adj_ in graph_1['adj']] # adj: list(sp.csr_matrix), 5个类型
        adj_1 = [adj2edge_index(adj) for adj in adj_]
        feature_1 = graph_1['feature'] # np.array, [n, 4]

    with open(f'{DPATH}/individual_data/graph_' + str(year - 4) + '_nf.pkl', 'rb') as f:
        graph_2 = pkl.load(f)
        adj_ = [normalize(adj_)for adj_ in graph_2['adj']]
        adj_2 = [adj2edge_index(adj) for adj in adj_]
        feature_2 = graph_2['feature']
    
    with open(f'{DPATH}/individual_data/graph_' + str(year - 3) + '_nf.pkl', 'rb') as f:
        graph_3 = pkl.load(f)
        adj_ = [normalize(adj_)for adj_ in graph_3['adj']]
        adj_3 = [adj2edge_index(adj) for adj in adj_]
        feature_3 = graph_3['feature']

    with open(f'{DPATH}/individual_data/graph_' + str(year - 2) + '_nf.pkl', 'rb') as f:
        graph_4 = pkl.load(f)
        adj_ = [normalize(adj_)for adj_ in graph_4['adj']]
        adj_4 = [adj2edge_index(adj) for adj in adj_]
        feature_4 = graph_4['feature']

    with open(f'{DPATH}/individual_data/graph_' + str(year - 1) + '_nf.pkl', 'rb') as f:
        graph_5 = pkl.load(f)
        adj_ = [normalize(adj_)for adj_ in graph_5['adj']]
        adj_5 = [adj2edge_index(adj) for adj in adj_]
        feature_5 = graph_5['feature']
    
    edge_list = []
    for i in range(len(adj_1)):
        # tt = torch.stack([adj_1[0], adj_2[1], adj_3,adj_4,adj_5], dim=)
        edge_list.append( [adj_1[i], adj_2[i], adj_3[i], adj_4[i], adj_5[i] ] )

    print(feature_1.shape, feature_2.shape, feature_3.shape, feature_4.shape, feature_5.shape, )
    ls = (600391, 4)
    # for i in range(2):
    #     ls[i] = min(feature_1.shape[i], feature_2.shape[i], feature_3.shape[i], feature_4.shape[i], feature_5.shape[i])
    # ee = min(feature_1.shape[1], feature_2.shape[0], feature_3.shape[0], feature_4.shape[0], feature_5.shape[0])
    feature_np = np.stack([feature_1[:ls[0], :ls[1]], feature_2[:ls[0], :ls[1]], feature_3[:ls[0], :ls[1]],\
        feature_4[:ls[0], :ls[1]], feature_5[:ls[0], :ls[1]] ], axis=-1) # [n, 4, 5]
    feature = torch.from_numpy(feature_np).to(torch.float)

    if flag == "train":
        with open(f'{DPATH}/select_index_train.pkl','rb') as f:
            rank = pkl.load(f) # rank : shape=(3000,), [   2    7    9 ... 9009 9045 9062]

    elif flag == "test":
        with open(F'{DPATH}/select_index_test.pkl','rb') as f:
            rank = pkl.load(f)
    
    with open(f'{DPATH}/cumulative_log_labels_new.pkl','rb') as f: # len=13, 2005-2017的引用情况
        # 13个csv，每一列是year-2018的引用量
        labels = pkl.load(f)['P' + str(year) + '_label'].iloc[rank, 1:6].values # shape=(3000, 5)
    
    labels, label_max, label_min = lable_normorlization(labels)
    labels = torch.from_numpy(labels).to(torch.float)

    return edge_list, feature, labels, label_max, label_min, rank


def load_data_V13(year,flag):
    DPATH = '../07_HINTS_code-main/fanxing'
    # DPATH = '../07_HINTS_code-main/aminer'
    with open(f'{DPATH}/individual_data/graph_' + str(year - 5) + '_nf.pkl', 'rb') as f:
        graph_1 = pkl.load(f)
        # adj_ = [normalize(adj_)for adj_ in graph_1['adj']] # adj: list(sp.csr_matrix), 5个类型
        # adj_1 = [adj2edge_index(adj) for adj in adj_]
        adj_1 = [adj2edge_index(adj) for adj in graph_1['adj']]
        feature_1 = graph_1['feature'] # np.array, [n, 4]

    with open(f'{DPATH}/individual_data/graph_' + str(year - 4) + '_nf.pkl', 'rb') as f:
        graph_2 = pkl.load(f)
        # adj_ = [normalize(adj_)for adj_ in graph_2['adj']]
        # adj_2 = [adj2edge_index(adj) for adj in adj_]
        adj_2 = [adj2edge_index(adj) for adj in graph_2['adj']]
        feature_2 = graph_2['feature']
    
    with open(f'{DPATH}/individual_data/graph_' + str(year - 3) + '_nf.pkl', 'rb') as f:
        graph_3 = pkl.load(f)
        # adj_ = [normalize(adj_)for adj_ in graph_3['adj']]
        # adj_3 = [adj2edge_index(adj) for adj in adj_]
        adj_3 = [adj2edge_index(adj) for adj in graph_3['adj']]
        feature_3 = graph_3['feature']

    with open(f'{DPATH}/individual_data/graph_' + str(year - 2) + '_nf.pkl', 'rb') as f:
        graph_4 = pkl.load(f)
        # adj_ = [normalize(adj_)for adj_ in graph_4['adj']]
        # adj_4 = [adj2edge_index(adj) for adj in adj_]
        adj_4 = [adj2edge_index(adj) for adj in graph_4['adj']]
        feature_4 = graph_4['feature']

    with open(f'{DPATH}/individual_data/graph_' + str(year - 1) + '_nf.pkl', 'rb') as f:
        graph_5 = pkl.load(f)
        # adj_ = [normalize(adj_)for adj_ in graph_5['adj']]
        # adj_5 = [adj2edge_index(adj) for adj in adj_]
        adj_5 = [adj2edge_index(adj) for adj in graph_5['adj']]
        feature_5 = graph_5['feature']
    
    edge_list = []
    for i in range(len(adj_1)):
        # tt = torch.stack([adj_1[0], adj_2[1], adj_3,adj_4,adj_5], dim=)
        edge_list.append( [adj_1[i], adj_2[i], adj_3[i], adj_4[i], adj_5[i] ] )
    # edge_list = [adj_1,adj_2,adj_3,adj_4,adj_5]

    print(feature_1.shape, feature_2.shape, feature_3.shape, feature_4.shape, feature_5.shape, )
    ls = (750602, 4)
    # for i in range(2):
    #     ls[i] = min(feature_1.shape[i], feature_2.shape[i], feature_3.shape[i], feature_4.shape[i], feature_5.shape[i])
    # ee = min(feature_1.shape[1], feature_2.shape[0], feature_3.shape[0], feature_4.shape[0], feature_5.shape[0])
    feature_np = np.stack([feature_1[:ls[0], :ls[1]], feature_2[:ls[0], :ls[1]], feature_3[:ls[0], :ls[1]],\
        feature_4[:ls[0], :ls[1]], feature_5[:ls[0], :ls[1]] ], axis=-1) # [n, 4, 5]
    feature = torch.from_numpy(feature_np).to(torch.float)
    # feature_list=[feature_1,feature_2,feature_3,feature_4,feature_5]

    # if flag == "train":
    with open(f'../07_HINTS_code-main/aminer/select_index_train.pkl','rb') as f:
        rank_train = pkl.load(f) # rank : shape=(3000,), [   2    7    9 ... 9009 9045 9062]

# elif flag == "test":
    with open(F'../07_HINTS_code-main/aminer/select_index_test.pkl','rb') as f:
        rank_test = pkl.load(f)
    
    with open(f'{DPATH}/data/cumulative_log_labels_new.pkl','rb') as f: # len=13, 2005-2017的引用情况
        # 13个csv，每一列是year-2018的引用量
        df = pkl.load(f)['P' + str(year) + '_label']
        labels_train = df.iloc[rank_train, 1:6].values # shape=(3000, 5)
        labels_test = df.iloc[rank_test, 1:6].values # shape=(3000, 5)

    
    # labels, label_max, label_min = lable_normorlization(labels)
    labels_train = torch.from_numpy(labels_train).to(torch.float)
    labels_test = torch.from_numpy(labels_test).to(torch.float)

    # return edge_list, feature_list, labels_train, labels_test, rank_train, rank_test
    return edge_list, feature, labels_train, labels_test, rank_train, rank_test


def load_data_V11(year,flag):
    # DPATH = '../07_HINTS_code-main/fanxing'
    DPATH = '../07_HINTS_code-main/aminer'
    with open(f'{DPATH}/individual_data/graph_' + str(year - 5) + '_nf.pkl', 'rb') as f:
        graph_1 = pkl.load(f)
        # adj_ = [normalize(adj_)for adj_ in graph_1['adj']] # adj: list(sp.csr_matrix), 5个类型
        # adj_1 = [adj2edge_index(adj) for adj in adj_]
        adj_1 = [adj2edge_index(adj) for adj in graph_1['adj']]
        feature_1 = graph_1['feature'] # np.array, [n, 4]

    with open(f'{DPATH}/individual_data/graph_' + str(year - 4) + '_nf.pkl', 'rb') as f:
        graph_2 = pkl.load(f)
        # adj_ = [normalize(adj_)for adj_ in graph_2['adj']]
        # adj_2 = [adj2edge_index(adj) for adj in adj_]
        adj_2 = [adj2edge_index(adj) for adj in graph_2['adj']]
        feature_2 = graph_2['feature']
    
    with open(f'{DPATH}/individual_data/graph_' + str(year - 3) + '_nf.pkl', 'rb') as f:
        graph_3 = pkl.load(f)
        # adj_ = [normalize(adj_)for adj_ in graph_3['adj']]
        # adj_3 = [adj2edge_index(adj) for adj in adj_]
        adj_3 = [adj2edge_index(adj) for adj in graph_3['adj']]
        feature_3 = graph_3['feature']

    with open(f'{DPATH}/individual_data/graph_' + str(year - 2) + '_nf.pkl', 'rb') as f:
        graph_4 = pkl.load(f)
        # adj_ = [normalize(adj_)for adj_ in graph_4['adj']]
        # adj_4 = [adj2edge_index(adj) for adj in adj_]
        adj_4 = [adj2edge_index(adj) for adj in graph_4['adj']]
        feature_4 = graph_4['feature']

    with open(f'{DPATH}/individual_data/graph_' + str(year - 1) + '_nf.pkl', 'rb') as f:
        graph_5 = pkl.load(f)
        # adj_ = [normalize(adj_)for adj_ in graph_5['adj']]
        # adj_5 = [adj2edge_index(adj) for adj in adj_]
        adj_5 = [adj2edge_index(adj) for adj in graph_5['adj']]
        feature_5 = graph_5['feature']
    
    edge_list = []
    for i in range(len(adj_1)):
        # tt = torch.stack([adj_1[0], adj_2[1], adj_3,adj_4,adj_5], dim=)
        edge_list.append( [adj_1[i], adj_2[i], adj_3[i], adj_4[i], adj_5[i] ] )
    # edge_list = [adj_1,adj_2,adj_3,adj_4,adj_5]

    print(feature_1.shape, feature_2.shape, feature_3.shape, feature_4.shape, feature_5.shape, )
    ls = (600391, 4)
    # for i in range(2):
    #     ls[i] = min(feature_1.shape[i], feature_2.shape[i], feature_3.shape[i], feature_4.shape[i], feature_5.shape[i])
    # ee = min(feature_1.shape[1], feature_2.shape[0], feature_3.shape[0], feature_4.shape[0], feature_5.shape[0])
    feature_np = np.stack([feature_1[:ls[0], :ls[1]], feature_2[:ls[0], :ls[1]], feature_3[:ls[0], :ls[1]],\
        feature_4[:ls[0], :ls[1]], feature_5[:ls[0], :ls[1]] ], axis=-1) # [n, 4, 5]
    feature = torch.from_numpy(feature_np).to(torch.float)
    # feature_list=[feature_1,feature_2,feature_3,feature_4,feature_5]

    # if flag == "train":
    with open(f'../07_HINTS_code-main/aminer/select_index_train.pkl','rb') as f:
        rank_train = pkl.load(f) # rank : shape=(3000,), [   2    7    9 ... 9009 9045 9062]

# elif flag == "test":
    with open(F'../07_HINTS_code-main/aminer/select_index_test.pkl','rb') as f:
        rank_test = pkl.load(f)
    
    with open(f'{DPATH}/cumulative_log_labels_new.pkl','rb') as f: # len=13, 2005-2017的引用情况
        # 13个csv，每一列是year-2018的引用量
        df = pkl.load(f)['P' + str(year) + '_label']
        labels_train = df.iloc[rank_train, 1:6].values # shape=(3000, 5)
        labels_test = df.iloc[rank_test, 1:6].values # shape=(3000, 5)

    
    # labels, label_max, label_min = lable_normorlization(labels)
    labels_train = torch.from_numpy(labels_train).to(torch.float)
    labels_test = torch.from_numpy(labels_test).to(torch.float)

    # return edge_list, feature_list, labels_train, labels_test, rank_train, rank_test
    return edge_list, feature, labels_train, labels_test, rank_train, rank_test
import os
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch

# 继承InMemoryDataset
class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis', 
                 xd=None, xt=None, y=None, transform=None,
                 pre_transform=None,smile_graph=None):

        #root is required for save preprocessed data, default is '/tmp'
        # root用于保存预处理的数据，默认为tmp
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # super继承父类中的作用
        # benchmark dataset, default = 'davis'
        # 使用davis数据集
        self.dataset = dataset

        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
            # 加载数据
            # 如果有预处理数据，进行处理，没有的话提示，并且进行process操作再处理
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt, y,smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])
            # 加载数据

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']
# 处理过的数据保存在哪里


    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass
# 数据下载到哪里

    def _process(self):
        if not os.path.exists(self.processed_dir):

            # 就是判断括号里的文件是否存在的意思，括号内的可以是文件路径。
            os.makedirs(self.processed_dir)
            # os.makedirs()方法用于递归创建目录，如果子目录创建失败或者已经存在，会抛出一个OSError的异常



    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs
    # XD - list of SMILES, XT: list of encoded target (categorical or one-hot),
    # Y: list of labels (i.e. affinity)
    # Return: PyTorch-Geometric format processed data

    def process(self, xd, xt, y,smile_graph):
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        # assert用来报错的语句

        data_list = []
        data_len = len(xd)
        for i in range(data_len):
            print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smiles = xd[i]
            target = xt[i]
            labels = y[i]

            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smiles]

            # make the graph ready for PyTorch Geometrics GCN algorithms:

            # transpose(1, 0)，转置
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.FloatTensor([labels]))

            GCNData.target = torch.LongTensor([target])
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # _setitem__(self, key, value)：该方法应该按一定的方式存储和key相关的value。在设置类实例属性时自动调用的。

            # append graph, label and target sequence to data list
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
            # 其次，如果要对数据做过滤的话，我们执行数据过滤的过程。
            # 接着，如果要对数据做处理的话，我们执行数据处理的过程。

        print('Graph construction done. Saving to file.')

        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])
        #  collate()
        # 函数接收一个列表的Data对象，
        # 返回合并后的Data对象以及用于从合并后的Data对象重构各个原始Data对象的切片字典slices。
        # 最后我们将这个巨大的Data对象和切片字典slices保存到文件。

# 评估参数,rmse,mse,person,spearman,ci
def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs
def ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci
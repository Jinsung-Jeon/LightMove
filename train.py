# coding: utf-8
from __future__ import print_function
from __future__ import division

import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd 
import sys
import numpy as np
import pickle
from collections import deque, Counter, defaultdict
from tqdm import tqdm
GRID_COUNT = 100
def geo_grade(index, x, y, m_nGridCount=GRID_COUNT):  # index: [pids], x: [lon], y: [lat]. 100 by 100
    dXMax, dXMin, dYMax, dYMin = max(x), min(x), max(y), min(y)
    # print dXMax, dXMin, dYMax, dYMin
    m_dOriginX = dXMin
    m_dOriginY = dYMin
    dSizeX = (dXMax - dXMin) / m_nGridCount
    dSizeY = (dYMax - dYMin) / m_nGridCount
    m_vIndexCells = []  # list of lists
    center_location_list = []
    for i in range(0, m_nGridCount * m_nGridCount + 1):
        m_vIndexCells.append([])
        y_ind = int(i / m_nGridCount)
        x_ind = i - y_ind * m_nGridCount
        center_location_list.append((dXMin + x_ind * dSizeX + 0.5 * dSizeX, dYMin + y_ind * dSizeY + 0.5 * dSizeY))
    # print (m_nGridCount, m_dOriginX, m_dOriginY, \
    #        dSizeX, dSizeY, len(m_vIndexCells), len(index))
    poi_index_dict = {}
    _poi_index_dict = defaultdict(list)
    for i in range(len(x)):
        nXCol = int((x[i] - m_dOriginX) / dSizeX)
        nYCol = int((y[i] - m_dOriginY) / dSizeY)
        if nXCol >= m_nGridCount:
            # print 'max X'
            nXCol = m_nGridCount - 1

        if nYCol >= m_nGridCount:
            # print 'max Y'
            nYCol = m_nGridCount - 1

        iIndex = nYCol * m_nGridCount + nXCol
        poi_index_dict[index[i]] = iIndex  # key: raw poi, val: grid id
        _poi_index_dict[iIndex].append(index[i])  # key: grid id, val: raw pid
        m_vIndexCells[iIndex].append([index[i], x[i], y[i]])
    
    return poi_index_dict, center_location_list
    # return poi_index_dict, _poi_index_dict

def normalize(lon_lat):
    # normalize longitude and latitude
    lon = np.array([l[0] for l in lon_lat])
    lon_m = np.mean(lon)
    lon_s = np.std(lon)
    lon_norm = [(l-lon_m)/lon_s for l in lon]

    lat = np.array([l[1] for l in lon_lat])
    lat_m = np.mean(lat)
    lat_s = np.std(lat)
    lat_norm = [(l-lat_m)/lat_s for l in lat]

    lon_lat_norm = list(zip(lon_norm, lat_norm))
    norm_dict = {}
    for i in range(len(lon_lat)):
        norm_dict[tuple(lon_lat[i])] = lon_lat_norm[i]
    return norm_dict


class RnnParameterData(object):
    def __init__(self, loc_emb_size=500, uid_emb_size=40, voc_emb_size=50, tim_emb_size=10, hidden_size=500,
                 lr=1e-3, lr_step=3, lr_decay=0.1, dropout_p=0.5, L2=1e-5, clip=5.0, optim='Adam',
                 history_mode='avg', attn_type='dot', epoch_max=30, rnn_type='LSTM', model_mode="simple",
                 data_path='../data/', save_path='../results/', data_name='foursquare',model_method='test3', pretrain=0):
        self.data_path = data_path
        self.save_path = save_path
        self.data_name = data_name
        #data = pickle.load(open(self.data_path + self.data_name + '.pk', 'rb'))
        with open(self.data_path + self.data_name + '.pk', 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            data = u.load()
        self.vid_list = data['vid_list']  # key: raw poi, val: [int vid, number of visits]
        # pickle.dump(self.vid_list, open(self.data_name + '_pid.pk', 'wb'))
        # sys.exit()
        self.uid_list = data['uid_list']  # key: raw uid, val: [int uid, number of sessions]
        self.data_neural = data['data_neural']
        self.data_filter = data['data_filter']  # key: raw uid, val: {sessions: {sid: [[raw pid, raw tim], ...]}, ...}
        self.vid_lookup = data['vid_lookup']  # key: int vid, val: [float(lon), float(lat)]
        self.uid_lookup = {}  # key: int uid, val: raw uid
        for k, v in self.uid_list.items():
            self.uid_lookup[v[0]] = k
        self.raw_pid = [p for p in self.vid_list.keys() if p != 'unk']  # list of raw pois
        self._int_vid = [self.vid_list[p][0] for p in self.raw_pid]  # list of int vids
        self._raw_xy = [self.vid_lookup[i] for i in self._int_vid]  # list of [lon, lat]
        _raw_xy_norm = normalize(self._raw_xy)  # key: (lon, lat), val: standardized (lon, lat)
        self._vid_lookup_norm =  {}  # key: int pid, val: standardized (lon, lat)
        for v, l in self.vid_lookup.items():
            self._vid_lookup_norm[v] = _raw_xy_norm[tuple(l)]
        self.raw_x = [i[0] for i in self._raw_xy]
        self.raw_y = [i[1] for i in self._raw_xy]
        self._raw_grid_lookup, self.center_location_list = geo_grade(self.raw_pid, self.raw_x, self.raw_y)  # key: raw pid, val: grid id
        self.grid_lookup = {}  # key: int vid, val: grid id
        for k, v in self.vid_list.items():  # k: raw poi, v[0]: int pid
            if k == 'unk':
                continue
            self.grid_lookup[v[0]] = self._raw_grid_lookup[k]

        self.tim_size = 48
        self.loc_size = len(self.vid_list)
        self.uid_size = len(self.uid_list)
        self.loc_emb_size = loc_emb_size
        self.tim_emb_size = tim_emb_size
        self.voc_emb_size = voc_emb_size
        self.uid_emb_size = uid_emb_size
        self.hidden_size = hidden_size
        self.model_method = model_method
        self.epoch = epoch_max
        self.dropout_p = dropout_p
        self.use_cuda = True
        self.lr = lr
        self.lr_step = lr_step
        self.lr_decay = lr_decay
        self.optim = optim
        self.L2 = L2
        self.clip = clip

        self.attn_type = attn_type
        self.rnn_type = rnn_type
        self.history_mode = history_mode
        self.model_mode = model_mode

        self.pretrain = pretrain
    def write_tsv(self):
        # raw train/valid/test data
        raw_train = defaultdict(list) # key: uid, val: list([[lon, lat], tim, pid, grid_id])
        raw_valid = defaultdict(list)
        raw_test = defaultdict(list)
        for u in self.data_filter.keys():  # raw uid
            data = self.data_neural[self.uid_list[u][0]]   # data = {sid: [[int pid, int tid]]}
            for sid in data['train']:  # list of train sessions
                session = self.data_filter[u]['sessions'][sid]  # [[raw pid, raw tim]]
                for i in range(len(session)):
                    record = session[i]
                    raw_poi = record[0]
                    int_pid = self.vid_list[raw_poi][0]
                    grid_id = self.grid_lookup[int_pid]
                    raw_train[u].append([self.vid_lookup[int_pid], record[1], int_pid, grid_id])
                    # raw_train[u].append([self._vid_lookup_norm[int_pid], record[1], int_pid, grid_id])
            for sid in data['test']:  # list of test sessions
                session = self.data_filter[u]['sessions'][sid]  # [[raw pid, raw tim]]
                for i in range(len(session)):
                    record = session[i]
                    raw_poi = record[0]
                    int_pid = self.vid_list[raw_poi][0]
                    grid_id = self.grid_lookup[int_pid]
                    raw_test[u].append([self.vid_lookup[int_pid], record[1], int_pid, grid_id])
                    # raw_test[u].append([self._vid_lookup_norm[int_pid], record[1], int_pid, grid_id])
            for sid in data['valid']:  # list of test sessions
                session = self.data_filter[u]['sessions'][sid]  # [[raw pid, raw tim]]
                for i in range(len(session)):
                    record = session[i]
                    raw_poi = record[0]
                    int_pid = self.vid_list[raw_poi][0]
                    grid_id = self.grid_lookup[int_pid]
                    raw_valid[u].append([self.vid_lookup[int_pid], record[1], int_pid, grid_id])
                    # raw_valid[u].append([self._vid_lookup_norm[int_pid], record[1], int_pid, grid_id])

        # write on files
        w_train = open(self.data_name + '_train.tsv', 'w')
        # l = '\t'.join(['uid', 'lon', 'lat', 'tim', 'filtered_grid'])
        l = '\t'.join(['uid', 'lon', 'lat', 'tim', 'pid', 'grid_id'])
        w_train.write(l + '\n')
        for key, val in raw_train.items():  # key: uid, val: list([[lon, lat], tim, int pid, grid_id])
            for v in val:  # list([[lon, lat], tim])
                # l = '\t'.join([str(key), str(v[0][0]), str(v[0][1]), str(v[1]), str(v[2])])
                l = '\t'.join([str(key), str(v[0][0]), str(v[0][1]), str(v[1]), str(v[2]), str(v[3])])
                w_train.write(l + '\n')
        w_test = open(self.data_name + '_test.tsv', 'w')
        l = '\t'.join(['uid', 'lon', 'lat', 'tim', 'pid', 'grid_id'])
        # l = '\t'.join(['uid', 'lon', 'lat', 'tim', 'filtered_grid'])
        w_test.write(l + '\n')
        for key, val in raw_test.items():  # key: uid, val: list([[lon, lat], tim, int pid, grid_id])
            for v in val:  # [[lon, lat], tim]
                l = '\t'.join([str(key), str(v[0][0]), str(v[0][1]), str(v[1]), str(v[2]), str(v[3])])
                # l = '\t'.join([str(key), str(v[0][0]), str(v[0][1]), str(v[1]), str(v[2])])
                w_test.write(l + '\n')
        w_valid = open(self.data_name + '_valid.tsv', 'w')
        l = '\t'.join(['uid', 'lon', 'lat', 'tim', 'pid', 'grid_id'])
        # l = '\t'.join(['uid', 'lon', 'lat', 'tim', 'filtered_grid'])
        w_valid.write(l + '\n')
        for key, val in raw_valid.items():  # key: uid, val: list([[lon, lat], tim, int pid, grid_id])
            for v in val:  # [[lon, lat], tim]
                l = '\t'.join([str(key), str(v[0][0]), str(v[0][1]), str(v[1]), str(v[2]), str(v[3])])
                # l = '\t'.join([str(key), str(v[0][0]), str(v[0][1]), str(v[1]), str(v[2])])
                w_valid.write(l + '\n')
        w_train.close()
        w_test.close()
        w_valid.close()

def generate_input_history(data_neural, mode, mode2=None, candidate=None):
    data_train = {}
    train_idx = {}
    if candidate is None:
        candidate = data_neural.keys()
    for u in candidate:
        sessions = data_neural[u]['sessions']
        train_id = data_neural[u][mode]
        data_train[u] = {}
        for c, i in enumerate(train_id):
            if mode == 'train' and c == 0:
                continue
            session = sessions[i]
            trace = {}
            loc_np = np.reshape(np.array([s[0] for s in session[:-1]]), (len(session[:-1]), 1))
            tim_np = np.reshape(np.array([s[1] for s in session[:-1]]), (len(session[:-1]), 1))
            # voc_np = np.reshape(np.array([s[2] for s in session[:-1]]), (len(session[:-1]), 27))
            target = np.array([s[0] for s in session[1:]])
            trace['loc'] = Variable(torch.LongTensor(loc_np))
            trace['target'] = Variable(torch.LongTensor(target))
            trace['tim'] = Variable(torch.LongTensor(tim_np))
            # trace['voc'] = Variable(torch.LongTensor(voc_np))

            history = []
            if mode == 'test':
                test_id = data_neural[u]['train']
                for tt in test_id:
                    history.extend([(s[0], s[1]) for s in sessions[tt]])
            for j in range(c):
                history.extend([(s[0], s[1]) for s in sessions[train_id[j]]])
            history = sorted(history, key=lambda x: x[1], reverse=False)

            # merge traces with same time stamp
            if mode2 == 'max':
                history_tmp = {}
                for tr in history:
                    if tr[1] not in history_tmp:
                        history_tmp[tr[1]] = [tr[0]]
                    else:
                        history_tmp[tr[1]].append(tr[0])
                history_filter = []
                for t in history_tmp:
                    if len(history_tmp[t]) == 1:
                        history_filter.append((history_tmp[t][0], t))
                    else:
                        tmp = Counter(history_tmp[t]).most_common()
                        if tmp[0][1] > 1:
                            history_filter.append((history_tmp[t][0], t))
                        else:
                            ti = np.random.randint(len(tmp))
                            history_filter.append((tmp[ti][0], t))
                history = history_filter
                history = sorted(history, key=lambda x: x[1], reverse=False)
            elif mode2 == 'avg':
                history_tim = [t[1] for t in history]
                history_count = [1]
                last_t = history_tim[0]
                count = 1
                for t in history_tim[1:]:
                    if t == last_t:
                        count += 1
                    else:
                        history_count[-1] = count
                        history_count.append(1)
                        last_t = t
                        count = 1
            history_loc = np.reshape(np.array([s[0] for s in history]), (len(history), 1))
            history_tim = np.reshape(np.array([s[1] for s in history]), (len(history), 1))
            trace['history_loc'] = Variable(torch.LongTensor(history_loc))
            trace['history_tim'] = Variable(torch.LongTensor(history_tim))
            if mode2 == 'avg':
                trace['history_count'] = history_count

            data_train[u][i] = trace
        train_idx[u] = train_id
    return data_train, train_idx


def generate_input_long_history2(data_neural, mode, candidate=None):
    data_train = {}
    train_idx = {}
    if candidate is None:
        candidate = data_neural.keys()
    for u in candidate:
        sessions = data_neural[u]['sessions']
        train_id = data_neural[u][mode]
        data_train[u] = {}

        trace = {}
        session = []
        for c, i in enumerate(train_id):
            session.extend(sessions[i])
        target = np.array([s[0] for s in session[1:]])

        loc_tim = []
        loc_tim.extend([(s[0], s[1]) for s in session[:-1]])
        loc_np = np.reshape(np.array([s[0] for s in loc_tim]), (len(loc_tim), 1))
        tim_np = np.reshape(np.array([s[1] for s in loc_tim]), (len(loc_tim), 1))
        trace['loc'] = Variable(torch.LongTensor(loc_np))
        trace['tim'] = Variable(torch.LongTensor(tim_np))
        trace['target'] = Variable(torch.LongTensor(target))
        data_train[u][i] = trace
        # train_idx[u] = train_id
        if mode == 'train':
            train_idx[u] = [0, i]
        else:
            train_idx[u] = [i]
    return data_train, train_idx


def generate_input_long_history(data_neural, mode, candidate=None, grid_train=False, grid=None, data_name=None, raw_uid=None, raw_sess=None):
    try:
        if grid_train and not grid:
            raise ValueError
    except ValueError:
        sys.exit('grid lookup table should be given in grid train mode')
    data_train = {}
    train_idx = {}  # key: int uid, val: list of train sids 
    if data_name:  # write tsv mode
        if grid:
            w = open(data_name+'_grid.tsv', 'w')
        else: 
            w = open(data_name+'.tsv', 'w')
        ww = '\t'.join(['uid','target[(pid, tim)]'])
        w.write(ww + '\n')
    if candidate is None:
        candidate = data_neural.keys()  # uids
    for u in candidate:
        sessions = data_neural[u]['sessions']  # {sid: [[vid, tid]]}
        if grid_train:  # train with grid id instead of vid
            _sessions = {}
            for k, v in sessions.items():
                _sessions[k] = [[grid[vt[0]], vt[1]] for vt in v]
            sessions = _sessions  # {sid: [[gid, tid]]}
        if data_name:
            raw_u = raw_uid[u]
            raw_sessions = raw_sess[raw_u]['sessions']
        train_id = data_neural[u][mode]  # list of train sid | test sid
        data_train[u] = {}  # key: sid, val: trace
        for c, i in enumerate(train_id):
            trace = {}  # key: 'loc', 'tim', 'target', val: tensor(int vid), tensor(int tid), tensor(int vid) 
            if mode == 'train' and c == 0:  # skip very first train sid
                continue
            session = sessions[i]  # [[vid, tid]]
            target = np.array([s[0] for s in session[1:]])  # [vid]
            if data_name:
                raw_s = raw_sessions[i]
                raw_target = [s[0] for s in raw_s[1:]]  # [raw pid]
                if grid:  # write grid id
                    raw_target = [grid[s] for s in target]
                raw_target_tim = [s[1] for s in raw_s[1:]]  # [raw tim]

            history = []
            if data_name:
                raw_history = []
            if mode == 'test':  # extend train data
                test_id = data_neural[u]['train'] + data_neural[u]['valid']
                for tt in test_id:
                    history.extend([(s[0], s[1]) for s in sessions[tt]])  # train records s
                    if data_name:
                        raw_history.extend([(s[0], s[1]) for s in raw_sessions[tt]])  # raw train records s
            elif mode == 'valid':  # extend train data
                test_id = data_neural[u]['train']
                for tt in test_id:
                    history.extend([(s[0], s[1]) for s in sessions[tt]])  # train records s
                    if data_name:
                        raw_history.extend([(s[0], s[1]) for s in raw_sessions[tt]])  # raw train records s
            for j in range(c):
                history.extend([(s[0], s[1]) for s in sessions[train_id[j]]])  # cumulative [vid, tid] until the last sid
                if data_name:
                    raw_history.extend([(s[0], s[1]) for s in raw_sessions[train_id[j]]])

            history_tim = [t[1] for t in history]
            history_count = [1]  # frequency of tids
            last_t = history_tim[0]
            count = 1
            for t in history_tim[1:]:
                if t == last_t:
                    count += 1
                else:
                    history_count[-1] = count
                    history_count.append(1)
                    last_t = t
                    count = 1

            history_loc = np.reshape(np.array([s[0] for s in history]), (len(history), 1))
            history_tim = np.reshape(np.array([s[1] for s in history]), (len(history), 1))
            trace['history_loc'] = Variable(torch.LongTensor(history_loc))
            trace['history_tim'] = Variable(torch.LongTensor(history_tim))
            trace['history_count'] = history_count

            loc_tim = history  # [vid, tid]
            loc_tim.extend([(s[0], s[1]) for s in session[:-1]])  # extend current session just before the last record
            loc_np = np.reshape(np.array([s[0] for s in loc_tim]), (len(loc_tim), 1))
            tim_np = np.reshape(np.array([s[1] for s in loc_tim]), (len(loc_tim), 1))
            trace['loc'] = Variable(torch.LongTensor(loc_np))
            trace['tim'] = Variable(torch.LongTensor(tim_np))
            trace['target'] = Variable(torch.LongTensor(target))
            data_train[u][i] = trace  # history_loc, history_tim, history_count, loc
            if data_name:
                # # history input
                # raw_loc_tim = raw_history  # [vid, tid]
                # raw_loc_tim.extend([(s[0], s[1]) for s in raw_s[:-1]])  # until recently
                # raw_loc_np = [s[0] for s in raw_loc_tim]
                # raw_tim_np = [s[1] for s in raw_loc_tim]
                # x = zip(raw_loc_np, raw_tim_np)
                # if grid:
                #     grid_loc_np = [grid[i] for i in [s[0] for s in loc_tim]]
                #     x = zip(grid_loc_np, raw_tim_np)
                y = zip(raw_target, raw_target_tim)  # [raw pid, raw tim]
                ww = '\t'.join([str(raw_u), str(y)])
                w.write(ww + '\n')
        train_idx[u] = train_id
    if data_name:
        w.close()
    return data_train, train_idx


def generate_queue(train_idx, mode, mode2):
    """return a deque. You must use it by train_queue.popleft()"""
    user = train_idx.keys()
    train_queue = deque()
    if mode == 'random':
        initial_queue = {}
        user = list(user)
        for u in user:
            if mode2 == 'train':
                initial_queue[u] = deque(train_idx[u][1:])  # train_id
            else:
                initial_queue[u] = deque(train_idx[u])
        queue_left = 1
        while queue_left > 0:
            np.random.shuffle(user)
            for j, u in enumerate(user):
                if len(initial_queue[u]) > 0:
                    train_queue.append((u, initial_queue[u].popleft()))
                if j >= int(0.01 * len(user)):
                    break
            queue_left = sum([1 for x in initial_queue if len(initial_queue[x]) > 0])
    elif mode == 'normal':
        for u in user:
            for i in train_idx[u]:
                train_queue.append((u, i))
    return train_queue


def get_acc(target, scores, grid=None):
    """target and scores are torch cuda Variable"""
    target = target.data.cpu().numpy()
    #scores = scores.squeeze(1)

    val, idxx = scores.data.topk(10, 1)  # top 10 predictions
    predx = idxx.cpu().numpy()
    acc = np.zeros((3, 1))
    for i, p in enumerate(predx):  # enumerate for the number of targets
        t = target[i]
        if grid:  # grid evaluation mode
            t = grid[t]
            p = [grid[i] for i in p]
        if t in p[:10] and t > 0:
            acc[0] += 1  # top10
        if t in p[:5] and t > 0:
            acc[1] += 1  # top5
        if t == p[0] and t > 0:
            acc[2] += 1  # top1
    return acc

def get_mrr(target, scores):
    val, idxx = scores.data.topk(scores.data.shape[1], 1)  # top 10 predictions
    target = target.data.cpu().numpy()
    idxx = idxx.cpu().numpy()
    mrr = .0
    for i in range(len(target)):
        rank = np.where(target[i] == idxx[i])
        mrr += (1.0/(rank[0]+1))
    return mrr

def dcg_at_k(r, k, method=1):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=1):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def get_hint(target, scores, users_visited):
    """target and scores are torch cuda Variable"""
    target = target.data.cpu().numpy()
    val, idxx = scores.data.topk(1, 1)
    predx = idxx.cpu().numpy()
    hint = np.zeros((3,))
    count = np.zeros((3,))
    count[0] = len(target)
    for i, p in enumerate(predx):
        t = target[i]
        if t == p[0] and t > 0:
            hint[0] += 1
        if t in users_visited:
            count[1] += 1
            if t == p[0] and t > 0:
                hint[1] += 1
        else:
            count[2] += 1
            if t == p[0] and t > 0:
                hint[2] += 1
    return hint, count

def twosomeplot(id,target, scores, look, ax):
    id = id.cpu().numpy()[0]
    target = target.data.cpu().numpy()
    _, idxx = scores.data.topk(1, 1)  # top 10 predictions
    predx = idxx.cpu().numpy().flatten()
    
    target_x = look.iloc[target-1,0]
    target_y = look.iloc[target-1,1]

    pred_x = look.iloc[predx-1,0]
    pred_y = look.iloc[predx-1,1]
    ax.scatter(target_x, target_y,color='red', label='Target')
    ax.scatter(pred_x, pred_y, color='blue',marker='x', label='Prediction')
    #BBox = (126,6, 126.95,37.2, 37.8)

    #plt.scatter(target_x, target_y,color='red', label='Target')
    #plt.scatter(pred_x, pred_y, color='blue',marker='x', label='Prediction')
    #plt.title('Prediction of user {}\' trajectory'.format(id+1), fontsize=14, pad=10)
    #plt.xlabel('Longitude',labelpad=10)
    #plt.ylabel('Latitude',labelpad=10)
    #plt.savefig('/home/jinsung/DeepMove/codes/deepmove_png_detail/{id}.png'.format(id=id, session=session), dpi=300)

def run_simple(data, run_idx, mode, lr, clip, model, optimizer, criterion, mode2=None, grid_eval=False, grid=None, center=None,look=None):
    """mode=train: return model, avg_loss
       mode=test: return avg_loss,avg_acc,users_rnn_acc"""
    try:
        if (grid_eval and not grid):
            raise ValueError
    except ValueError:
        sys.exit('grid lookup table should be given in grid evaluation mode')

    run_queue = None
    if mode == 'train':
        model.train(True)
        run_queue = generate_queue(run_idx, 'random', 'train')  # train sid
    elif mode == 'test':
        model.train(False)
        run_queue = generate_queue(run_idx, 'normal', 'test')  # test sid
    total_loss = []
    queue_len = len(run_queue)

    users_acc = {}
    users_mrr = {}

    look_df = pd.DataFrame(look)
    look_df_t = look_df.transpose()
    users_acc = {}
    users_mrr = {}
    prev_u = 0


    for c in tqdm(range(queue_len)):
        optimizer.zero_grad()

        u, i = run_queue.popleft()  # uid, train|test sid

        
        if u not in users_acc:
            users_acc[u] = [0, 0, 0, 0]  # [total # of target records, top1, top5, top10]
            users_mrr[u] = 0  # initiate'

        loc = data[u][i]['loc'].cuda()  # cumulative loc up to session_i[-1]
        tim = data[u][i]['tim'].cuda()  # cumulative tim up to session_i[-1]
        target = data[u][i]['target'].cuda()  # [vid], answer vid of session_i[1:]
        uid = Variable(torch.LongTensor([u])).cuda()

        if 'attn' in mode2:
            history_loc = data[u][i]['history_loc'].cuda()
            history_tim = data[u][i]['history_tim'].cuda()

        if mode2 in ['simple', 'simple_long']:
            scores = model(loc, tim)
        elif mode2 == 'attn_avg_long_user':
            history_count = data[u][i]['history_count']
            target_len = target.data.size()[0]
            scores = model(loc, tim, history_loc, history_tim, history_count, uid, target_len)
        elif mode2 == 'attn_local_long':
            target_len = target.data.size()[0]
            scores = model(loc, tim, target_len)
        elif mode2 == 'flashback':
            target_len = target.data.size()[0]
            scores = model(loc, tim, target_len)

        if scores.data.size()[0] > target.data.size()[0]:
            scores = scores[-target.data.size()[0]:]

        loss = criterion(scores, target)
        if mode == 'train':
            loss.backward()
            # gradient clipping
            try:
                torch.nn.utils.clip_grad_norm(model.parameters(), clip)
                for p in model.parameters():
                    if p.requires_grad:
                        p.data.add_(-lr, p.grad.data)
            except:
                pass
            optimizer.step()
        elif mode == 'test':
            users_acc[u][0] += len(target)
            acc = get_acc(target, scores)
            if grid_eval:
                acc = get_acc(target, scores, grid=grid)
            users_acc[u][1] += acc[2]  # top1 Acc
            users_acc[u][2] += acc[1]  # top5 Acc
            users_acc[u][3] += acc[0]  # top10 Acc

            users_mrr[u] += get_mrr(target, scores)

        prev_u = uid.item()
        

        total_loss.append(loss.data.cpu().numpy())

    avg_loss = np.mean(total_loss, dtype=np.float64)
    if mode == 'train':
        return model, avg_loss
    elif mode == 'test':
        users_rnn_acc = {}
        for u in users_acc:
            top1_acc = users_acc[u][1] / users_acc[u][0]  # user u's top1 accuracy
            top5_acc = users_acc[u][2] / users_acc[u][0]  # user u's top5 accuracy
            top10_acc = users_acc[u][3] / users_acc[u][0]  # user u's top10 accuracy
            mrr = users_mrr[u] / users_acc[u][0]  # user's average mrr
            users_rnn_acc[u] = (top1_acc.tolist()[0], top5_acc.tolist()[0], top10_acc.tolist()[0], mrr)  # top1, top5, top10, mrr

        avg_acc = np.mean([users_rnn_acc[x][0] for x in users_rnn_acc])  # average top1 accuracy
        return avg_loss, avg_acc, users_rnn_acc


def markov(parameters, candidate):
    print("This is Markov!")
    validation = {}
    for u in candidate:
        traces = parameters.data_neural[u]['sessions']
        train_id = parameters.data_neural[u]['train']
        test_id = parameters.data_neural[u]['test']
        trace_train = []
        for tr in train_id:
            trace_train.append([t[0] for t in traces[tr]])
        locations_train = []
        for t in trace_train:
            locations_train.extend(t)
        trace_test = []
        for tr in test_id:
            trace_test.append([t[0] for t in traces[tr]])
        locations_test = []
        for t in trace_test:
            locations_test.extend(t)
        validation[u] = [locations_train, locations_test]
    acc = 0
    count = 0
    user_acc = {}
    for u in validation.keys():
        topk = list(set(validation[u][0]))
        # print("\ntop k : ", topk)
        transfer = np.zeros((len(topk), len(topk)))

        # train
        sessions = parameters.data_neural[u]['sessions']
        train_id = parameters.data_neural[u]['train']
        for i in train_id:
            for j, s in enumerate(sessions[i][:-1]):
                loc = s[0]
                target = sessions[i][j + 1][0]
                if loc in topk and target in topk:
                    r = topk.index(loc)
                    c = topk.index(target)
                    transfer[r, c] += 1
        for i in range(len(topk)):
            tmp_sum = np.sum(transfer[i, :])
            if tmp_sum > 0:
                transfer[i, :] = transfer[i, :] / tmp_sum

        # validation
        user_count = 0
        user_acc[u] = 0
        test_id = parameters.data_neural[u]['test']
        for i in test_id:
            for j, s in enumerate(sessions[i][:-1]):
                # print(j, sessions[i][:-1])
                loc = s[0]
                target = sessions[i][j + 1][0]
                # print(target)
                count += 1
                user_count += 1
                if loc in topk:
                    # print("\nloc in topk -> transfer : ", transfer[topk.index(loc), :])
                    pred5 = transfer[topk.index(loc), :].argsort()[-5:][::-1]
                    # pred = np.argmax(transfer[topk.index(loc), :])
                    # import pdb
                    # pdb.set_trace()
                    if np.min(pred5) >= len(topk) - 1:
                        pred2 = np.random.randint(len(topk))
                        #print("\nrandom : \n", pred)

                    pred2 = [topk[x] for x in pred5]
                    if target in pred2:
                        acc += 1
                        user_acc[u] += 1
        user_acc[u] = user_acc[u] / user_count
    avg_acc = np.mean([user_acc[u] for u in user_acc])
    return avg_acc, user_acc

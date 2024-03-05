import h5py
import torch
import random
import numpy as np
from torch.nn import functional as F
from scipy.interpolate import interp1d


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')

    return out


def top_p_probs(probs, p):
    out = probs.clone()

    sorted_probs, sorted_indices = torch.sort(out, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    out[indices_to_remove] = 0

    return out


# top-p + top-k + pow&ratio sampling
def sample_logits(logits, pos, temperature=1.0, top_k=None, top_p=None, min_p_pow=None, min_p_ratio=None):
    logits = logits[:, pos, :] / temperature
    probs = F.softmax(logits, dim=-1)

    if min_p_ratio is not None:
        limit = torch.pow(torch.max(probs), min_p_pow) * min_p_ratio
        logits[probs < limit] = -float('Inf')

    if top_k is not None:
        logits = top_k_logits(logits, top_k)

    probs = F.softmax(logits, dim=-1)

    if top_p is not None:
        probs[0] = top_p_probs(probs[0], top_p)

    ix = torch.multinomial(probs, num_samples=1)
    return ix[0][0].cpu()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_data_to_txt(predict, target, predict_name, target_name):
    # transfer the mat file to text file
    if isinstance(predict, np.ndarray):
        with open(predict_name, 'w') as file:
            for i in range(len(predict)):
                for j in range(len(predict[i])):
                    file.write(str(predict[i][j]) + ' ')
            file.write('\n')

        with open(target_name, 'w') as file:
            for i in range(len(target)):
                file.write(str(target[i][0]) + ' ' + str(target[i][0]) + '\n')
    # transfer the tensor to the text file
    else:
        with open(predict_name, 'w') as file:
            for i in range(len(predict)):
                p = predict[i].detach().cpu().numpy()
                file.write("######" + '\n')
                for j in range(len(p)):
                    file.write(str(p[j][0]) + ' ' + str(p[j][1]) + '\n')

        with open(target_name, 'w') as file:
            for i in range(len(target)):
                t = target[i].detach().cpu().numpy()
                file.write('######' + '\n')
                for j in range(len(t)):
                    file.write(str(t[j][0]) + ' ' + str(t[j][1]) + '\n')


def save_data2txt(data, data_name):
    if isinstance(data, np.ndarray):
        with open(data_name, 'w') as file:
            for i in range(len(data)):
                # file.write("######" + '\n')
                for j in range(len(data[i])):
                    file.write(str(data[i][j]) + ' ')
                file.write('\n')
    else:
        with open(data_name, 'w') as file:
            for i in range(len(data)):
                p = data[i].detach().cpu().numpy()
                # file.write("######" + '\n')
                for j in range(len(p)):
                    for k in range(len(p[j])):
                        file.write(str(p[j][k]) + ' ')
                    file.write('\n')


def load_mat(mat_file_path):
    with h5py.File(mat_file_path, 'r') as file:
        for key in file.keys():
            if not isinstance(file[key], h5py.Dataset):
                continue

            if key == "cursor_pos":
                y = []
                data = file[key][:]
                for location in data:
                    v = [location[i] for i in range(0, len(location))]
                    y.append(v)
                y = np.array(y)

            if key == "t":
                time = file[key][:]

            # load data of the mat file which are spikes(96*5 cell) and
            # the cell is a reference in h5py that requires iterative processing
            if key == "spikes":
                x = []
                cell_array = file[key]

                for i in range(cell_array.shape[1]):
                    temp = []
                    for j in range(cell_array.shape[0]):
                        cell_ref = cell_array[j, i]
                        cell_data = file[cell_ref]
                        data = cell_data[()]
                        data = data.ravel()
                        temp += list(data)
                    temp.sort()
                    temp = [element for element in temp if element != 0]
                    x.append(temp)

    y = np.array(y)
    time = np.array(time)

    return x, np.transpose(y), time

def spike_to_counts2(spike, y, time, gap_num):
    if (len(time) - 1) % gap_num == 0:
        dim = int((len(time) - 1) / gap_num)
    else:
        dim = int((len(time) - 1) / gap_num) + 1
    span = np.zeros([len(spike), dim])
    label = []
    time_slice = 0.0010 * gap_num

    for i in range(0, y.shape[0] - gap_num, gap_num):
        label.append(y[i + gap_num, :] - y[i, :])

    # process the shorter gap_num slice
    remainder = (y.shape[0] - 1) % gap_num
    if remainder != 0:
        label.append(y[-1, :] - y[-remainder, :])

    label = np.array(label)

    for i in range(len(spike)):
        for x in spike[i]:
            # calculate the time index
            interval_index = int((x - time[0]) // time_slice)
            if 0 <= interval_index < dim:
                span[i, interval_index] += 1

    span = np.array(span)

    return span, label


def resample_data(data, original_interval, new_interval):
    # calculate the original sampling points and new sampling points
    n_samples = data.shape[0]
    original_time = np.linspace(0, original_interval * (n_samples - 1), n_samples)
    new_time = np.linspace(0, int(original_interval * (n_samples - 1)), int(original_interval /
                                                                            new_interval * n_samples))

    # interpolate each column
    n_columns = data.shape[1]
    new_data = np.zeros([len(new_time), n_columns])

    for i in range(n_columns):
        # linear interpolate func
        interp_func = interp1d(original_time, data[:, i], kind='linear')
        new_data[:, i] = interp_func(new_time)

    return new_data


def spike_to_counts1(spike, y, t):

    start_time = t[0]  # 获取t的起始时间
    end_time = t[-1]  # 获取t的结束时间

    # 第一部分：处理spike数据
    num_intervals = int((t[-1] - t[0]) / 0.01) + 1  # 使用秒作为单位
    spike_matrix = np.zeros((num_intervals, len(spike)))  # 初始化spike矩阵

    for channel_idx, channel_spikes in enumerate(spike):
        for spike_time in channel_spikes:
            if spike_time >= start_time:
                interval_idx = int((spike_time - t[0]) / 0.01)
                if interval_idx < num_intervals:  # 确保索引在矩阵范围内
                    spike_matrix[interval_idx, channel_idx] += 1

    # 第二部分：处理y和t数据生成target矩阵
    target_matrix = np.zeros((num_intervals, 2))  # 初始化target矩阵

    for i in range(num_intervals):
        start_time = t[0] + i * 0.01
        end_time = start_time + 0.01
        start_idx = np.searchsorted(t, start_time, side='left')
        end_idx = np.searchsorted(t, end_time, side='right') - 1
        # 保证索引不越界
        start_idx = max(0, min(start_idx, len(y) - 1))
        end_idx = max(0, min(end_idx, len(y) - 1))
        velocity = y[end_idx] - y[start_idx]  # 速度计算基于差分
        target_matrix[i] = velocity

    return spike_matrix, target_matrix

def gaussian_nomalization(x, y):
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()

    return x, y


def min_max_nomalization(x, y):
    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())

    return x, y


from read_data import get_data
import numpy as np

def generate_batch(batch_size, timestep, dimensions, data_path="None", data_set="train"):
    normalize_data = get_data(data_path)
    total = len(normalize_data)
    train_test = int(total*0.7)
    train = normalize_data[: train_test]
    test = normalize_data[train_test: ]
    if data_set == "train":
        data = train
    else:
        data = test
    by = []
    bx = []
    while True:
        index = np.random.randint(len(data))
        x = [e for e in map(lambda i: data[i],
            [index - i for i in range(1, timestep+1)])]
        
        if len(x) < timestep:
            continue
        bx.append(x)
        feature = data[index]
        y = [0 if feature[i] < x[-1][i] else 1 for i in range(dimensions)]
        by.append(y)
        if len(bx) == batch_size:
            yield np.array(bx), np.array(by).astype(np.int32)
            bx = []
            by = []
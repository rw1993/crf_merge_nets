from read_data import get_data
import numpy as np

def random_generate_batch(batch_size, timestep, dimensions, data_path="None", data_set="train"):
    normalize_data = get_data(data_path)
    total = normalize_data.shape[0]
    train_valid= int(total*0.6)
    valid_test = int(total*0.8)

    train = normalize_data[: train_valid]
    valid = normalize_data[train_valid: valid_test]
    test = normalize_data[valid_test:]
    if data_set == "train":
        data = train
    elif data_set == "test":
        data = test
    else:
        data = valid

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
            yield 1, np.array(bx), np.array(by).astype(np.int32)
            bx = []
            by = []

def loop_generate_batch(batch_size, timestep, dimensions, data_path="None", data_set="train"):
    normalize_data = get_data(data_path)
    total = normalize_data.shape[0]
    print(total)
    train_valid= int(total*0.6)
    valid_test = int(total*0.8)

    train = normalize_data[: train_valid]
    valid = normalize_data[train_valid: valid_test]
    test = normalize_data[valid_test:]
    if data_set == "train":
        data = train
    elif data_set == "test":
        data = test
    else:
        data = valid

    by = []
    bx = []
    epoch = 1
    while True:
        for index in range(data.shape[0]):
            x = [e for e in map(lambda i: data[i],
                [index - i for i in range(1, timestep+1)])]
            if len(x) < timestep:
                continue
            bx.append(x)
            feature = data[index]
            y = [0 if feature[i] < x[-1][i] else 1 for i in range(dimensions)]
            by.append(y)
            if len(bx) == batch_size:
                yield epoch, np.array(bx), np.array(by).astype(np.int32)
                bx = []
                by = []
        epoch += 1
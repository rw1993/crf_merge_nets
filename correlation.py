from scipy.stats import pearsonr
import read_data

# get_data

def get_pearson(data_path="yfj.csv"):
    normalize_data = read_data.get_data(data_path=data_path)
    _, dimensions = normalize_data.shape
    pearson_dict = {}
    for i in range(dimensions):
        for j in range(i+1, dimensions):
            r = pearsonr(normalize_data[:,i], normalize_data[:,j])
            pearson_dict[(i, j)] = pearson_dict[(j, i)] = abs(r[0]) 
    return pearson_dict

if __name__ == "__main__":
    get_pearson()

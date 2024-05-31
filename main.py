from imports import *
from mvc_model import mvc_model
from purity import purity_score

mat_data = scipy.io.loadmat(r"C:\Users\user\PycharmProjects\mvc_Dr_jamshidi\NUSWIDEOBJ.mat")

print(type(mat_data))
print(mat_data.keys())
data_matrices = mat_data['X']
data_labels = mat_data['Y']

print(data_matrices.shape)
print(type(data_matrices))
print(type(data_labels))

data_matrices = list(data_matrices)


torch_data_labels = torch.from_numpy(data_labels)

data_matrices[0][2] = data_matrices[0][2].astype(float)
XX =[]
x0 = torch.from_numpy(data_matrices[0][0])
XX.append(x0)
x1 = torch.from_numpy(data_matrices[0][1])
XX.append(x1)
x2 = torch.from_numpy(data_matrices[0][2])
XX.append(x2)
x3 = torch.from_numpy(data_matrices[0][3])
XX.append(x3)
x4 = torch.from_numpy(data_matrices[0][4])
XX.append(x4)

#####################################################

counts , Labels_unique = np.unique(data_labels, return_counts=True)

print(Labels_unique)
nu = len(data_matrices) # number of views
n = len(data_labels)  # number of labels
k = len(Labels_unique) # number of unique label
scaler = MinMaxScaler()
XX = [torch.tensor(scaler.fit_transform(tensor), dtype=torch.float) for tensor in XX]
lap = []
X = []
for tensor in XX:
    transposed_tensor = tensor.T
    X.append(transposed_tensor)

alpha = torch.tensor([100 , 0.0000001 , 0.000001, 0.00001]).unsqueeze(dim=1)
beta = torch.tensor([ 100, 0.0001 , 0.00001 , 0.000001]).unsqueeze(dim=1)
gg = torch.tensor([1 , 0.01 , 0.001,0.0003]).unsqueeze(dim=1)
landa = torch.tensor([0.000001, 0.000001 , 0.0001 , 0.01]).unsqueeze(dim=1)
etha = torch.tensor([0.1,100000,10000000,100]).unsqueeze(dim=1)

F =  mvc_model(nu , n , k , X,alpha,beta,landa,etha,gg)
kmeans = torch.kthvalue(F, k)
labelF = kmeans.indices

labelF = labelF.unsqueeze(dim = 1)

print("labelF",labelF.shape)
print("torch_data_labels",torch_data_labels.shape)

accuracy_metric = torchmetrics.Accuracy(task='multiclass', num_classes=len(Labels_unique))
torch_data_labels = torch.from_numpy(data_labels)
acc = accuracy_metric(labelF, torch_data_labels)
print("acc", acc)

#  N M I
pred_value = labelF.numpy()
true_value = torch_data_labels.numpy()
nmi_score = normalized_mutual_info_score(true_value, pred_value)
print(f"Normalized Mutual Information Score: {nmi_score:.7f}")

#  A R I
ari_score = adjusted_rand_score(true_value, pred_value)
print(f"Adjusted Rand Index: {ari_score:.5f}")

f1 = f1_score(true_value, pred_value)
print(f1)

purity = purity_score(true_value, pred_value)
print(f"Purity score: {purity:.5f}")
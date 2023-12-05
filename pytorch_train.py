import pandas as pd
import numpy as np
import torch

from torch import nn, optim, Tensor
from category_encoders.target_encoder import TargetEncoder
from sklearn.model_selection import KFold


# 데이터프레임 불러오기
train_df = pd.read_csv('data/train_data_total1204.csv',encoding='cp949')
test_df = pd.read_csv('data/test_data_total1204.csv',encoding='cp949')
countrywide_df = pd.read_csv('data/countrywide_data_total1204.csv',encoding='cp949')

train_df = train_df.drop(columns="Unnamed: 0")
test_df = test_df.drop(columns="Unnamed: 0")
countrywide_df = countrywide_df.drop(columns="Unnamed: 0")

total_df = pd.concat([train_df, countrywide_df])
print(total_df.shape)

test_x = test_df.drop(columns=['ID']).copy()
train_x = total_df[test_x.columns].copy()
train_y = total_df['ECLO'].copy()

print(train_x.shape, train_y.shape)


# 타겟 인코딩 진행
categorical_features = list(train_x.dtypes[train_x.dtypes == "object"].index) 
print(categorical_features)

for i in categorical_features: 
    le = TargetEncoder(cols=[i])
    train_x[i] = le.fit_transform(train_x[i], train_y)
    test_x[i] = le.transform(test_x[i])


# train 결측치 처리하기
train_x['설치개수'] = train_x['설치개수'].fillna(train_x['설치개수'].mean())
train_x['School Zone'] = train_x['School Zone'].fillna(train_x['School Zone'].median())

train_x['급지구분_1'] = train_x['급지구분_1'].fillna(0)
train_x['급지구분_2'] = train_x['급지구분_2'].fillna(0)
train_x['급지구분_3'] = train_x['급지구분_3'].fillna(0)

# test 결측치 처리하기
test_x['설치개수'] = test_x['설치개수'].fillna(test_x['설치개수'].mean())
test_x['School Zone'] = test_x['School Zone'].fillna(test_x['School Zone'].median())

test_x['급지구분_1'] = test_x['급지구분_1'].fillna(0)
test_x['급지구분_2'] = test_x['급지구분_2'].fillna(0)
test_x['급지구분_3'] = test_x['급지구분_3'].fillna(0)


# VIF 높은 상위 3개 드롭
columns_to_drop = ['기상상태', '노면상태', '연']

# train_x와 test_x에서 해당 열들을 제거
train_x = train_x.drop(columns=columns_to_drop, axis=1)
test_x = test_x.drop(columns=columns_to_drop, axis=1)


# print(type(train_x.values), type(train_y.values), type(test_x.values))
# print(train_x.values.shape, test_x.values.shape)
# print(train_x.columns)


# 모델 정의
class MyModel(nn.Module):
    def __init__(self, input_size):
        super(MyModel, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, 24),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.Linear(24, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        return x


# RMSLE 정의
def rmsle(y_true, y_pred):    
    # y_true = tf.maximum(tf.cast(y_true, tf.float32), 0)
    # y_pred = tf.maximum(tf.cast(y_pred, tf.float32), 0)
    # squared_error = tf.square(tf.math.log1p(y_pred) - tf.math.log1p(y_true))
    # return tf.sqrt(tf.reduce_mean(squared_error))

    y_true = y_true.type(torch.flaot32)
    y_pred = y_pred.type(torch.flaot32)

    y_true = Tensor.clamp(y_true, min=0)
    y_pred = Tensor.clamp(y_pred, min=0)

    squared_error = (Tensor.log1p(y_pred) - Tensor.log1p(y_true))**2

    return torch.sqrt(Tensor.mean(squared_error))

# 커스텀 로스
def my_loss(y_true, y_pred):
    return rmsle(y_true, y_pred)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


# ! TODO
# pandas -> numpy -> tensor 로 모델에 집어 넣기


input_size = train_x.shape[1]

# Train

skf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_histories = []
test_preds = np.zeros(len(test_x))

# numpy -> tensor
train_x = torch.from_numpy(train_x.values)
train_y = torch.from_numpy(train_y.values)
test_x = torch.from_numpy(test_x.values)


for i, (train_index, valid_index) in enumerate(skf.split(train_x, train_y)):
    x_train_fold, x_valid_fold = train_x[train_index], train_x[valid_index]
    y_train_fold, y_valid_fold = train_y[train_index], train_y[valid_index]

    x_train_fold, x_valid_fold = x_train_fold.to(device), x_valid_fold.to(device)
    y_train_fold, y_valid_fold = y_train_fold.to(device), y_valid_fold.to(device)

    model = MyModel(input_size).to(device)
    loss_fn = my_loss()
    optim = optim.Adam(model.parameters(), lr=0.001)

    # train
    for epoch in range(111):
        optim.zero_grad()
        outputs = model(x_train_fold)
        loss = loss_fn(outputs, y_train_fold)
        loss.backward()
        optim.step()
    
    # valid
    # with torch.no_grad():
        
# if __name__ == '__main__':
#     pass

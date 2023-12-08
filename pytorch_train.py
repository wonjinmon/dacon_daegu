import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch


from torch import nn, optim, Tensor
from torch.utils.data import DataLoader, TensorDataset
from category_encoders.target_encoder import TargetEncoder
from sklearn.model_selection import KFold


# 모델 정의
class MyModel(nn.Module):
    def __init__(self, input_size):
        super(MyModel, self).__init__()
        self.input_size = input_size
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=24),
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

    # y_true = y_true.type(torch.flaot32)
    # y_pred = y_pred.type(torch.flaot32)

    y_true = torch.clamp(y_true, min=0)
    y_pred = torch.clamp(y_pred, min=0)

    squared_error = (torch.log1p(y_pred) - torch.log1p(y_true)) ** 2
    # print(type(squared_error))
    # print(squared_error)

    return torch.sqrt(torch.mean(squared_error))


# 커스텀 로스
def my_loss(y_true, y_pred):
    return rmsle(y_true, y_pred)


if __name__ == "__main__":
    # 데이터프레임 불러오기
    train_df = pd.read_csv("data/train_data_total1204.csv", encoding="cp949")
    test_df = pd.read_csv("data/test_data_total1204.csv", encoding="cp949")
    countrywide_df = pd.read_csv(
        "data/countrywide_data_total1204.csv", encoding="cp949"
    )[:10000]

    train_df = train_df.drop(columns="Unnamed: 0")
    test_df = test_df.drop(columns="Unnamed: 0")
    countrywide_df = countrywide_df.drop(columns="Unnamed: 0")

    total_df = pd.concat([train_df, countrywide_df])
    # print(total_df.shape)

    test_x = test_df.drop(columns=["ID"]).copy()
    train_x = total_df[test_x.columns].copy()
    train_y = total_df["ECLO"].copy()

    # print(train_x.shape, train_y.shape)

    # 타겟 인코딩 진행
    categorical_features = list(train_x.dtypes[train_x.dtypes == "object"].index)
    # print(categorical_features)

    for i in categorical_features:
        le = TargetEncoder(cols=[i])
        train_x[i] = le.fit_transform(train_x[i], train_y)
        test_x[i] = le.transform(test_x[i])

    # train 결측치 처리하기
    train_x["설치개수"] = train_x["설치개수"].fillna(train_x["설치개수"].mean())
    train_x["School Zone"] = train_x["School Zone"].fillna(
        train_x["School Zone"].median()
    )

    train_x["급지구분_1"] = train_x["급지구분_1"].fillna(0)
    train_x["급지구분_2"] = train_x["급지구분_2"].fillna(0)
    train_x["급지구분_3"] = train_x["급지구분_3"].fillna(0)

    # test 결측치 처리하기
    test_x["설치개수"] = test_x["설치개수"].fillna(test_x["설치개수"].mean())
    test_x["School Zone"] = test_x["School Zone"].fillna(test_x["School Zone"].median())

    test_x["급지구분_1"] = test_x["급지구분_1"].fillna(0)
    test_x["급지구분_2"] = test_x["급지구분_2"].fillna(0)
    test_x["급지구분_3"] = test_x["급지구분_3"].fillna(0)

    # VIF 높은 상위 3개 드롭
    columns_to_drop = ["기상상태", "노면상태", "연"]

    # train_x와 test_x에서 해당 열들을 제거
    train_x = train_x.drop(columns=columns_to_drop, axis=1)
    test_x = test_x.drop(columns=columns_to_drop, axis=1)

    # print(type(train_x.values), type(train_y.values), type(test_x.values))
    # print(train_x.values.shape, test_x.values.shape)
    # print(train_x.columns)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # numpy -> tensor
    train_x = torch.from_numpy(train_x.values).type(torch.FloatTensor).to(device)
    train_y = torch.from_numpy(train_y.values).type(torch.FloatTensor).to(device)
    test_x = torch.from_numpy(test_x.values).type(torch.FloatTensor).to(device)
    # print(type(train_x))

    # train_x = torch.tensor(train_x, dtype=torch.float32)
    # train_y = torch.tensor(train_y, dtype=torch.float32)
    # test_x = torch.tensor(test_x, dtype=torch.float32)
    # print(type(train_x))\

    num_epochs = 50

    skf = KFold(n_splits=5, shuffle=True, random_state=42)

    fold_histories = []

    input_size = train_x.shape[1]
    # print((train_x.shape))

    test_preds = torch.zeros(len(test_x), dtype=torch.float32).to(device)

    for i, (train_index, valid_index) in enumerate(skf.split(train_x, train_y)):
        print(f"============== fold {i} ==============")
        x_train_fold, x_valid_fold = train_x[train_index], train_x[valid_index]
        y_train_fold, y_valid_fold = train_y[train_index], train_y[valid_index]

        print(
            x_train_fold.shape,
            y_train_fold.shape,
            x_valid_fold.shape,
            y_valid_fold.shape,
        )

        train_dataset = TensorDataset(x_train_fold, y_train_fold)
        train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

        valid_dataset = TensorDataset(x_valid_fold, y_valid_fold)
        valid_dataloader = DataLoader(valid_dataset, batch_size=128, shuffle=False)
        # print(len(valid_dataloader))
        # print(type(input_size))
        # print(input_size)

        # print(type(x_train_fold))
        model = MyModel(input_size).to(device)
        # loss_fn = my_loss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # TRAIN
        for epoch in range(num_epochs):
            print("Epoch:", epoch + 1)
            # print("Train...")
            model.train()
            # optim.zero_grad()
            # outputs = model(x_train_fold)
            # train_loss = my_loss(outputs, y_train_fold)
            # train_loss.backward()
            # optim.step()
            train_loss_batch = []
            for x_batch, y_batch in train_dataloader:
                optimizer.zero_grad()
                outputs = model(x_batch)
                train_loss = my_loss(outputs, y_batch)
                train_loss_batch.append(train_loss.item())
                train_loss.backward()
                optimizer.step()

            # VALID
            # print("Valid...")
            val_loss_batch = []
            model.eval()
            with torch.no_grad():
                for x_batch, y_batch in valid_dataloader:
                    outputs = model(x_batch)
                    val_loss = my_loss(outputs, y_batch)
                    val_loss_batch.append(val_loss.item())

            fold_histories.append(
                {"train_loss": train_loss.item(), "val_loss": val_loss.item()}
            )
            print("train_loss", train_loss.item(), "val_loss", val_loss.item())

        # TEST
        print("Test...")
        model.eval()
        with torch.no_grad():
            for _ in range(len(valid_dataloader)):
                test_preds += model(test_x).reshape(-1) / len(valid_dataloader)
                # test_preds /= skf.n_splits
            # test_preds += (
            #     torch.cat([MyModel(test_x) for _ in range(len(valid_dataloader))]).reshape(-1) / skf.n_splits
            # )
    print(fold_histories)

    # 그래프
    train_losses = [fold["train_loss"] for fold in fold_histories]
    val_losses = [fold["val_loss"] for fold in fold_histories]

    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, label="Train Loss")

    plt.plot(epochs, val_losses, label="Valid Loss")
    plt.legend()
    plt.show()

    # 제출
    sample_submission = pd.read_csv("data/sample_submission.csv")

    sample_submission["ECLO"] = test_preds.cpu() / 5

    sample_submission.to_csv("result/1205_pytorch_test.csv", index=False)
    print(sample_submission["ECLO"].value_counts())

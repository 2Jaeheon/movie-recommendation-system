import pandas as pd
import torch
from sklearn.model_selection import train_test_split


# 사용자 및 영화 ID를 연속적으로 매핑하는 함수
# 연속된 숫자 인덱스로 매핑함.
def create_mappings(matrix):
    user_mapping = {user: idx for idx, user in enumerate(matrix.index)}
    movie_mapping = {movie: idx for idx, movie in enumerate(matrix.columns)}
    return user_mapping, movie_mapping


# NaN이 아닌 값만 추출하는 함수
def get_non_nan_values(matrix, user_mapping, movie_mapping):
    # stack()을 통해 행렬을 1차원 Series로 변환
    # reset_index()를 통해 인덱스를 열로 변환
    values = matrix.stack().reset_index()
    # 각 열의 이름을 지정
    values.columns = ['user', 'movie', 'rating']

    # 사용자 및 영화 ID를 연속 인덱스로 변환
    return [(user_mapping[user], movie_mapping[movie], rating)
            for user, movie, rating in
            zip(values['user'], values['movie'], values['rating'])]


# RMSE 계산 함수
def calculate_rmse(values, Q, P):
    total_loss = 0
    for user_idx, movie_idx, rating in values:
        # 예측값 계산
        prediction = torch.dot(Q[user_idx], P[movie_idx]).item()
        total_loss += (rating - prediction) ** 2
    # RMSE 계산
    rmse = torch.sqrt(torch.tensor(total_loss / len(values)))
    return rmse.item()


def predict_ratings(test_data, user_mapping, movie_mapping, Q, P):
    predictions = []
    for _, row in test_data.iterrows():
        user_id = row['userId']
        movie_id = row['movieId']
        if user_id in user_mapping and movie_id in movie_mapping:
            user_idx = user_mapping[user_id]
            movie_idx = movie_mapping[movie_id]
            prediction = torch.dot(Q[user_idx], P[movie_idx]).item()
            # 0.5 단위로 반올림
            rounded_prediction = round(prediction * 2) / 2
            predictions.append((row['rId'], rounded_prediction))
        else:
            # 매핑되지 않은 사용자나 영화의 경우 평균 평점으로 대체
            predictions.append((row['rId'], 3.0))
    return predictions


# 데이터 로드
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# train_data를 6:4로 분할
training_data, evaluation_data = train_test_split(train_data, test_size=0.4,
                                                  random_state=42)

# train_data를 사용자-영화 행렬로 변환
user_movie_matrix = train_data.pivot_table(index='userId', columns='movieId',
                                           values='rating')

evaluation_matrix = evaluation_data.pivot_table(index='userId',
                                                columns='movieId',
                                                values='rating')

# 사용자와 영화 ID를 연속 인덱스로 매핑
user_mapping, movie_mapping = create_mappings(user_movie_matrix)
evaluation_user_mapping, evaluation_movie_mapping = create_mappings(
    evaluation_matrix)

# NaN이 아닌 값만 추출
value = get_non_nan_values(user_movie_matrix, user_mapping, movie_mapping)
evaluation_values = get_non_nan_values(evaluation_matrix,
                                       evaluation_user_mapping,
                                       evaluation_movie_mapping)

# 사용자-영화 행렬의 크기
rows = len(user_movie_matrix.index)
cols = len(user_movie_matrix.columns)

# 학습 설정
rank = 20  # Latent Factor의 차원
epoch = 100  # 학습 횟수
learning_rate = 0.005  # 학습률
epsilon = 0.001  # RMSE 감소 폭 기준 설정
prev_evaluation_rmse = float('inf')  # 초기 이전 RMSE를 매우 큰 값으로 설정

# P, Q 행렬 초기화
Q = torch.empty(rows, rank, requires_grad=True)
P = torch.empty(cols, rank, requires_grad=True)
torch.nn.init.xavier_uniform_(Q)
torch.nn.init.xavier_uniform_(P)

optimizer = torch.optim.SGD([Q, P], lr=learning_rate)
# optimizer = torch.optim.Adam([Q, P], lr=learning_rate)

for epoch in range(epoch):
    for i, x, rating in value:
        predict = torch.sum(Q[i] * P[x])
        rating = torch.tensor(rating, dtype=torch.float32)

        # loss 계산
        loss = torch.nn.functional.mse_loss(predict, rating)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # RMSE 계산
    rmse = calculate_rmse(value, Q, P)
    evaluation_rmse = calculate_rmse(evaluation_values, Q, P)

    # 에포크 진행 상황 출력
    print(
        f"Epoch: {epoch}, Loss: {loss.item():.4f}, train RMSE: {rmse:.4f}, test RMSE: {evaluation_rmse:.4f}"
    )

    # Early Stopping 기준
    if abs(prev_evaluation_rmse - evaluation_rmse) < epsilon:
        print(f"Early stopping at epoch {epoch}.")
        break

    # 이전 평가 RMSE 갱신
    prev_evaluation_rmse = evaluation_rmse

# test_data에서 예측 수행
predictions = predict_ratings(test_data, user_mapping, movie_mapping, Q, P)

# 예측 결과를 DataFrame으로 변환
# 변환하는 이유: DataFrame을 통해 쉽게 CSV 파일로 저장할 수 있음
predictions_df = pd.DataFrame(predictions, columns=['rId', 'rating'])

# 결과를 CSV 파일로 저장
predictions_df.to_csv('submission.csv', index=False)

# 결과 확인
print(predictions_df.head())

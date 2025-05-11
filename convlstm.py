# 이 코드는 기온, 미세먼지, 오존의 64*64 이미지를 각각의 channel로 사용하여 만든 3*64*64 를 입력으로 할 때의 코드이다. 

# 먼저 ConvLSTM 구현 라이브러리 설치
# pip install torch_lstm

# 필요한 라이브러리 임포트
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
# import torchvision.transforms as transforms # 데이터 정규화 시 사용
# from your_convlstm_library import ConvLSTM # 실제 ConvLSTM 구현체 임포트

# torch_lstm 라이브러리의 ConvLSTM 모듈 사용 가정
try:
    from torch_lstm import ConvLSTM
except ImportError:
    print("torch_lstm 라이브러리가 설치되지 않았습니다. 'pip install torch_lstm' 명령으로 설치해주세요.")
    print("또는 ConvLSTM 구현체를 직접 정의하거나 다른 라이브러리를 사용해주세요.")
    # 에러 발생 시 학습 코드는 실행되지 않습니다.

# --- 1. 데이터셋 준비 ---

class TimeSeriesMultiChannelImageDataset(Dataset):
    def __init__(self, image_data, sequence_length):
        """
        Args:
            image_data (numpy.ndarray or torch.Tensor):
                Shape: (Total_Timesteps, Channels, Height, Width)
                (N, 3, 64, 64) 형태의 전체 시계열 이미지 데이터 (N은 총 시점 수)
            sequence_length (int):
                모델 입력으로 사용할 시퀀스의 길이 (L)
        """
        self.image_data = self._to_tensor(image_data)

        # 데이터 형태 검사 (Channel=3, ImageSize=64x64 가정)
        if self.image_data.ndim != 4 or self.image_data.shape[1] != 3 or self.image_data.shape[2] != 64 or self.image_data.shape[3] != 64:
             raise ValueError(f"Input image_data must be (Total_Timesteps, 3, 64, 64), but got {self.image_data.shape}")

        self.total_timesteps = self.image_data.shape[0]
        self.sequence_length = sequence_length

        if self.sequence_length >= self.total_timesteps:
            raise ValueError(f"sequence_length ({sequence_length}) must be less than total_timesteps ({self.total_timesteps})")

        # 데이터 정규화 (필요시 여기에 추가)
        # 예: self.image_data = self.normalize_data(self.image_data)

    def _to_tensor(self, data):
        """ 데이터를 torch.Tensor로 변환하고 dtype을 float32로 설정 """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        elif not isinstance(data, torch.Tensor):
            data = torch.tensor(data)
        return data.float()

    # def normalize_data(self, data):
    #     """ 채널별 정규화 (예시) """
    #     # 이 부분에서 훈련 데이터 전체에 대한 채널별 평균과 표준편차를 계산해야 합니다.
    #     # 실제 구현에서는 __init__ 밖에서 평균/표준편차 계산 함수를 호출하고,
    #     # 그 결과 통계치를 여기에 전달받아 사용합니다.
    #     mean = [0.5, 0.5, 0.5] # 예시 평균 (실제 데이터로 계산 필요)
    #     std = [0.5, 0.5, 0.5]   # 예시 표준편차 (실제 데이터로 계산 필요)
    #     normalize = transforms.Normalize(mean=mean, std=std)

    #     # 각 시점의 이미지에 정규화 적용
    #     # (N, C, H, W) 형태이므로, 각 (C, H, W) 이미지에 대해 normalize 적용
    #     normalized_data = torch.stack([normalize(data[i]) for i in range(data.shape[0])])
    #     print(f"Data normalized. Shape: {normalized_data.shape}")
    #     return normalized_data


    def __len__(self):
        """ 데이터셋의 총 샘플 수 반환 """
        return self.total_timesteps - self.sequence_length

    def __getitem__(self, idx):
        """
        주어진 인덱스(idx)에 해당하는 학습 샘플 반환

        Args:
            idx (int): 샘플의 시작 시점 인덱스 (0부터 __len__()-1 까지)

        Returns:
            tuple: (input_sequence, target_image)
                   input_sequence: (sequence_length, Channels, Height, Width)
                   target_image: (Channels, Height, Width) - 전체 3채널 이미지 (타겟 채널은 나중에 선택)
        """
        if idx < 0 or idx >= self.__len__():
            raise IndexError(f"Index {idx} is out of bounds for dataset of size {self.__len__()}")

        # 샘플의 입력 시퀀스 시작 시점
        start_timestep = idx
        # 샘플의 입력 시퀀스 끝 시점
        end_timestep = start_timestep + self.sequence_length
        # 샘플의 정답 이미지 시점 (입력 시퀀스 다음 시점)
        target_timestep = end_timestep

        # 해당 시점 범위의 이미지 시퀀스 추출
        input_sequence = self.image_data[start_timestep:end_timestep] # shape: (L, 3, 64, 64)

        # 정답 이미지 추출 (아직 3채널 그대로)
        target_image = self.image_data[target_timestep] # shape: (3, 64, 64)

        return input_sequence, target_image

# --- 2. 신경망 모델 구성 ---

class NextTemperatureModel(nn.Module):
    def __init__(self, sequence_length, total_input_channels, convlstm_hidden_channels,
                 output_image_channels, image_height, image_width, conv_kernel_size=(3, 3)):
        super().__init__()

        self.sequence_length = sequence_length
        self.total_input_channels = total_input_channels # 3
        self.convlstm_hidden_channels = convlstm_hidden_channels # 24
        self.output_image_channels = output_image_channels # 1
        self.image_height = image_height # 64
        self.image_width = image_width # 64
        self.conv_kernel_size = conv_kernel_size

        # ConvLSTM 레이어 정의
        # torch_lstm.ConvLSTM은 (Batch, Seq, C, H, W) 형태를 바로 입력받음
        self.conv_lstm = ConvLSTM(
            input_channels=self.total_input_channels, # 3
            hidden_channels=self.convlstm_hidden_channels, # 24
            kernel_size=self.conv_kernel_size, # (3, 3)
            num_layers=1, # 단일 ConvLSTM 레이어
            batch_first=True # DataLoader 출력이 batch 차원 먼저 오도록 설정
        )
        # ConvLSTM 출력이 (Batch, Seq, hidden_channels, H_out, W_out) 형태가 됩니다.
        # H_out, W_out은 커널, 패딩, 스트라이드에 따라 달라지지만,
        # torch_lstm 구현체는 기본적으로 padding='same' 효과로 H_out, W_out을 입력과 같게 유지합니다.
        # 따라서 여기서는 H_out = H, W_out = W = 64 라고 가정합니다.

        # ConvLSTM의 최종 은닉 상태 (h_t)를 받아 최종 출력 이미지(1채널)를 생성하는 레이어
        # ConvLSTM 마지막 출력 형태: (Batch, convlstm_hidden_channels=24, H, W=64, 64)
        # 최종 출력 이미지 형태: (Batch, output_image_channels=1, H, W=64, 64)
        self.output_conv = nn.Conv2d(
            in_channels=self.convlstm_hidden_channels, # 24 채널 입력
            out_channels=self.output_image_channels, # 1 채널 출력
            kernel_size=(1, 1) # 1x1 Conv는 채널 수만 변경하고 공간 크기 유지
            # 만약 ConvLSTM이 공간 크기를 변경했다면, UpSampling 또는 ConvTranspose2d 등으로 복원 필요
        )

        # 가중치 초기화 (선택 사항, 필요시 주석 해제 및 init_weights 함수 구현)
        # self.apply(self.init_weights)

    # def init_weights(self, m):
    #     # 여기에 가중치 초기화 로직 구현 (이전 답변 참고)
    #     pass


    def forward(self, input_sequence):
        # input_sequence shape: (batch_size, sequence_length, Channels, Height, Width)
        # 예: (B, L, 3, 64, 64)

        batch_size, seq_len, channels, h, w = input_sequence.shape

        # ConvLSTM 레이어 통과
        # convlstm_output_seq: (Batch, Seq, hidden_channels, H_out, W_out)
        # (hn, cn): 마지막 시점의 hidden/cell state (Layer 수, Batch, hidden_channels, H_out, W_out)
        convlstm_output_seq, (hn, cn) = self.conv_lstm(input_sequence)

        # ConvLSTM 시퀀스 출력 중 마지막 시점의 은닉 상태 (h_t) 추출
        # shape: (Batch, hidden_channels, H_out, W_out)
        final_convlstm_hidden_state = convlstm_output_seq[:, -1, :, :, :] # h_t에 해당

        # 최종 출력 레이어를 통과시켜 다음 기온 이미지 예측
        # shape: (Batch, output_image_channels, Final_H, Final_W)
        predicted_temperature_image = self.output_conv(final_convlstm_hidden_state) # (B, 1, 64, 64)

        return predicted_temperature_image

# --- 3. 학습 설정 및 루프 ---

# 하이퍼파라미터 설정
sequence_length = 10 # 입력 시퀀스 길이 L
batch_size = 32
learning_rate = 0.001
num_epochs = 20

# 모델 파라미터 설정
total_input_channels = 3
convlstm_hidden_channels = 24 # 요청하신 은닉 채널 수
output_image_channels = 1 # 예측할 기온 이미지 채널 수
image_height = 64
image_width = 64
conv_kernel_size = (3, 3)

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 예제 데이터 생성 (실제 데이터로 대체 필요)
# (총 시점 수, Channels, Height, Width) 형태의 데이터
total_timesteps_example = 100 # 예시로 100 시점 데이터 생성
example_image_data_np = np.random.rand(total_timesteps_example, total_input_channels, image_height, image_width).astype(np.float32)

# 데이터셋 및 데이터로더 생성
# 실제 데이터로 바꿀 때 example_image_data_np 부분을 실제 데이터 로딩 코드로 대체
dataset = TimeSeriesMultiChannelImageDataset(example_image_data_np, sequence_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 모델 인스턴스 생성
model = NextTemperatureModel(
    sequence_length=sequence_length,
    total_input_channels=total_input_channels,
    convlstm_hidden_channels=convlstm_hidden_channels,
    output_image_channels=output_image_channels,
    image_height=image_height,
    image_width=image_width,
    conv_kernel_size=conv_kernel_size
)
model.to(device)

# 가중치 초기화 (필요시 주석 해제)
# model.apply(init_weights)

print("\nModel Architecture:")
print(model)

# 손실 함수 (이미지 픽셀 값 예측이므로 MSELoss 사용)
criterion = nn.MSELoss()

# 옵티마이저
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 학습 루프
print("\nStarting Training...")
for epoch in range(1, num_epochs + 1):
    model.train() # 훈련 모드
    running_loss = 0.0

    for batch_idx, (input_seq_batch, target_img_batch) in enumerate(dataloader):
        # 데이터를 디바이스로 이동
        input_seq_batch = input_seq_batch.to(device)
        target_img_batch = target_img_batch.to(device)

        # 타겟 이미지에서 기온 채널 (첫 번째 채널)만 선택
        # shape: (Batch, 1, 64, 64)
        target_temp_batch = target_img_batch[:, 0:1, :, :] # 0:1 슬라이싱으로 차원 유지

        # 그라디언트 초기화
        optimizer.zero_grad()

        # 순전파: 모델에 입력 시퀀스 배치 전달
        # 모델은 다음 시점의 1채널 기온 이미지 배치 (B, 1, 64, 64)를 예측
        predicted_temp_batch = model(input_seq_batch)

        # 손실 계산: 예측된 기온 이미지와 실제 타겟 기온 이미지 비교
        loss = criterion(predicted_temp_batch, target_temp_batch)

        # 역전파
        loss.backward()

        # 가중치 업데이트
        optimizer.step()

        running_loss += loss.item()

        # 진행 상황 출력 (선택 사항)
        if batch_idx % 10 == 0:
             print(f'Epoch: {epoch}, Batch: {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}')

    avg_loss = running_loss / len(dataloader)
    print(f'--- Epoch {epoch} Finished. Average Loss: {avg_loss:.4f} ---')

print("\nTraining finished!")

# --- 4. 추론 예제 (선택 사항) ---
# 실제 추론 시에는 훈련된 모델을 로드해야 합니다.
# 예: model.load_state_dict(torch.load('trained_model.pth'))
# model.eval()
# with torch.no_grad():
#    # 추론할 입력 시퀀스 (L, 3, 64, 64) 형태
#    inference_input_sequence = ... # 준비된 데이터 로드
#    inference_input_sequence = inference_input_sequence.unsqueeze(0) # 배치 차원 추가 (1, L, 3, 64, 64)
#    inference_input_sequence = inference_input_sequence.to(device)

#    predicted_next_temp = model(inference_input_sequence) # (1, 1, 64, 64)
#    predicted_next_temp = predicted_next_temp.squeeze(0).cpu().numpy() # 배치, 채널 차원 제거 -> (64, 64) numpy 배열
#    print(f"Predicted next temperature image shape: {predicted_next_temp.shape}")

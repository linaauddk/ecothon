# 이 코드는 기온, 미세먼지, 오존의 64*64 이미지를 각각의 channel로 사용하여 만든 3*64*64 를 입력으로 할 때의 코드이다. 
# 24개 시점의 input을 받아서 25번째 시점에서의 기온 image를 출력한다. 

# 먼저 ConvLSTM 구현 라이브러리 설치 (설치되어 있지 않다면)
# pip install torch_lstm

# --- 필요한 라이브러리 임포트 ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os # 데이터 저장/로드 시 필요
# import torchvision.transforms as transforms # 데이터 정규화 시 사용

# ConvLSTM 구현 라이브러리 임포트
try:
    from torch_lstm import ConvLSTM
except ImportError:
    print("Error: torch_lstm library not found.")
    print("Please install it using: pip install torch_lstm")
    # 프로그램 종료 또는 예외 처리 로직 추가
    exit() # 예시에서는 프로그램 종료

# --- 하이퍼파라미터 설정 ---
SEQUENCE_LENGTH = 24     # 입력 시퀀스 길이 (24개 시점)
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 20

# 모델 파라미터 설정
TOTAL_INPUT_CHANNELS = 3       # 입력 이미지 채널 수 (기온, 미세먼지, 오존)
OUTPUT_IMAGE_CHANNELS = 1      # 출력 이미지 채널 수 (기온)
IMAGE_HEIGHT = 64              # 이미지 높이
IMAGE_WIDTH = 64               # 이미지 너비
CONVLSTM_HIDDEN_CHANNELS = 32  # ConvLSTM 레이어의 은닉 채널 수 (예시 값)
CONV_KERNEL_SIZE = (3, 3)      # ConvLSTM 및 기타 Conv 레이어 커널 사이즈 (예시)
NUM_CONVLSTM_LAYERS = 1        # ConvLSTM 레이어 수

# --- 디바이스 설정 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- 1. 데이터셋 준비 ---

class TimeSeriesMultiChannelImageDataset(Dataset):
    def __init__(self, image_data, sequence_length):
        """
        Args:
            image_data (numpy.ndarray or torch.Tensor):
                Shape: (Total_Timesteps, Channels, Height, Width)
                (N, 3, 64, 64) 형태의 전체 시계열 이미지 데이터 (N은 총 시점 수)
            sequence_length (int):
                모델 입력으로 사용할 시퀀스의 길이 (L). 24
        """
        self.image_data = self._to_tensor(image_data)

        # 데이터 형태 검사 (Channel=3, ImageSize=64x64 가정)
        if self.image_data.ndim != 4 or self.image_data.shape[1] != TOTAL_INPUT_CHANNELS or self.image_data.shape[2] != IMAGE_HEIGHT or self.image_data.shape[3] != IMAGE_WIDTH:
             raise ValueError(f"Input image_data must be (Total_Timesteps, {TOTAL_INPUT_CHANNELS}, {IMAGE_HEIGHT}, {IMAGE_WIDTH}), but got {self.image_data.shape}")

        self.total_timesteps = self.image_data.shape[0]
        self.sequence_length = sequence_length

        if self.sequence_length >= self.total_timesteps:
            raise ValueError(f"sequence_length ({sequence_length}) must be less than total_timesteps ({self.total_timesteps})")

        # --- 데이터 정규화 Placeholder ---
        # 실제 데이터 사용 시, 여기서 훈련 데이터의 채널별 평균과 표준편차를 계산하고
        # self.image_data에 정규화를 적용해야 합니다.
        # 예: self.image_data = self.normalize_data(self.image_data)
        # normalize_data 메소드는 별도로 구현하며, 테스트/검증 데이터에는
        # 훈련 데이터의 통계치를 그대로 사용해야 합니다.
        # print("Data Normalization Placeholder: Apply normalization here.")


    def _to_tensor(self, data):
        """ 데이터를 torch.Tensor로 변환하고 dtype을 float32로 설정 """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        elif not isinstance(data, torch.Tensor):
            data = torch.tensor(data)
        return data.float()

    # def normalize_data(self, data):
    #     """ 채널별 정규화 구현 (예시) """
    #     # 이 함수는 실제 데이터의 통계치를 사용하여 구현해야 합니다.
    #     # transforms.Normalize를 사용하려면 (C, H, W) 형태로 변환 후 적용해야 할 수 있습니다.
    #     # 또는 채널별로 직접 평균/표준편차 계산 후 연산 수행
    #     print("Executing Placeholder Normalization")
    #     mean = data.mean(dim=(0, 2, 3), keepdim=True) # (1, C, 1, 1) 형태
    #     std = data.std(dim=(0, 2, 3), keepdim=True)   # (1, C, 1, 1) 형태
    #     normalized_data = (data - mean) / (std + 1e-5) # 분모 0 방지
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
        # 샘플의 입력 시퀀스 끝 시점 (sequence_length 개를 포함하므로)
        end_timestep = start_timestep + self.sequence_length
        # 샘플의 정답 이미지 시점 (입력 시퀀스 다음 시점)
        target_timestep = end_timestep

        # 해당 시점 범위의 이미지 시퀀스 추출
        # shape: (sequence_length, 3, 64, 64)
        input_sequence = self.image_data[start_timestep:end_timestep]

        # 정답 이미지 추출 (아직 3채널 그대로)
        # shape: (3, 64, 64)
        target_image = self.image_data[target_timestep]

        return input_sequence, target_image

# --- 2. 신경망 모델 구성 ---

# PyTorch 기본 Conv 레이어 등을 위한 베이스 클래스 임포트
from torch.nn.modules.conv import _ConvNd

class NextTemperatureModel(nn.Module):
    def __init__(self, sequence_length, total_input_channels, convlstm_hidden_channels,
                 output_image_channels, image_height, image_width, conv_kernel_size=(3, 3), num_convlstm_layers=1):
        super().__init__()

        self.sequence_length = sequence_length
        self.total_input_channels = total_input_channels # 3
        self.convlstm_hidden_channels = convlstm_hidden_channels # 24
        self.output_image_channels = output_image_channels # 1
        self.image_height = image_height # 64
        self.image_width = image_width # 64
        self.conv_kernel_size = conv_kernel_size
        self.num_convlstm_layers = num_convlstm_layers

        # ConvLSTM 레이어 정의
        # torch_lstm.ConvLSTM은 (Batch, Seq, C, H, W) 형태를 바로 입력받음
        self.conv_lstm = ConvLSTM(
            input_channels=self.total_input_channels, # 3
            hidden_channels=self.convlstm_hidden_channels, # 24
            kernel_size=self.conv_kernel_size, # (3, 3)
            num_layers=self.num_convlstm_layers, # 레이어 수
            batch_first=True # DataLoader 출력이 batch 차원 먼저 오도록 설정
        )
        # ConvLSTM 출력이 (Batch, Seq, hidden_channels, H_out, W_out) 또는
        # (Batch, hidden_channels, H_out, W_out) (return_sequence=False 시) 형태가 됩니다.
        # torch_lstm 구현체는 return_sequence=True가 기본이며, padding='same' 효과로 H_out, W_out을 입력과 같게 유지합니다.
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

        
        '''
        좀 더 정교한 출력을 내야 할 때에는 마지막에 conv layer 1개 대신에 더욱 복잡한 신경망을 연결. 
        self.intermediate_processing = nn.Sequential(
        첫 번째 블록: Conv -> BatchNorm -> ReLU
        nn.Conv2d(in_channels=self.convlstm_hidden_channels, out_channels=self.convlstm_hidden_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(num_features=self.convlstm_hidden_channels),
        nn.ReLU(inplace=True),

        두 번째 블록 (선택 사항, 더 복잡한 모델 원하면 추가)
        nn.Conv2d(in_channels=self.convlstm_hidden_channels, out_channels=self.convlstm_hidden_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(num_features=self.convlstm_hidden_channels),
        nn.ReLU(inplace=True),

        ... 필요시 블록 더 추가 ...
        '''

            # 마지막 컨볼루션: 특징 채널 수를 최종 출력 채널 수(1)로 매핑
            # 이 최종 Conv는 intermediate_processing Sequential에 포함시키거나,
            # 별도로 self.output_conv로 분리할 수 있습니다.
            # 여기서는 Sequential 내에 포함시켜 중간 처리의 마지막 단계로 만듭니다.
            nn.Conv2d(in_channels=self.convlstm_hidden_channels, out_channels=self.output_image_channels, kernel_size=1) # 1x1 Conv로 채널 수만 변경
        )

        # --- 가중치 초기화 Placeholder ---
        # 모델 인스턴스 생성 후 model.apply(init_weights) 호출하여 적용
        # print("Weight Initialization Placeholder: Apply init_weights function after model creation.")

    # def init_weights(self, m):
    #     # 여기에 가중치 초기화 로직 구현 (이전 답변 참고)
    #     if isinstance(m, (nn.Conv2d, _ConvNd)):
    #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # ConvLSTM 내부 Conv 포함 시
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.Linear):
    #         nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     # LSTM/ConvLSTM 내부 가중치 접근은 라이브러리 구현에 따라 다름


    def forward(self, input_sequence):
        # input_sequence shape: (batch_size, sequence_length, Channels, Height, Width)
        # 예: (B, 24, 3, 64, 64)

        # ConvLSTM 레이어 통과
        # torch_lstm.ConvLSTM은 기본적으로 return_sequence=True
        # convlstm_output_seq: (Batch, Seq, hidden_channels, H_out, W_out)
        # (hn, cn): 마지막 시점의 hidden/cell state (Layer 수, Batch, hidden_channels, H_out, W_out)
        convlstm_output_seq, (hn, cn) = self.conv_lstm(input_sequence)

        # ConvLSTM 시퀀스 출력 중 마지막 시점의 은닉 상태 (h_t) 추출
        # shape: (Batch, hidden_channels, H_out, W_W)
        # hn[-1]은 마지막 레이어의 최종 hidden state, convlstm_output_seq[:,-1]는 모든 레이어의 마지막 시점 출력
        # torch_lstm은 hn[-1]을 사용하는 것이 마지막 레이어의 최종 h_t에 해당합니다.
        # final_convlstm_hidden_state = convlstm_output_seq[:, -1, :, :, :] # 시퀀스 출력에서 마지막 시점
        final_convlstm_hidden_state = hn[-1] # <-- 마지막 레이어의 최종 hidden state (더 정확)

        # 최종 출력 레이어를 통과시켜 다음 기온 이미지 예측
        # shape: (Batch, output_image_channels, Final_H, Final_W)
        predicted_temperature_image = self.output_conv(final_convlstm_hidden_state) # (B, 1, 64, 64)

        # H_out, W_out이 입력 이미지 크기(64x64)와 다르다면 Resize 등의 추가 작업 필요

        return predicted_temperature_image # shape: (Batch, 1, 64, 64)


# --- 3. 학습 설정 및 루프 ---

# 예제 데이터 생성 (실제 데이터로 대체 필요)
# (총 시점 수, Channels, Height, Width) 형태의 데이터
# sequence_length (24) + 1 (정답) 만큼의 최소 시점이 필요
TOTAL_TIMESTEPS_EXAMPLE = 100 # 예시로 100 시점 데이터 생성 (총 100 - 24 = 76개 샘플 생성 가능)
# 더미 데이터: 0 ~ 1 사이 랜덤 값
example_image_data_np = np.random.rand(TOTAL_TIMESTEPS_EXAMPLE, TOTAL_INPUT_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH).astype(np.float32)

# --- 데이터셋 및 데이터로더 생성 ---
# 실제 데이터로 바꿀 때 example_image_data_np 부분을 실제 데이터 로딩 코드로 대체
dataset = TimeSeriesMultiChannelImageDataset(example_image_data_np, sequence_length=SEQUENCE_LENGTH)

# 훈련 데이터와 검증 데이터 분리 (선택 사항이지만 좋은 관행)
# train_size = int(0.8 * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
# train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 분리하지 않고 전체 데이터셋으로 학습 (간단한 예제 목적)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


print(f"Total number of samples (sequences of length {SEQUENCE_LENGTH}): {len(dataset)}")
print(f"Number of batches per epoch: {len(dataloader)}")

# --- 모델 인스턴스 생성 ---
model = NextTemperatureModel(
    sequence_length=SEQUENCE_LENGTH,
    total_input_channels=TOTAL_INPUT_CHANNELS,
    convlstm_hidden_channels=CONVLSTM_HIDDEN_CHANNELS,
    output_image_channels=OUTPUT_IMAGE_CHANNELS,
    image_height=IMAGE_HEIGHT,
    image_width=IMAGE_WIDTH,
    conv_kernel_size=CONV_KERNEL_SIZE,
    num_convlstm_layers=NUM_CONVLSTM_LAYERS # 단일 레이어
)
model.to(DEVICE)

# --- 가중치 초기화 적용 (필요시 주석 해제) ---
# print("\nApplying Weight Initialization...")
# model.apply(init_weights)


print("\nModel Architecture:")
print(model)

# --- 손실 함수 (이미지 픽셀 값 예측이므로 MSELoss 사용) ---
criterion = nn.MSELoss()

# --- 옵티마이저 ---
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 학습 루프 ---
print("\nStarting Training...")
for epoch in range(1, NUM_EPOCHS + 1):
    model.train() # 훈련 모드
    running_loss = 0.0

    for batch_idx, (input_seq_batch, target_img_batch) in enumerate(dataloader):
        # input_seq_batch shape: (Batch, 24, 3, 64, 64)
        # target_img_batch shape: (Batch, 3, 64, 64) <- 정답 이미지 전체 (기온 포함)

        # 데이터를 디바이스로 이동
        input_seq_batch = input_seq_batch.to(DEVICE)
        target_img_batch = target_img_batch.to(DEVICE)

        # 타겟 이미지에서 기온 채널 (첫 번째 채널, 인덱스 0)만 선택
        # shape: (Batch, 1, 64, 64)
        target_temp_batch = target_img_batch[:, 0:1, :, :] # 0:1 슬라이싱으로 채널 차원 크기 1 유지

        # 그라디언트 초기화
        optimizer.zero_grad()

        # 순전파: 모델에 입력 시퀀스 배치 전달 (24개 시점)
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

# --- 학습 완료 ---
print("\nTraining finished!")

# --- 모델 저장 (선택 사항) ---
# try:
#     # 모델의 상태 딕셔너리 저장
#     torch.save(model.state_dict(), 'convlstm_temp_prediction_model.pth')
#     print("Model state_dict saved successfully.")
# except Exception as e:
#     print(f"Error saving model: {e}")


# --- 추론 예제 (선택 사항, 간단히 구조만 보여줌) ---
# 실제 추론 시에는 훈련된 모델을 로드해야 합니다.
# loaded_model = NextTemperatureModel(...) # 모델 구조는 동일하게 정의
# loaded_model.load_state_dict(torch.load('convlstm_temp_prediction_model.pth'))
# loaded_model.to(DEVICE)
# loaded_model.eval() # 평가 모드

# with torch.no_grad():
#    # 추론할 입력 시퀀스 (sequence_length, 3, 64, 64) 형태의 데이터 로드 또는 준비
#    # 예: inference_input_sequence_np = ...
#    # inference_input_sequence = torch.from_numpy(inference_input_sequence_np).float().to(DEVICE)
#    # inference_input_sequence = inference_input_sequence.unsqueeze(0) # 배치 차원 추가 (1, L, 3, 64, 64)

#    # predicted_next_temp_batch = loaded_model(inference_input_sequence) # (1, 1, 64, 64)
#    # predicted_next_temp_image = predicted_next_temp_batch.squeeze(0).cpu().numpy() # 배치, 채널 차원 제거 -> (64, 64) numpy 배열

#    # print(f"Predicted next temperature image shape: {predicted_next_temp_image.shape}")
#    # # 예측 결과 사용 (예: 시각화)

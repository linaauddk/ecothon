import pandas as pd
import numpy as np
import os
from datetime import datetime
from scipy.interpolate import griddata
from PIL import Image
from tqdm import tqdm

# 위경도 정보
location_coords = {
    "강남": (37.4982, 127.08162),
    "강동": (37.55556, 127.14498),
    "강북": (37.63972, 127.02576),
    "강서": (37.5739, 126.82953),
    "관악": (37.45284, 126.95015),
    "광진": (37.5338, 127.08566),
    "구로": (37.49328, 126.82629),
    "금천": (37.46551, 126.90016),
    "기상청": (37.4933, 126.91746),
    "남현": (37.46347, 126.98154),
    "노원": (37.62186, 127.09192),
    "도봉": (37.66557, 127.03042),
    "동대문": (37.58463, 127.06036),
    "마포": (37.55165, 126.92915),
    "서대문": (37.57047, 126.94078),
    "서초": (37.48462, 127.02601),
    "성동": (37.54721, 127.03885),
    "성북": (37.61134, 126.99981),
    "송파": (37.51151, 127.0967),
    "양천": (37.52823, 126.87937),
    "영등포": (37.52706, 126.90705),
    "용산": (37.51955, 126.97629),
    "은평": (37.64647, 126.94273),
    "중구": (37.55236, 126.98736),
    "중랑": (37.58551, 127.08682),
    "한강": (37.52489, 126.93904),
    "현충원": (37.50036, 126.97652)
}

# 그리드 설정
lat_range = (37.4, 37.7)
lon_range = (126.75, 127.2)
grid_res = 64
grid_y, grid_x = np.mgrid[lat_range[0]:lat_range[1]:grid_res*1j,
                          lon_range[0]:lon_range[1]:grid_res*1j]

# 경로 설정
root_dir = "서울_AWS_기온_지점별"
img_dir = "heatmaps_png"
npy_dir = "heatmaps_npy"
time_embed_path = "time_embeddings.npy"
os.makedirs(img_dir, exist_ok=True)
os.makedirs(npy_dir, exist_ok=True)

# 모든 CSV 불러오기
dfs = []
for loc_name in os.listdir(root_dir):
    loc_path = os.path.join(root_dir, loc_name)
    if not os.path.isdir(loc_path) or loc_name not in location_coords:
        continue
    for fname in os.listdir(loc_path):
        if fname.endswith(".csv"):
            file_path = os.path.join(loc_path, fname)
            df = pd.read_csv(file_path)
            df["지점명"] = loc_name
            dfs.append(df)

# 통합 DataFrame
df = pd.concat(dfs, ignore_index=True)
df["일시"] = pd.to_datetime(df["일시"])

# 결과 저장
image_list = []
embed_list = []

# 시간별 그룹화 후 반복
for timestamp, group in tqdm(df.groupby("일시")):
    coords = []
    temps = []

    for _, row in group.iterrows():
        loc = row["지점명"]
        temp = row["기온(°C)"]
        if loc in location_coords and not pd.isna(temp):
            lat, lon = location_coords[loc]
            coords.append((lat, lon))
            temps.append(temp)

    if len(coords) < 5:
        continue  # 보간 안 되는 경우

    temps = np.array(temps)
    norm_temp = (temps - temps.min()) / (temps.max() - temps.min() + 1e-5) * 255
    grid = griddata(coords, norm_temp, (grid_y, grid_x), method='cubic', fill_value=0)

    # 이미지 저장
    image_array = np.clip(grid, 0, 255).astype(np.uint8)
    image = Image.fromarray(image_array, mode='L')
    fname = timestamp.strftime("%Y%m%d_%H%M")
    image.save(os.path.join(img_dir, f"{fname}.png"))

    # 넘파이 저장용
    image_list.append(image_array)

    # 시간 임베딩 (1년 365일 기준)
    day_of_year = timestamp.timetuple().tm_yday
    sin_val = np.sin(2 * np.pi * day_of_year / 365)
    cos_val = np.cos(2 * np.pi * day_of_year / 365)
    embed_list.append([sin_val, cos_val])

# 저장
np.save(os.path.join(npy_dir, "heatmaps.npy"), np.stack(image_list))  # [T, 64, 64]
np.save(time_embed_path, np.array(embed_list))  # [T, 2]

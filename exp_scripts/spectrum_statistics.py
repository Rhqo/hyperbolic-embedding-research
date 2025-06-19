import json
import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib
if sys.platform != 'darwin':
    matplotlib.use('TkAgg')

# JSON 파일 읽기
file_names = ['hyperbolic_embeddings']

for file_name in file_names:
    with open(f'./embeddings/{file_name}.json', 'r') as file:
        data = json.load(file)

    # JSON 데이터를 2D 배열로 변환 (각 이미지의 512차원 배열을 한 행으로 변환)
    array_data = np.array([value[0] for key, value in data.items()], dtype=float)

    # 데이터 스케일 조정
    # array_data *= 1000

    # 'coolwarm' 컬러맵과 정규화 객체 준비 (Matplotlib 3.7 이상 권장 방식)
    norm = matplotlib.colors.Normalize(vmin=-0.3, vmax=0.3)
    cmap = plt.colormaps.get_cmap('coolwarm')  # 권장 방식

    # RGBA 이미지로 변환
    rgba_data = cmap(norm(array_data))

    # 히트맵 생성
    fig, ax = plt.subplots(figsize=(20, 16))
    im = ax.imshow(rgba_data, aspect='auto')

    # 컬러바 생성 (ScalarMappable에 array 할당 필요)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(array_data)
    fig.colorbar(sm, ax=ax)

    plt.title(f'Heatmap of {file_name} Data')
    plt.xlabel('Dimension')
    plt.ylabel('Image ID')
    plt.savefig(f'{file_name}_heatmap.png')
    plt.close()

    # 히스토그램 생성
    plt.figure(figsize=(10, 8))
    plt.hist(array_data.flatten(), bins=1000, color='blue', alpha=0.7)
    plt.title('Histogram of JSON Data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig(f'{file_name}_histogram.png')
    plt.close()
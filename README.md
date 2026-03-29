# adversarial-fgsm-pgd


## 📂 Project Structure
'''bash
├── ckpt/                   # 학습된 모델 체크포인트 저장
├── results/                # 생성된 적대적 예제 이미지 및 로그
├── attack.py               # FGSM / PGD 핵심 알고리즘 구현
├── models.py               # MNIST / CIFAR-10용 CNN 아키텍처
├── train.py                # 베이스라인 모델 학습 스크립트
├── test.py                 # 모델 평가 및 공격 실행 스크립트
├── analyze.ipynb           # 결과 데이터 분석 (JSON 기반)
├── visualize.ipynb         # 적대적 예제 시각화 노트북
├── attack_success_rate.json # 공격 성능 지표 결과
├── requirements.txt        # 의존성 패키지 목록
└── README.md

## Run attack / evaluation
python test.py

### 📊 Datasets
- **MNIST** (grayscale, 28×28)
- **CIFAR-10** (RGB, 32×32)

### ⚔️ Attacks
- **FGSM**
  - targeted
  - untargeted
- **PGD**
  - targeted
  - untargeted

### 📉 Loss Functions
- **Cross Entropy** (`ce`)
- **Mean Squared Error** (`mse`)
- **KL Divergence** (`kl`)
- **Margin Loss** (`margin`)

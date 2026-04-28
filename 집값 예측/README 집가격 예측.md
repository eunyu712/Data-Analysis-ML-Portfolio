# California Housing Price Prediction

> scikit-learn의 California Housing 데이터셋을 활용한 회귀 및 이진 분류 프로젝트

---

## 프로젝트 개요

캘리포니아 주택 데이터를 바탕으로 **중위 주택 가격(MedHouseVal)을 예측**하는 머신러닝 프로젝트입니다.  
회귀 모델(Linear, Polynomial, Ridge, Lasso) 비교 실험과, 동일 데이터를 이진 분류 문제로 변환하여 여러 분류 모델(KNN, Logistic Regression, Decision Tree, Random Forest, SVM)의 성능을 비교·분석하였습니다.

---

## 데이터셋

- **출처**: `sklearn.datasets.fetch_california_housing`
- **샘플 수**: 20,640개
- **결측치**: 없음 (0개)
- **Train / Test 분리**: 8:2 (random_state=42)
  - Train: 16,512개 / Test: 4,128개

### 변수 설명

| 변수 | 설명 |
|------|------|
| `MedInc` | 블록 내 가구 중위 소득 |
| `HouseAge` | 주택 평균 연식 |
| `AveRooms` | 가구당 평균 방 수 |
| `AveBedrms` | 가구당 평균 침실 수 |
| `Population` | 블록 내 인구 수 |
| `AveOccup` | 가구당 평균 거주자 수 |
| `Latitude` | 위도 |
| `Longitude` | 경도 |
| `MedHouseVal` | **타겟 변수** — 중위 주택 가격 (단위: $100,000) |

---

## 전처리

- **StandardScaler**를 사용하여 정규화 수행
- 데이터 누출 방지를 위해 **train 데이터에만 `fit`** 적용, test 데이터에는 `transform`만 적용

---

## EDA 및 Feature Selection

### 주요 발견

- **MedHouseVal 분포**: 5.0 부근에서 데이터가 급증하는 **천장 효과(ceiling effect)** 관찰
  - 50만 달러 초과 주택이 모두 5.0으로 기록됨 → 고가 주택 예측 정확도 제한
- **AveRooms & AveBedrms** 상관계수: **0.85** → 다중공선성 주의
- **Latitude & Longitude** 상관계수: **-0.92** → 다중공선성 주의

### 종속변수와의 피어슨 상관계수 (절댓값 기준 내림차순)

| 변수 | 상관계수 | \|상관계수\| |
|------|---------|------------|
| MedInc | +0.6882 | 0.6882 |
| AveRooms | +0.1521 | 0.1521 |
| Latitude | -0.1446 | 0.1446 |
| HouseAge | +0.1059 | 0.1059 |
| AveBedrms | -0.0469 | 0.0469 |
| Longitude | -0.0457 | 0.0457 |
| Population | -0.0246 | 0.0246 |
| AveOccup | -0.0237 | 0.0237 |

### Feature Selection

상관계수 절댓값 기준 상위 4개 변수 선택:  
**`MedInc`, `AveRooms`, `Latitude`, `HouseAge`**

> 선택 이유:
> - `MedInc`: 소득이 높을수록 집값이 높은 강한 양의 관계 (+0.6882)
> - `AveRooms`: 방이 많을수록 집값이 높은 양의 관계, 단 AveBedrms와의 다중공선성 고려하여 AveRooms만 선택
> - `Latitude`: 남쪽일수록(위도 낮을수록) 집값이 높은 음의 관계, Longitude와의 다중공선성 고려하여 Latitude만 선택
> - `HouseAge`: 오래된 집일수록 집값이 높은 약한 양의 관계

---

## 회귀 모델 결과

### 4개 Feature 사용 시 모델 성능 비교

| 모델 | Train R² | Test R² | R² 격차 | Test RMSE |
|------|----------|---------|---------|-----------|
| Linear Regression | 0.5225 | 0.5043 | 0.0182 | - |
| Polynomial Regression (degree=2) | 0.5501 | 0.5188 | 0.0313 | - |
| Ridge Regression (alpha=10) | 0.5501 | 0.5188 | 0.0313 | - |
| Lasso Regression (alpha=0.01) | ~0.52 | ~0.50 | 0.0179 | 0.8060 |

> ※ Lasso의 Train/Test R² 정확값은 실행 output 미저장으로 노트북 텍스트 기준 "약 0.50~0.52" 인용

> 세 모델 모두 Train-Test R² 격차가 **0.05 이하**로 **과적합 없음**  
> 단, R² ≈ 0.50~0.55 수준으로 4개 변수만으로는 집값을 충분히 설명하지 못함

### Polynomial Regression 특이사항

- 4개 feature → degree=2 변환 후 feature 수: **4 → 14개** (2차항 + 교호작용항)

### Ridge Regression — alpha 실험 (4개 Feature)

| alpha | Train R² | Test R² | Gap |
|-------|----------|---------|-----|
| 0.001 | ~0.5501 | ~0.5188 | ~0.0313 |
| 0.01 | ~0.5501 | ~0.5188 | ~0.0313 |
| 0.1 | ~0.5501 | ~0.5188 | ~0.0313 |
| 1 | ~0.5501 | ~0.5188 | ~0.0313 |
| 10 | 0.5501 | 0.5188 | 0.0313 |
| 100 | 유사 | 유사 | 유사 |
| 1000 | 감소 | 약간 감소 | 더 작아짐 |
| 100000 | 급격히 감소 | 급격히 감소 | - |

> 변수 수가 4개로 적어 alpha가 작은 구간에서 성능 변화가 거의 없음  
> alpha ≥ 1000부터 과소적합(underfitting) 발생

---

### 전체 변수(8개) 사용 시 Ridge 성능 향상

Ridge의 특성(모든 변수의 영향력을 살려둠, L2 패널티)을 고려하여 전체 8개 변수 사용 시 실험:

| 조건 | Train R² | Test R² |
|------|----------|---------|
| 4개 변수, alpha=10 | 0.5501 | 0.5188 |
| 8개 변수, alpha=100 | **0.69** | **0.66** |

> 8개 변수 모두 사용 시 Train R², Test R² 모두 **약 +14% 향상**  
> **최적 alpha = 100** (Test R² 최고점, Train-Test 격차 최소화)  
> alpha ≥ 1000부터 과소적합 발생 (w² 감소에 과도하게 집중)

- 8개 변수 → degree=2 변환 후 feature 수: **8 → 44개**

---

## 이진 분류 문제로 변환

MedHouseVal을 **중앙값 기준**으로 이진 변환:
- `1` (고가): 중앙값 이상 (상위 50%)
- `0` (저가): 중앙값 미만 (하위 50%)

**클래스 분포 (균형 잡힌 데이터)**:
- Train — 고가(1): 8,273 / 저가(0): 8,239
- Test — 고가(1): 2,052 / 저가(0): 2,076

### 분류 모델 성능 비교 (전체 8개 변수, StandardScaler 적용 / 다항변환 없음, Test 기준, F1-Score 내림차순)

| 순위 | 모델 | Accuracy | Precision | Recall | F1-Score |
|------|------|----------|-----------|--------|----------|
| 1 | Random Forest | 0.8932 | 0.8962 | 0.8879 | 0.8920 |
| 2 | KNN (k=5) | 0.8307 | 0.8279 | 0.8324 | 0.8301 |
| 3 | Logistic Regression | 0.8249 | 0.8237 | 0.8241 | 0.8239 |
| 4 | LinearSVC (C=1) | 0.8164 | 0.8165 | 0.8134 | 0.8149 |
| 5 | Decision Tree (max_depth=5) | 0.7873 | 0.8706 | 0.6720 | 0.7585 |

> **Random Forest**가 F1-Score 0.8920으로 압도적 1위  
> **Decision Tree**는 Precision(0.8706)은 높지만 Recall(0.6720)이 낮아 고가 주택을 많이 놓치는 경향

---

## 분석 및 해석

### Ridge vs Lasso

| | Ridge (L2) | Lasso (L1) |
|--|------------|------------|
| 패널티항 | λ × Σw² | λ × Σ\|w\| |
| 변수 제거 | X (계수가 0에 가까워지되 완전히 0이 되지 않음) | O (불필요한 변수 계수를 정확히 0으로) |
| 다중공선성 처리 | 상대적으로 강함 | 약함 |
| 적합 조건 | 모든 변수가 유효할 때 | 변수 선택이 필요할 때 |

### 과적합 판단 기준

- Train R² - Test R² 격차 **< 0.05** → 과적합 아님으로 판단
- 세 회귀 모델 모두 해당 기준 충족

### MedHouseVal 천장 효과

- 고가 주택(5.0 초과)이 모두 5.0으로 기록되어, **고가 주택 구간 예측 시 저평가되는 경향** 발생
- Actual vs Predicted 산점도에서 실제값 4~5 구간의 점들이 y=x 아래쪽에 편향됨

---

## 결론

1. **MedInc(중위 소득)** 이 집값과 가장 강한 상관관계(+0.6882)를 보이며 예측에서 핵심 변수
2. 상관계수 기반 Feature Selection으로 4개 변수를 선택했을 때 R² ≈ 0.52 수준으로 성능이 제한적
3. **Ridge 회귀에서 전체 8개 변수 사용 + alpha=100** 조합이 최적 (Test R² = 0.66), 4개 변수 대비 약 +14% 향상
4. 데이터의 천장 효과(MedHouseVal 5.0 제한)가 고가 주택 예측 정확도를 제한하는 구조적 한계로 작용
5. 이진 분류 문제로 변환 시 **Random Forest가 F1-Score 0.8920**으로 가장 높은 성능 기록, Decision Tree는 Recall이 0.6720으로 낮아 고가 주택 탐지에 취약

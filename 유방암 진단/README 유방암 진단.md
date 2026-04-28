# 유방암 진단 분류 모델 (Breast Cancer Classification)

> **Wisconsin Breast Cancer 데이터셋을 활용한 머신러닝 분류 모델 비교 분석**  
> KNN · SVM · Decision Tree · Logistic Regression · Random Forest · Gradient Boosting

---

## 프로젝트 개요

유방암의 조기 진단은 환자의 생존율에 직결되는 문제입니다.  
본 프로젝트는 `sklearn.datasets`에서 제공하는 **Wisconsin Breast Cancer 데이터셋(569개 샘플, 30개 피처)**을 활용하여 유방암의 악성(Malignant) / 양성(Benign) 여부를 분류하는 6가지 머신러닝 모델을 구현하고, 의료 데이터의 특성에 맞는 평가 기준(Recall, FN 최소화)으로 최적 모델을 선정합니다.

---

## 데이터셋

| 항목 | 내용 |
|------|------|
| 출처 | `sklearn.datasets.load_breast_cancer` |
| 전체 샘플 수 | 569개 |
| 피처 수 | 30개 (세포핵 특성 관련 수치 변수) |
| 클래스 | 0 = 악성(Malignant), 1 = 양성(Benign) |
| 악성 비율 | 37.26% (212개) |
| 양성 비율 | 62.74% (357개) |
| 결측치 | 없음 (0개) |

**Train / Test 분리 (7:3, `random_state=42`)**

| 분할 | 전체 | 악성(0) | 양성(1) |
|------|------|---------|---------|
| Train | 398개 | 149개 | 249개 |
| Test | 171개 | 63개 | 108개 |

**전처리:** `StandardScaler`를 이용하여 모든 피처를 평균=0, 표준편차=1로 표준화  
(KNN의 유클리드 거리 계산 시 스케일 차이로 인한 왜곡 방지)

---

## 피처 분석

히스토그램 시각화(30개 피처 × 악성/양성 분포)를 통해 분류에 유의미하지 않은 피처를 식별하였습니다.

**분류 변별력이 낮은 피처 (악성/양성 분포가 유사):**
`mean smoothness`, `mean symmetry`, `mean fractal dimension`, `texture error`, `smoothness error`, `compactness error`, `concavity error`, `concave points error`, `symmetry error`, `fractal dimension error`, `worst texture`, `worst smoothness`, `worst fractal dimension`

대부분의 피처에서 특정 임계값을 기준으로 악성/양성이 구분되는 패턴이 확인되었습니다.

---

## 모델 구현 및 실험

### 1. KNN (K-Nearest Neighbors)

k값 `[1, 3, 5, 7, 9, 11, 13, 15]`에 대한 Train / Test Accuracy 및 F1-Score 비교

| k | Train Acc | Test Acc |
|---|-----------|----------|
| 1 | 1.0000 | 낮음 (과적합) |
| 9 | — | **최고 Test 성능** |
| 15+ | 성능 감소 (과소적합) |

**최적 k = 9 선택 근거:**
- k < 9: Train 성능은 우수하나 Test 성능이 다소 낮음 → 과적합
- k > 9: Train/Test 성능 모두 지속 감소 → 과소적합
- k = 1일 때 Train Accuracy = 1.000 (자기 자신과 거리=0으로 완전 암기)
- k = 9에서 Train/Test 성능 차이가 가장 작고 Test 성능이 최고

**표준화 필수 이유:** KNN은 유클리드 거리 기반으로 이웃을 탐색하므로, 스케일이 큰 피처가 거리 계산을 지배하여 스케일이 작은 중요 피처가 무시될 수 있음

---

### 2. SVM (Support Vector Machine)

`LinearSVC` + `Pipeline(StandardScaler)` 구조, `loss='hinge'`, `max_iter=10000`  
C값 `[0.01, 0.1, 1, 10, 100]` 실험

| C | Train Acc | Test Acc | Precision | Recall | F1 |
|---|-----------|----------|-----------|--------|-----|
| 0.01 | 0.9774 | **0.9883** | 0.9818 | **1.0000** | 최고 |
| 0.1 | 0.9824 | 0.9825 | 0.9817 | 0.9907 | — |
| 1 | 0.9899 | 0.9766 | 0.9815 | 0.9815 | — |
| 10 | 0.9950 | 0.9649 | 0.9811 | 0.9630 | — |
| 100 | **0.9975** | 0.9298 | 0.9800 | 0.9074 | 하락 |

**C값 분석:**
- C가 작을수록: 마진 크게 유지, 분류 오류 허용 → 일반화 우수
- C가 클수록: 분류 오류를 최소화하는 데 집중 → Train 과적합 (C=100에서 Train 0.9975 vs Test 0.9298)
- **의료 데이터 관점 최적 C = 0.01**: Recall이 1.0000으로 가장 높아 실제 암 환자를 놓치지 않는 능력 최우수

**비교 시 사용 C = 1** (노트북 최종 비교 코드 기준)

---

### 3. Decision Tree

`max_depth=5`, `min_samples_leaf=3`, `random_state=42`

| Criterion | Train Acc | Test Acc |
|-----------|-----------|----------|
| Gini | 0.9874 | 0.9649 |
| Entropy | 0.9849 | 0.9766 |

**Gini vs Entropy:**
- Gini: 잘못 분류할 통계적 확률 기반, 제곱 연산으로 계산 빠름, 특정 클래스 지배적일 때 유리
- Entropy: 정보 획득량 극대화, 로그 스케일로 복잡한 분포 세밀 파악, 하위 피처에 중요도 고르게 분산
- Breast Cancer 데이터에서 클래스 경계가 비교적 뚜렷하여 두 방식 모두 유사한 분기점 탐색

**Feature Importance 상위 (Gini / Entropy 공통):**
1. `mean concave points` — 압도적 1위
2. 상위 3~4개 피처가 모델 예측의 대부분 설명

**하이퍼파라미터 튜닝 결과 (max_depth × min_samples_leaf 전수 탐색):**

| 조건 | 현상 |
|------|------|
| max_depth 작을수록 | Train/Test 모두 성능 감소 → 과소적합 |
| max_depth 클수록 | Train-Test 성능 Gap 증가 → 과적합 |
| min_samples_leaf=1 | Train 0.9950, Test 0.9532 → 과적합 |
| min_samples_leaf=5 | Train 0.9799, Test 0.9708 → 균형 |
| min_samples_leaf=10 | Train 0.9447, Test 0.9415 → 과소적합 |

---

### 4. 추가 모델 비교 (Logistic Regression · Random Forest · Gradient Boosting)

`random_state=42`, `n_estimators=100` (RF/GB 공통)

| 모델 | Accuracy | Precision | Recall | F1-Score |
|------|----------|-----------|--------|----------|
| Logistic Regression | **0.9825** | **0.9907** | 0.9815 | **0.9860** |
| Random Forest | 0.9708 | 0.9640 | **0.9907** | 0.9772 |
| Gradient Boosting | 0.9591 | 0.9633 | 0.9722 | 0.9677 |

**분석:**
- 로지스틱 회귀: Accuracy 0.9825, Precision 0.9907로 전반적으로 안정적인 성능
- 랜덤 포레스트: Recall 0.9907로 세 모델 중 최고 → **실제 암 환자를 놓치지 않는 핵심 목적에 가장 적합**
- 그래디언트 부스팅: 전 지표 0.95 이상이지만 타 모델 대비 소폭 낮음

---

## 최종 모델 비교 (KNN · SVM · DT)

| 모델 | Accuracy | Precision | Recall | F1 | FN 수 |
|------|----------|-----------|--------|----|-------|
| KNN (k=9) | 0.9708 | 0.9725 | 0.9815 | 0.9770 | 3개 |
| SVM (C=1) | **0.9766** | **0.9815** | **0.9815** | **0.9815** | **2개** |
| DT Gini | 0.9649 | 0.9722 | 0.9722 | 0.9722 | 3개 |

### FN(False Negative) 중심 평가

유방암 진단 맥락에서:
- **FN (위험):** 실제 유방암인데 정상으로 오진 → 치료 지연 → 암 진행 및 전이 → 생존율 급감, 사망 가능
- **FP (상대적으로 낮은 위험):** 정상인데 암으로 오진 → 심리적 불안 + 추가 정밀검사 → 오진 확인 가능, 실질적 생존 위협 없음

따라서 **Recall(실제 암 환자 중 암으로 올바르게 예측한 비율)** 이 핵심 평가 지표이며, FN이 가장 적은 모델이 최우선입니다.

> 노트북 Confusion Matrix 분석 결과: KNN FN=2개, SVM FN=2개, DT(Gini) FN=3개 — KNN과 SVM이 공동 최소, Accuracy까지 고려 시 SVM이 최적

---

## 최적 모델: **SVM (C=1)**

**선택 근거 3가지:**

1. **성능:** Accuracy 0.9766(최고), Recall 0.9815(KNN과 공동 최고), FN 2개(DT 3개 대비 최소)
2. **일반화:** 최대 마진 최적화가 목적함수에 내재되어 과적합/과소적합 위험이 낮고, 새로운 환자 데이터에도 안정적으로 작동
3. **설명 가능성 한계 수용:** SVM은 고차원 초평면 기반으로 비전문가에게 설명이 어렵지만, 의료 현장에서는 설명 가능성보다 **진단 정확도가 우선**시됨

> **결론:** FN 수가 가장 적고 (2개), Accuracy(0.9766)가 가장 높고, Recall(0.9815)이 가장 높으며, 과적합 위험이 낮아 실제 의료 현장에서의 일반화 성능이 가장 뛰어남


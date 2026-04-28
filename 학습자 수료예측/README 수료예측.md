# BDA 학회원 수료 예측 머신러닝 프로젝트

> **데이터 분석 학회(BDA) 학회원의 수료 여부를 예측하는 이진 분류 프로젝트**  
> 설문 데이터 기반 피처 엔지니어링 → 앙상블 모델 → 최적 Threshold 탐색

---

## 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **태스크** | 이진 분류 (수료 여부 예측: 0 / 1) |
| **데이터** | BDA 학회 9기 가입 설문 데이터 (`train.csv`, `test.csv`) |
| **Train 데이터 크기** | 748행 |
| **평가 지표** | F1 Score |
| **사용 언어** | Python 3 |
| **주요 라이브러리** | pandas, scikit-learn, XGBoost, scipy |

---

## 분석 파이프라인

```
데이터 로드
    ↓
결측치 분석 (카이제곱 검정 기반 변수 선별)
    ↓
이상치 / 오류값 처리
    ↓
피처 엔지니어링 (범주 재분류 + 파생 변수 생성)
    ↓
피처 선택 (Cramér's V, Mann-Whitney U, 카이제곱 검정)
    ↓
One-Hot 인코딩
    ↓
모델 학습 (Logistic Regression / XGBoost / Random Forest)
    ↓
앙상블 + 최적 Threshold 탐색
    ↓
최종 예측 제출
```

---

## 1단계: 데이터 전처리

### 1-1. 결측치 처리

결측 비율이 50%를 초과하는 변수들에 대해 **카이제곱 검정(Chi-square test)** 을 수행하여, 결측 여부와 수료 여부 간의 독립성 검정을 실시했습니다. (`p-value > 0.05` → 수료 여부와 무관 → 제거)

**제거한 변수 (12개):**
- `class2`, `class3`, `class4`
- `contest_award`, `idea_contest`, `contest_participation`
- `previous_class_3`, `previous_class_4`, `previous_class_5`, `previous_class_6`, `previous_class_7`, `previous_class_8` (결측 비율 80% 이상)

**`nationality` (내/외국인):** 결측치가 있는 행 자체를 제거  

**학교(`school1`) 관련 전공 변수 처리:**
- 대학 미진학(`school1 == 0`)인 경우: `major type`, `major_field`, `major1_1`, `major1_2`, `completed_semester` → `'없음'`으로 처리
- 대학 진학(`school1 != 0`)이나 `completed_semester` 결측인 경우: 평균값(6학기)으로 대체

**개별 결측치 직접 처리:**
- train: `major type` 3건(ID 166→복수전공, 393→단일전공, 290→단일전공), `major_field` 3건: 전공 정보를 참조해 직접 값 입력
- test: `major type` 1건(ID 344→단일전공), `major_field` 1건(ID 344→의상학과) 직접 입력

---

### 1-2. 이상치 및 오류값 처리

| 변수 | 문제 | 처리 방식 |
|------|------|-----------|
| `completed_semester` (이수학기) | 8학기 초과값 존재 (e.g. `20241`) | 8 이상 → 8로 교정 |
| `time_input` (하루 가능 시간) | 24시간 등 비현실적 값 존재 | 10 초과 → 평균값으로 대체 |

---

## 2단계: 피처 엔지니어링

원본 텍스트/수치형 변수들을 의미 기반으로 재범주화하여 **19개의 파생 변수**를 생성했습니다.

| 파생 변수 | 원본 변수 | 변환 기준 |
|-----------|-----------|-----------|
| `cs_category` | `completed_semester` | 0(없음), 1(0~2학기), 2(3~4), 3(5~6), 4(7~8) |
| `ti_category` | `time_input` | 0(0시간), 1(1~3시간), 2(4~6시간), 3(7~10시간) |
| `ir_category` | `inflow_route` | SNS(에브리타임·인스타그램·교내플랫폼·기타) / 기존회원 / 지인추천 / 검색(인터넷검색·대외활동) |
| `wB_category` | `whyBDA` | 관리 / 만족 / 혜택 / 부담없음 |
| `hg_category` | `hope_for_group` | 오프라인그룹 / 온라인그룹 / 개인 |
| `wtg_category` | `what_to_gain` | 데이터분석역량 / 경험 / 인적네트워크 |
| `job_category` | `desired_job` | 데이터관련(1) / 비데이터(0) |
| `cert_level` | `certificate_acquisition` | 없음(0) / 1개(1) / 2개 이상(2) |
| `dc_category` | `desired_certificate` | 데이터 자격증(1) / 기타(0) |
| `edjob_category` | `desired_job_except_data` | IT관련(1) / 비IT(0) |
| `il_category` | `incumbents_lecture` | 커리어패스과정 / 직무강의 / 산업트렌드 |
| `icl_category` | `incumbents_company_level` | 국내대기업IT / 국내빅테크IT / 해외기업 / 기타 |
| `ils_category` | `incumbents_lecture_scale` | 10명내외 / 50명내외 / 100명이상 리스너 |
| `ic_category` | `interested_company` | 있음(1) / 없음(0) |
| `ed_category` | `expected_domain` | 데이터중요도메인(1) / 기타(0) |
| `mf_category` | `major_field` | IT전공(1) / 비IT(0) |
| `sch_category` | `school1` | 빈도 15 이상 학교 개별 분류, 나머지 '기타' |
| `j_category` | `job` | 학생(대학생·대학원생) / 직장인 / 취준생 |
| `level_category` | `incumbents_level` | 주니어(1) / 시니어(2) |

피처 엔지니어링 후 불필요해진 원본 변수 정리:
- **불필요 변수 6개 제거** (`col_drop1`): `generation`, `major1_1`, `major1_2`, `major_data`, `incumbents_lecture_scale_reason`, `onedayclass_topic`
- **파생 변수로 대체된 원본 변수 19개 제거** (`col_drop2`): `school1`, `inflow_route`, `whyBDA`, `what_to_gain`, `hope_for_group`, `major_field`, `completed_semester`, `time_input`, `desired_job`, `certificate_acquisition`, `desired_certificate`, `desired_job_except_data`, `incumbents_lecture`, `incumbents_company_level`, `incumbents_lecture_scale`, `interested_company`, `expected_domain`, `job`, `incumbents_level`

---

## 3단계: 피처 선택

세 가지 통계 검정 방법을 결합하여 예측력이 낮은 변수를 제거했습니다.

| 방법 | 대상 변수 유형 | 기준 |
|------|---------------|------|
| **Cramér's V** | 범주형 변수 | 수료 여부와의 연관성 측정 |
| **Mann-Whitney U 검정** | 순서형 수치 변수 | `p-value < 0.05` 기준 |
| **카이제곱 + Phi 계수** | 이진 변수 | `p < 0.05` & `φ ≥ 0.1` 기준 |

이후 모델에 따라 두 가지 피처셋을 별도로 구성했습니다.

**① Logistic Regression + XGBoost 앙상블용 (`X_tree` / `X_linear`)**

공통 제거 변수 (`drop`, 14개): `nationality`, `project_type`, `ti_category`, `wtg_category`, `job_category`, `dc_category`, `edjob_category`, `ils_category`, `ic_category`, `ed_category`, `mf_category`, `level_category`, `ID`, `completed`

선형 모델 추가 제거 (`drop_li`, 3개): `wB_category`, `ir_category`, `major type`

**② XGBoost + Random Forest 앙상블용 (`X_train2` / `test8`)**

제거 변수 (`col_6`, 16개): `desired_career_path`, `il_category`, `icl_category`, `j_category`, `ils_category`, `wtg_category`, `project_type`, `nationality`, `level_category`, `ti_category`, `ic_category`, `mf_category`, `edjob_category`, `ed_category`, `dc_category`, `job_category`

---

## 4단계: 모델링

### 모델 구성

모델별로 피처셋을 다르게 구성하여 최적 성능을 유도했습니다.

- **트리 모델용 피처 (`X_tree`):** One-Hot 인코딩 적용, `drop='first'` 미사용
- **선형 모델용 피처 (`X_linear`):** 다중공선성 방지를 위해 `wB_category`, `ir_category`, `major type` 추가 제거 후 `drop='first'` 적용, `StandardScaler` 정규화

### Logistic Regression (L1 정규화)

```python
LogisticRegression(
    penalty='l1',
    solver='liblinear',
    C=4.64,          # GridSearchCV로 탐색한 최적값
    max_iter=1000
)
```

- `StratifiedKFold(n_splits=5)`로 OOF(Out-of-Fold) 검증
- 각 Fold에서 `GridSearchCV`로 최적 C 탐색 (`C ∈ [0.01, 10]`)

### XGBoost

```python
XGBClassifier(
    n_estimators=300~600,
    max_depth=3~4,
    learning_rate=0.03~0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=neg/pos  # 클래스 불균형 보정
)
```

### Random Forest

```python
RandomForestClassifier(
    n_estimators=300~500,
    max_depth=6~8,
    class_weight='balanced',  # 클래스 불균형 보정
    n_jobs=-1
)
```

---

## 5단계: 앙상블 전략

노트북에서는 두 가지 앙상블 전략을 실험했습니다.

### ① Logistic Regression + XGBoost 앙상블

`X_linear` / `X_tree` 피처셋을 각각 사용한 두 모델의 예측 확률을 가중 평균:

```
final_prob = 0.8 × P(LogisticRegression) + 0.2 × P(XGBoost)
```

- `StratifiedKFold(n_splits=5)` OOF로 각 모델 확률 산출
- 가중치 w: `np.arange(0.1, 0.9, 0.05)` 범위에서 탐색 → 최적 w = **0.55 (Linear)** (초기 고정값 0.8로도 실험)
- 최적 Threshold: OOF F1 Score 기준으로 `np.arange(0.1, 0.6, 0.01)` 범위 탐색 → 최적 threshold = **0.2**
- 최종 C 값: GridSearchCV(`C ∈ np.logspace(-2, 1, 10)`, cv=3) 탐색 결과 `C=4.64` 사용

### ② XGBoost + Random Forest 앙상블 (최종 제출 전략)

`X_train2` 피처셋을 사용하여 두 모델의 예측 확률을 가중 평균:

```
final_prob = w × P(XGBoost) + (1-w) × P(RandomForest)
```

**Multi-Seed 앙상블:** 과적합 방지를 위해 **3개의 랜덤 시드(42, 52, 62)** 에서 각각 `StratifiedKFold(n_splits=5)`를 수행한 뒤 평균:

```python
SEEDS = [42, 52, 62]
# 각 seed별 OOF 확률 평균 → 최종 OOF로 최적 가중치/threshold 탐색
```

**최적 가중치 및 Threshold 탐색:**
- 가중치 w: `np.arange(0.34, 0.46, 0.005)` 범위에서 탐색
- Threshold t: `np.arange(0.32, 0.42, 0.002)` 범위에서 탐색
- 평가 기준: OOF F1 Score (단, 극단적 예측 방지를 위해 양성 예측 비율 `0.25 ~ 0.75` 제한)

---

## 결과

| 실험 | 모델 조합 | 비고 |
|------|-----------|------|
| 실험 1 | Logistic Regression (L1) 단독 | `submission.csv` |
| 실험 2 | Logistic Regression + XGBoost (0.8/0.2) | `submission_tl4.6482.csv` 등 |
| 실험 3 | XGBoost + RandomForest (0.55/0.45) | `submission_a.csv` 등 |
| **최종** | **XGBoost + RF, Multi-Seed(42/52/62), StratifiedKFold(5)** | **`submission_final_finetuned.csv`** |

> 모든 실험에서 평가 지표는 **F1 Score**이며, OOF 기반 최적 Threshold를 탐색하여 적용했습니다.

---

## 주요 인사이트 및 의사결정

1. **결측 패턴 자체를 활용:** 결측 비율이 높더라도 결측 여부와 수료 여부 사이의 관계가 있을 경우 변수 보존 또는 파생 변수 생성 고려 (카이제곱 검정 활용)

2. **도메인 기반 범주 재설계:** 텍스트 응답형 설문 변수를 단순 레이블 인코딩이 아닌 **의미 기반으로 재그룹화**하여 모델의 학습 효율 향상

3. **선형/트리 모델 피처 분리:** 다중공선성에 민감한 로지스틱 회귀와 이에 강건한 트리 계열 모델에 대해 **서로 다른 피처셋과 인코딩 전략** 적용

4. **Threshold 최적화:** 클래스 불균형 상황에서 기본 0.5 threshold가 아닌 **F1 Score를 기준으로 최적 분류 경계값**을 탐색하여 성능 향상

5. **Multi-Seed 앙상블:** 단일 랜덤 시드 의존성을 줄이기 위해 복수 시드 결과를 평균하여 **예측 안정성 확보**

---

## 회고

- 설문 데이터 특성상 자유 텍스트 응답이 많아 **수작업 범주 설계**에 많은 시간을 투자했습니다.
- 피처 수가 적고 데이터 수(748개)가 작은 환경에서 과적합 방지를 위한 **정규화 및 앙상블 전략**이 중요했습니다.
- 통계 검정 기반 피처 선택을 통해 모델의 **해석 가능성**을 유지했습니다.

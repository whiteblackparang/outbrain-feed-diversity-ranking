# Outbrain Feed Diversity Ranking

## 문제 정의

개인화 추천의 클릭 예측 정확도와 추천 결과의 다양성 사이 트레이드오프를 정량적으로 관리할 수 있는가

## 데이터

- 출처: [Outbrain Click Prediction](https://www.kaggle.com/c/outbrain-click-prediction) (Kaggle)
- 규모: clicks_train 8,714만 행, events 2,312만 행, promoted_content 55,958건
- 기간: 2016년 6월 14일 ~ 6월 28일, 미국 내 다수 퍼블리셔 사이트
- 주요 컬럼: display_id, ad_id, uuid, document_id, category_id, topic_id, entity_id, confidence_level

## 기술 스택

| 영역 | 도구 |
| --- | --- |
| 데이터 처리 | pandas, scipy.sparse |
| 차원 축소 | scikit-learn (TruncatedSVD) |
| 딥러닝 모델 | PyTorch (Multi-head Self-Attention) |
| 재랭킹 | MMR (Maximal Marginal Relevance) |
| LLM 활용 | Groq API (openai/gpt-oss-20b) — 재랭킹 결과 자연어 설명 생성 |
| 시각화 | matplotlib |

## 분석 과정

| 단계 | 내용 |
| --- | --- |
| EDA | display_id당 후보 수 분포(평균 5.16개), clicked 비율(19.4%), document_id 커버리지(99.68%) 확인 |
| 전처리 | clicks_train → events → promoted_content 조인, 카테고리·토픽·엔티티 confidence를 sparse 행렬로 결합 후 SVD로 128차원 압축 |
| 베이스라인 | 인기도 기반 CTR, 카테고리 기반 CTR |
| 딥러닝 모델 | 같은 세션(display_id) 내 후보들을 self-attention으로 인코딩하는 세션 어텐션 모델 |
| 다양성 재랭킹 | MMR로 lambda값별 hit rate–다양성(ILD) 트레이드오프 실험, LLM으로 재랭킹 근거 자연어 설명 생성 |

## 주요 발견

- 인기도 베이스라인 AUC 0.704, MRR 0.634 — 카테고리 베이스라인(AUC 0.580)보다 유의하게 높음. 카테고리 단위로 뭉뚱그리면 개별 광고 단위 신호를 놓친다는 것을 보여줌
- 세션 어텐션 모델 AUC 0.711로 인기도 베이스라인을 근소하게 상회. 콘텐츠 피처만으로는 광고 크리에이티브·브랜드 인지도 같은 인기도에 암묵적으로 반영된 요인을 완전히 대체하기 어려움을 시사
- lambda를 0.5에서 1.0으로 올릴수록 hit rate는 0.917에서 0.940으로 상승, 다양성(ILD)은 0.88에서 0.83으로 하락하는 명확한 트레이드오프 확인
- 세션의 44%만 후보가 5개를 초과해, 다양성 재랭킹의 실질적 효과는 후보가 많은 세션에서 더 뚜렷하게 나타남
- 최초 계획한 유저 히스토리 기반 개인화(GRU 시퀀스)는 유저 재방문율이 낮은 데이터 특성상(히스토리 86%가 공백) 성립하지 않아, 같은 세션 내 후보 간 관계를 활용하는 방향으로 설계를 전환함

## 폴더 구조

```
outbrain-feed-diversity-ranking/
├── data/
├── src/
│ ├── evaluate.py
│ ├── models.py
│ ├── preprocessing.py
│ └── reranker.py
├── notebooks/
│ ├── 01_eda.ipynb
│ ├── 02_baseline.ipynb
│ ├── 03_two_tower_training.ipynb
│ └── 04_diversity_tradeoff.ipynb
├── reports/
│ └── final_report.md
├── README.md
└── requirements.txt

```

## 실행 방법

pip install -r requirements.txt
jupyter notebook notebooks/01_eda.ipynb


노트북 01 → 04 순서로 실행. 04번은 Groq API 키를 환경변수 `GROQ_API_KEY`로 설정해야 LLM 설명 생성 셀이 동작함

## 상세 리포트

전체 분석 과정과 시도했다 폐기한 방향(MIND → Outbrain 전환, 유저 시퀀스 → 세션 기반 전환)은 [reports/final_report.md](reports/final_report.md) 참고
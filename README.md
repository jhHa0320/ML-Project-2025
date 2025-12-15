# 머신러닝 기반 외식 서비스 리뷰 감성 분석 및 만족도 지수 개발

> **프로젝트 기간**: 2025.11 ~ 2025.12.08  
> **수행자**: AI융합학부 20243305 하정훈

## 1. 프로젝트 개요
### 프로젝트 목표
외식 서비스(식당)의 긍/부정 리뷰를 분석하여 실제 고객 만족도를 객관적으로 평가할 수 있는 **'AI 만족도 지수'**를 제시합니다.
기존의 단순 평균 별점이 가지는 정보의 비대칭성과 주관성 문제를 해결하고, 리뷰 텍스트 내면에 숨겨진 실제 만족도를 정량화하는 것을 목표로 합니다.

### 핵심 성과
- **모델 정확도**: TF-IDF와 로지스틱 회귀 기반 감성 분류 모델 최종 정확도 **97% 이상** 달성
- **지표 개발**: AI 모델 기반의 만족도 지수 및 '거품 지수(Gap Index)' 개발
- **인사이트 도출**: 리뷰 평균 평점과 AI 만족도 지수를 비교하여 식당의 과대/과소 평가 여부(거품 지수) 확인

## 2. 사용 기술 (Tech Stack)
- **Language**: Python
- **Machine Learning**: Scikit-learn (Logistic Regression, Random Forest, XGBoost)
- **Data Processing**: Pandas, NumPy, NLTK
- **Visualization**: Matplotlib

## 3. 데이터셋 및 전처리
### 데이터셋
- **Main Data**: [Zomato Bangalore Restaurants](https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants) (Kaggle)
- **Validation Data**: [Yelp Dataset](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset) (Kaggle)

### ⚙️ 전처리 파이프라인
1. **파싱 및 정규화**: HTML 태그, URL, 특수문자, 이모지 제거
2. **구조 변환**: `reviews_list`를 explode하여 개별 행으로 분리
3. **토큰화 및 정제**: NLTK 활용 소문자 변환, 불용어 제거, 표제어 추출
4. **라벨링**:
    - **긍정(1)**: 4~5점
    - **부정(0)**: 1~2점
    - **3점 처리**: 초기 학습에서 제외 후, 학습된 모델로 재평가하여 만족도 지수에 활용

## 4. 모델링 및 실험 결과
### 주요 실험
- **Feature Extraction**: TF-IDF (Max Features 50,000, N-gram (1,2))
- **Model Selection**:
    - **Logistic Regression (채택)**: 정확도 0.9789 (해석 용이성 우수)
    - **Random Forest**: 정확도 0.9935 (성능은 최고이나 복잡도 높음)
    - **XGBoost**: 정확도 0.9638

### 분석 결과
- **단어 분석**: 1-gram보다 **1,2-gram(Bi-gram)** 사용 시 감성 문맥 파악 성능이 우수함 ('good food' 등 조합의 중요성)
- **일반화 성능 검증**:
    - **Yelp 데이터(타 플랫폼)**: 상관계수 **0.9264** (매우 높음)
    - **일본어 리뷰(타 언어)**: Multilingual BERT 적용 시 상관계수 **0.5866** (유의미한 양의 상관관계)

## 5. 결론 및 제언
### 연구 의의
- **객관적 지표 제시**: 단순 별점이 아닌 텍스트 문맥 기반의 확률적 점수(Gap 지수) 제공
- **데이터 활용 극대화**: 모호한 3점 리뷰를 재평가하여 숨겨진 긍/부정 판별
- **높은 일반화 성능**: Zomato(인도) 데이터로 학습한 모델이 Yelp(미국) 데이터에서도 높은 적중률을 보임

### 한계 및 향후 과제
- **언어적 뉘앙스**: TF-IDF의 한계를 넘어 BERT, GPT 등 트랜스포머 모델 도입 필요
- **다국어 확장**: 충분한 다국어 데이터셋 확보를 통한 글로벌 모델로의 고도화
- **최신성 반영**: 리뷰 작성 시점에 따른 가중치 적용(Time-decay) 모델 개발 예정

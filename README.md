# GPT2_chatbot

## Summary
KoGPT2의 사전학습된 모델을 활용하여 Chatbot을 생성하였다. 이때 AI Hub의 정신건강 상담 데이터를 이용한 미세조정을 통해 우울증 방지에 목적을 둔 AI Chatbot을 구축하였다. 

데이터 수집 과정에서 자연어 데이터셋 구축의 어려움을 이해하고 이를 보정하기 위해 BLEU평가지표를 이용하여 데이터 전처리를 진행하였다. 또한 모델 구축 과정에서 Tokenizer부터 GPT와 같은 트랜스포머 기반의 모형까지 다양한 자연어 처리 방식과 응용 모델을 활용하였다. 평가지표로는 BLEU를 통해 챗봇의 성능을 평가하였다.

본 프로젝트를 통해 자연어 처리 모델링 과정에서의 편향, 대용량 사전학습 모델의 비용 문제 등의 문제를 접하며 앞으로 자연어 처리의 방향성에 대해서도 생각해보게 되었다.

## Project

### 서론
#### 주제
  우울증과 같은 정신질환을 겪고 있는 분들에게 치유를 해줄 수 있는 챗봇을 개발한다.
#### 접근방식
  GPT2모델에 정신질환 관련 데이터를 학습시켜 챗봇을 만든다
#### 기대효과
  최근 우울증 진료 환자가 증가하는 추세인 만큼 정신질환 치료와 관련된 관심이 늘어나고 있다. 이런 환자들에게 정신치료 챗봇을 제공하며 꼭 사람과 대화하지 않더라도 챗봇을 통해 치유가 가능하게 한다.
#### Chatbot에 대한 이해
  * Open-domain chatbot: 
    아무 주제나 대화가 가능하며 특별한 목적이 없는 챗봇
    EX) Google의 Meena, FaceBook의 Blender
  * Closed-domain chatbot:
    단어나 의도(intent)에 반응하며 특정한 과업을 달성하는 것이 목표이다.
    EX) 기술 고객 지원 챗봇, 쇼핑 도우미 챗봇
    
    **본 챗봇은 우울증 방지가 목표이므로 Closed-domain Chatbot을 활용한다.**

### 데이터
#### AI Hub의 '웰니스 대화 스크립트 데이터셋' 활용
  * 정신건강 상담 주제의 챗봇 발화 (1,023개)
#### github의 챗봇 데이터 활용
  * [https://github.com/songys/Chatbot_data/raw/master/ChatbotData .csv]

### 전처리
#### 중복 문장 제거
  ![중복Q](https://user-images.githubusercontent.com/75753717/123137537-0426de00-d48f-11eb-96ab-c2f4a5eda86c.PNG)
  * 중복되어 있는 질문과 답변 존재
    → 중복된 답변을 가진 질문들 중 비슷한 의미의 질문은 제거
#### 데이터 분할
  * 대화 데이터 셋을 Train, Validation, Test set으로 나누어 분석 진행
#### 토크나이저
  * taeminlee의 KoGPT2 토크나이저를 활용
  * 데이터에 시작토큰과 종결토큰을 추가하여 <s> Q: 질문 A:대답 </s>의 형태로 토큰화
  ![토크나이저](https://user-images.githubusercontent.com/75753717/123137539-04bf7480-d48f-11eb-8463-c553fd42b7e6.PNG)

### 모형 및 데이터 입력 방식
#### 사전학습모델(PreTrained Model) - KoGPT2
  * GPT2란? 
    - https://www.notion.so/Generative-Pre-Training-GPT-6994270592e0446d879484f98732c415
  * KoGPT2: GPT의 부족한 한국어 성능을 극복하기 위해 개발된 한국어 디코더 언어 모델
    - Tokenizer
      + BPE tokenizer를 이용하여 학습
#### 데이터 입력 방법
  ![데이터입력방법](https://user-images.githubusercontent.com/75753717/123137532-02f5b100-d48f-11eb-899c-05617a5b7eb4.PNG)
  * 토큰화의 방식은 위의 토크나이저 부분에서 설명한 부분과 동일하다.
#### 학습 및 미세조정
  * 미세조정의 과정
  ![미세조정과정](https://user-images.githubusercontent.com/75753717/123137533-038e4780-d48f-11eb-9dbc-00aca33a3c96.PNG)
    - 최대한 많은 데이터를 이용하기 위해 1차 미세조정을 통해 최선의 에포크를 탐색한 뒤, 전체의 데이터를 이용하여 2차 미세조정을 하여 최종 모델을 평가한다.
  ![1epoch](https://user-images.githubusercontent.com/75753717/123137521-012bed80-d48f-11eb-9915-c156831a895a.PNG)
    - 컴퓨터 사양 상의 이유로 32배치를 이용하여 총 10번의 epoch동안 학습을 진행하였고 최선의 epoch은 9번으로 확인하였다.
  ![9](https://user-images.githubusercontent.com/75753717/123137527-025d1a80-d48f-11eb-8955-78a8cc62e553.PNG)
### 평가
#### 최선의 대답 찾기
  * 디코딩 시에 다양한 파라미터 조정을 통해 최선의 답을 찾기 위해 노력하였다.
    - max_length=50 - 답변의 최장 길이가 50을 넘지 않아야 한다
    - do_sample = True
    - top_k=50 - 확률의 상위 50개의 후보집합에서 문장들을 선택한다.
    - top_p=0.95 - 확률의 누적이 0.95가 되는 후보 집합에서 문장들을 선택한다.
    - num_return_sequence=3 총 3개의 결과를 디코딩 한다.
#### 평가지표
  * BLEU
    - 단어 겹침 측정치 중 하나
    - modified n-gram precision(Pn)
      + 단어 하나하나의 겹침 뿐만 아니라 n-gram(연속된 n개의 단어들)의 겹침치도 구하여 평균을 낸다
    - smoothing
      + BLEU 계산 시 0이 나올 경우 보정을 하는 방법
      + 본 분석에서는 precision이 0이면 문장의 길이에 반비례하게 precision을 채우는 4번 방법을 활용
    - BP
      + 짧은 문장의 경우 분모가 작아지므로 precision이 커지는 경우를 방지
      + 짧음(Brevity)에 불이익(penalty)를 주어 균형을 맞춤
   
    **test data의 평균 BLEU: 0.2715912..**
### 결과
  ![결과](https://user-images.githubusercontent.com/75753717/123137530-02f5b100-d48f-11eb-97bd-3b091cb6400b.PNG)
  * 위로해주는 문맥을 가진 답변들이 생성되었다. 그러나 앞선 질문에 있는 단어를 반복하여 대답한다는 것이 본 챗봇이 더 발전해야 할 부분이라고 생각한다.

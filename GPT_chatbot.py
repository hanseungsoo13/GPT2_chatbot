#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install transformers')


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import TFGPT2LMHeadModel, PreTrainedTokenizerFast
import tensorflow as tf
import tqdm
from nltk.translate.bleu_score import SmoothingFunction, modified_precision, sentence_bleu


# # 2. 모델링 및 데이터

# In[3]:


get_ipython().system('wget -c https://github.com/songys/Chatbot_data/raw/master/ChatbotData%20.csv')


# In[4]:


df2=pd.read_csv('ChatbotData .csv')


# In[5]:


from google.colab import drive
drive.mount('/gdrive')


# In[6]:


df=pd.read_excel('/gdrive/My Drive/웰니스_대화_스크립트_데이터셋.xlsx')


# In[7]:


df.columns=['label','Q','A']


# In[8]:


chat=pd.concat([df2,df],axis=0).reset_index(drop=True)
chat.dropna(inplace=True)
chat=chat.reset_index(drop=True)


# # 3. 전처리

# In[9]:


chat.Q.value_counts() #데이터의 Q열에 중복된 질문들이 있음


# In[10]:


chat.A.value_counts() #데이터의 각기 다른 질문에 대한 중복된 대답이 있음


# In[11]:


chat[chat.Q=='외로워'] #같은질문에도 다른 대답들이 들어있음


# - 그중에서 중복된 대답의 갯수가 많은 것으로 판단되어 중복된 대답을 가진 질문 중 질문의 내용이 비슷한 행을 제거하기로 하였다.

# In[12]:


l=pd.DataFrame(df2.A.value_counts())[pd.DataFrame(df2.A.value_counts()).loc[:,'A']>2].index.tolist()


# In[13]:


sameanswer=chat[chat['A'].isin(l)]['Q'].tolist()


# 비슷한 질문을 파악하기 위해 bleu지표를 사용하여 bleu가 0.85가 넘는 질문은 비슷한 내용의 질문이라고 판단하고 제거한다

# In[14]:


bl=[]
for i, j in tqdm.tqdm_notebook(enumerate(sameanswer)):
  
  if i<2060:
    for k in range(i+1,i+10):
      bleu=sentence_bleu(j, sameanswer[k])
      if bleu>0.85: 
        bl.append(j)
  else:
    for k in range(i,2069-i):
      bleu=sentence_bleu(j, sameanswer[k])
      if bleu>0.85:
        bl.append(j) 


# In[15]:


chat=chat.drop(index=chat.Q[chat.Q.isin(list(set(bl)))].index)


# In[47]:


#test셋 형성
train=chat[:-26]
test=chat[-26:]


# In[17]:


import random
random.seed(30)


# In[18]:


valid_num=random.sample(range(len(train)),3684)


# In[19]:


plt.hist(valid_num) #어느정도 고르게 분포되어있다
plt.show()


# In[20]:


#validation set 생성
valid_chat=trainiloc[valid_num,:]
train_chat=train.iloc[~train.index.isin(valid_num),:]


# In[22]:


model = TFGPT2LMHeadModel.from_pretrained("/gdrive/My Drive/kogpt2/taeminlee", from_pt=True)
tokenizer = PreTrainedTokenizerFast.from_pretrained("/gdrive/My Drive/kogpt2/taeminlee")


# In[23]:


traindata = []
for row in train_chat.itertuples():
    print(row)
    tokens = tokenizer.encode(f'<s>Q: {row.Q} A: {row.A}</s>') #Q:question, A:answer
    tokens = [t for t in tokens if t < tokenizer.vocab_size]
    traindata.append(tokens)


# In[24]:


validdata = []
for row in valid_chat.itertuples():
    print(row)
    tokens = tokenizer.encode(f'<s>Q: {row.Q} A: {row.A}</s>') #Q:question, A:answer
    tokens = [t for t in tokens if t < tokenizer.vocab_size]
    validdata.append(tokens)


# # 4. 모형 및 데이터 입력 방식

# In[26]:


def data_generator():
  for datum in traindata:
    yield datum

def validdata_generator():
  for datum in validdata:
    yield datum


# In[27]:


train_dataset=tf.data.Dataset.from_generator(data_generator, output_types=tf.int32)
train_dataset=train_dataset.padded_batch(32,padded_shapes=(None, ),padding_values=tokenizer.pad_token_id )#32개씩 묶어서 배치로 만들어라
train_batch=next(iter(train_dataset))

for train_batch in train_dataset:
    print(train_batch)
    break


# In[28]:


valid_dataset=tf.data.Dataset.from_generator(validdata_generator, output_types=tf.int32)
valid_dataset=valid_dataset.padded_batch(32,padded_shapes=(None, ),padding_values=tokenizer.pad_token_id )#32개씩 묶어서 배치로 만들어라
valid_batch=next(iter(valid_dataset))


# # 5. 미세조정

# In[42]:


total = len(traindata)//32 #batchsize=32
total_val = len(validdata)//32+1 #batchsize=32
opt = tf.keras.optimizers.RMSprop(learning_rate=1e-6, epsilon = 1e-08)
history1=[]
history2=[]

for i in range(10): #epoch:10
  for batch in tqdm.tqdm_notebook(train_dataset,total=total):
    with tf.GradientTape() as tape:
      result=model(train_batch, labels=train_batch)
      loss = result[0] #손실 기록
      mean_loss=tf.reduce_mean(loss) #손실의 평균을 구함 (해당 배치의)
    
    grads=tape.gradient(mean_loss,model.trainable_variables) #훈련시킬수 있는 변수들: 파라미터들의 변수를 업데이트 해라
    opt.apply_gradients(zip(grads,model.trainable_variables))
  print(mean_loss)
  history1.append(mean_loss)

  for batch in tqdm.tqdm_notebook(valid_dataset,total=total_val):
    with tf.GradientTape() as tape:
      result=model(valid_batch, labels=valid_batch,training=False) #training=False 해애함
      loss = result[0] #손실 기록
      mean_val_loss=tf.reduce_mean(loss) #손실의 평균을 구함 (해당 배치의)

  print(mean_val_loss)
  history2.append(mean_val_loss)


# In[75]:


fig, ax = plt.subplots(2,1,figsize=(8,12))

ax[0].plot(history1)
ax[0].set_title('train_loss',fontsize=15)
ax[0].set_xlim([0,9])
ax[0].set_xlabel('epochs')

ax[1].plot(history2)
ax[1].set_title('validation_loss',fontsize=15)
ax[1].set_xlim([0,9])
ax[1].set_xlabel('epochs')
plt.show()

#9번 에포크가 Val_loss를 줄여준다.


# In[ ]:


#에포크 확인 후 재학습


# In[51]:


data = []
for row in train.itertuples():
    print(row)
    tokens = tokenizer.encode(f'<s>Q: {row.Q} A: {row.A}</s>') #Q:question, A:answer
    tokens = [t for t in tokens if t < tokenizer.vocab_size]
    data.append(tokens)


# In[52]:


def data_generator():
  for datum in data:
    yield datum


# In[53]:


dataset=tf.data.Dataset.from_generator(data_generator, output_types=tf.int32)
dataset=dataset.padded_batch(32,padded_shapes=(None, ),padding_values=tokenizer.pad_token_id )#32개씩 묶어서 배치로 만들어라
batch=next(iter(dataset))


# In[54]:


model = TFGPT2LMHeadModel.from_pretrained("/gdrive/My Drive/kogpt2/taeminlee", from_pt=True)
tokenizer = PreTrainedTokenizerFast.from_pretrained("/gdrive/My Drive/kogpt2/taeminlee")


# In[57]:


total = len(data)//32
opt = tf.keras.optimizers.RMSprop(learning_rate=1e-6, epsilon = 1e-08)

for i in range(9): #epoch:10
  for batch in tqdm.tqdm_notebook(dataset,total=total):
    with tf.GradientTape() as tape:
      result=model(batch, labels=batch)
      loss = result[0] #손실 기록
      mean_loss=tf.reduce_mean(loss) #손실의 평균을 구함 (해당 배치의)
    
    grads=tape.gradient(mean_loss,model.trainable_variables) #훈련시킬수 있는 변수들: 파라미터들의 변수를 업데이트 해라
    opt.apply_gradients(zip(grads,model.trainable_variables))
  print(mean_loss)


# In[148]:


model.save_pretrained("/gdrive/My Drive/kogpt2/my_chat_model2")


# In[28]:


model=TFGPT2LMHeadModel.from_pretrained("/gdrive/My Drive/kogpt2/my_chat_model2")


# In[ ]:





# # 6. 평가
# 
# 

# In[90]:


bleu_hist=[]
for text,answer in zip(test_chat.Q,test_chat.A):
  sent = f'<s>Q: {text} A:'
  input_ids = tokenizer.encode(sent, return_tensors='tf')
  output = model.generate(input_ids, max_length=50, do_sample=True,    
  top_k=50, # 확률 순위가 50위 밖인 토큰은 샘플링에서 제외
  top_p=0.95, # 누적 확률이 95%인 후보집합에서만 생성
  num_return_sequences=3, #3개의 결과를 디코딩해낸다 -> 최고의 bleu결과를 뽑아내는애를 도출
  early_stopping=True, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)

  a=answer
  b=tokenizer.decode(output[0]).split('A: ')[1].split('</s>')[0]

  bleu=sentence_bleu(a, b,smoothing_function=SmoothingFunction().method4)
  bleu_hist.append(bleu)
  print('BLEU:',bleu )
print(np.mean(bleu_hist))


# In[61]:


text='나 우울해'
sent = f'<s>Q: {text} A:'
input_ids = tokenizer.encode(sent, return_tensors='tf')
output = model.generate(input_ids, max_length=50, do_sample=True,    
top_k=50, # 확률 순위가 50위 밖인 토큰은 샘플링에서 제외
top_p=0.95, # 누적 확률이 95%인 후보집합에서만 생성
num_return_sequences=3, #3개의 결과를 디코딩해낸다 -> 최고의 bleu결과를 뽑아내는애를 도출
early_stopping=True, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
tokenizer.decode(output[0]).split('A:')[1].split('</s>')[0]


# In[96]:


text='아무도 내편이 없어'
sent = f'<s>Q: {text} A:'
input_ids = tokenizer.encode(sent, return_tensors='tf')
output = model.generate(input_ids, max_length=50, do_sample=True,    
top_k=50, # 확률 순위가 50위 밖인 토큰은 샘플링에서 제외
top_p=0.95, # 누적 확률이 95%인 후보집합에서만 생성
num_return_sequences=3, #3개의 결과를 디코딩해낸다 -> 최고의 bleu결과를 뽑아내는애를 도출
early_stopping=True, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
tokenizer.decode(output[0]).split('A:')[1].split('</s>')[0]


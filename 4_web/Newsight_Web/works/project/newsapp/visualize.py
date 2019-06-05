
import newsapp.newsight as ns

# 시각화
import matplotlib.pyplot as plt
import seaborn as sb

import pandas as pd
import numpy as np # 연산
# 경고 메시지 숨기기
import warnings
warnings.filterwarnings("ignore")

#스트링으로 변환
def to_string(self):
    return self.__str__()


#데이터 불러오기
def read_data():
    df_main = pd.read_csv(rf'C:\Users\student\Desktop\works\project\\newsapp\static\\df_main.csv',encoding = 'utf-8-sig')
    df_final = pd.read_csv(rf'C:\Users\student\Desktop\works\project\\newsapp\static\\df_final.csv',encoding = 'utf-8-sig')

    # df_main['id'].apply(print)
    # print('===============================================================================')
    # df_final['id'].apply(print)

    return df_main, df_final

# 워드클라우드생성
def make_wordcloud(word_list,img_name):
    from collections import Counter
    from wordcloud import WordCloud

    cnt = Counter(word_list)
    tags = cnt.most_common(20)
    
    wc_obj = WordCloud(width = 700, height = 700,
                    font_path = 'data/H2GTRE.TTF',    
                    background_color ='white', 
                    min_font_size = 10)

    cloud = wc_obj.generate_from_frequencies(dict(tags))
    plt.figure(figsize = (5, 5), facecolor = None)
    plt.imshow(cloud) 
    plt.axis("off") 
    plt.tight_layout()
    plt.savefig(rf'C:\Users\student\Desktop\works\project\\newsapp\static\\images\\{img_name}.jpg')

def cat_ts(df_main, df_final, category):
    # {대표 기사의 날짜 : 대표 기사가 속한 토픽의 기사 수}
    n_docs_topic = df_final[df_final['cat_selected'] == category].groupby(['topic'])['title'].count()  # n docs per topic
    main_doc_date = df_main[df_main['cat_selected'] == category].sort_values(by='date', ascending=True)
    main_doc_date['n_docs_topic'] = main_doc_date['topic'].apply(lambda x: n_docs_topic[x])
    main_doc_date = main_doc_date.drop(main_doc_date[main_doc_date['topic'] == -1].index).reset_index(drop=True)  # 토픽 = -1 제거

    #그래프 생성
    plt.figure(figsize=(10, 6))
    plt.plot(main_doc_date['n_docs_topic'])
    x = list(range(main_doc_date.shape[0]))
    labels = main_doc_date['date'].values.tolist()
    plt.xticks(x, labels, rotation=35)
    plt.xlabel('DATE')
    plt.ylabel('NEWS')
    plt.savefig(rf'C:\Users\student\Desktop\works\project\\newsapp\static\\images\\{category}_ts.jpg')

def cat_news(df_main, category):
    represent_df = df_main[['id','date','title']][df_main['cat_selected']==category].sort_values(by='date')
    represent_id = represent_df.id
    represent_date = represent_df.date
    represent_title = represent_df.title

    return represent_id, represent_date, represent_title

def detail_news(df_final, id):
    tmp_id = int(id)
    detail_title = df_final[df_final['id']==tmp_id]['title']
    detail_content= df_final[df_final['id']==tmp_id]['content']

    #동일 topic의 title, id를 pd.Series로 반환
    tmp_cat = df_final[df_final['id']==tmp_id]['cat_selected']
    tmp_cat = tmp_cat.iloc[0]
    tmp_topic = df_final[df_final['id']==tmp_id]['topic']
    tmp_topic = int(tmp_topic)

    topic_title = df_final[df_final['cat_selected']==tmp_cat][df_final['topic']==tmp_topic]['title']
    topic_id = df_final[df_final['cat_selected']==tmp_cat][df_final['topic']==tmp_topic]['id']

    return detail_title, detail_content, topic_title, topic_id


def detail_wc(df_final, id):
    tmp_id = int(id)

    tmp_category = df_final['cat_selected'][df_final['id'] == tmp_id]
    df_cat = df_final[df_final['cat_selected'] == f'{tmp_category.iloc[0]}']

    # print('============df_cat============')
    # print(df_cat)

    #id값을 받아와 해당 토픽 받아옴
    tmp_topic = df_final['topic'][df_final['id'] == tmp_id]
    tmp_topic = int(tmp_topic)
    df_cat_topic = df_cat[df_cat['topic'] == tmp_topic]
    # print(f'df_cat_topic : {df_cat_topic }')
    img_name = f'{tmp_category.iloc[0]}_{tmp_topic}_wc'
    # print(img_name)

    #title 단어들 리스트로 변환
    # print(df_cat_topic['tokenized_title'])
    df_cat_topic['tokenized_title'] = df_cat_topic['tokenized_title'].apply(lambda x : x[1:-1])
    df_cat_topic['tokenized_title'] = df_cat_topic['tokenized_title'].apply(lambda x : x.split(','))
    tk_docs = df_cat_topic['tokenized_title'].values.tolist()
    # print(type(df_cat_topic['tokenized_title'].iloc[0]))
    tmp = []
    for tk in tk_docs:
        tmp.extend(tk)
    #print(tmp)
    #워드클라우드 생성
    make_wordcloud(tmp, img_name)

    return img_name





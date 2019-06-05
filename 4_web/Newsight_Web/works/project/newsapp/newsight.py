

#libraries
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pickle
import datetime
import re
from gensim.summarization import bm25
from ckonlpy.tag import Twitter
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import OPTICS # sklearn 0.21.1 


def readme(): 
    print("******Description*****")
    print("code by 현호킴, description by 승현백")
    print("클래스 이름.help() : 해당 클래스에서 사용할 수 있는 함수 출력")
    print("******Class names******")
    print("1) 데이터 불러오기 : Pickle2DF")
    print("2) 전처리 : PreprocessingText")
    print("3) 불용어,유의어 처리 : GetSimilarWords, GetStopWords")
    print("4) 문서 검색 :  GetDocsFromQuery")
    print("5) 벡터화 :  Vectorizer")
    print("6) 시각화 : Get2DPlot, AnalyzingNewsData, WordCloud")
    print("7) 이상치 제거 : CleaningNoise")
    print("8) 키워드 추출 : GetKeyword")
    print("수정사항은 history 함수를 참고하세효")

def history():
    print("******수정 사항 기록******")
    print("******20190517 백승현******")
    print('''
        1) Plot2D - get_cluster_labels : 
            1-1) 최적 엡실론 산출 기능 추가, optimal_eps = True 면 최적 엡실론 산출
            1-2) Plot2D - get_cluster_labels : cluster method - optics 추가 (sklearn 0.21.1 필요)
    ''')
    print("******20190515 백승현******")
    print('''
    1) GetKeyword - get_news : 각 Topic에서 키워드를 뽑고, BM25가 가장 높은 기사를 뽑는 함수 추가
    2) Plot2D - get_cluster_labels : eps를 input이 아닌 입력인자로 받도록 변경
    3) GetKeyword - select_category : 인덱스가 초기화되는 문제, 데이터 프레임의 인덱스를 초기화해야함(변경 없음, 이슈기록)
    4) GetKeyword - get_news : 인풋,아웃풋을 데이터 프레임 형식으로 수정
    ''')
    print("******20190514 백승현******")
    print(''' 
    1) Get2DPlot - get_cluster_labels
        1-1) 클러스터 내 최소 문서수 5로 고정
        1-2) DBSCAN metric 코사인 유사도로 변경
    2) GetKeyword - get_word_list : word_list.extend(words)로 변경(처리 속도 향상, 기능 변경 없음)
    3) GetKeyword - top_df : self 추가(오류 수정), toknized -> tokenized(오타 수정)
    4) GetDocsFromQuery - select_news : 반환값 문서 순서 => 문서 인덱스 로 수정
    5) PreprocessingText - add_noun_dict : self.noun_list.extend(noun_list) 로 변경(처리 속도 향상, 기능 변경 없음)
    6) PreprocessingText - add_stopwords : self.stopwords.extend(stopword_list) 로 변경(처리 속도 향상, 기능 변경 없음)
    7) GetKeyword - top_tfidf : top_word_list.extend(list(top_word_list))로 변경(처리 속도 향상, 기능 변경 없음)
    8) PreprocessingText - change_similar_words : 유의어 처리 함수 추가
    9) GetKeyword - getCategory : 문서 카테고리 할당 함수 추가 
    10) GetKeyword - remove_na_category(데이터프레임(DataFrame)) : 카테고리가 없는 문서를 제거하는 함수 추가
    11) GetKeyword - select_cat(카테고리(Series)) : 카테고리 리스트중 원소 한개를 선택하는 함수 추가(자세한 내용은 help 문서 참고)
    ''')
    print("******20190510 백승현******")
    print(''' 
    1) Pickle2DF - get_dataframe : 데이터프레임에 category 추가
    2) GetDocsFromQuery - select_news : 
        2-1) title 인자 추가, title = True 면 제목에 query를 포함한 문서 추출, title = False 면 본문에 query를 포함한 문서 추출
        2-2) 입력 파라메터 데이터 프레임으로 수정 : tokenized doc => df
        2-3) 반환값 수정 : 문서 => 문서 인덱스
    3) Get2DPlot - get_2D_vec : 입력인자 변경 - 벡터, 벡터 종류, 차원축소방법
    4) Get2DPlot - get_cluster_labels : 
        4-1) 입력인자 변경 - 클러스터링 방법
        4-2) 클러스터링 방법에 따라 필요한 인자를 input으로 받도록 변경
            ex) kmeans : n_cluster, DBSCAN : eps, 클러스터 내 최소 문서 수
        4-3) 반환값 변경 : 클러스터 라벨
    5) Get2DPlot help() 추가
    ''')


# 1) 데이터 불러오기
class Pickle2DF:
    def help(self):
        print("******Pickle2DF******")
        print("1) get_dataframe(피클 경로(str)) : 피클을 데이터프레임으로 반환")
        print("2) get_dataframe_from_list(피클 경로를 저장한 리스트(list)) : 피클여러개를 데이터프레임으로 반환")
        print("**********************")

    def get_dataframe(self, data_name_with_route):
        with open(data_name_with_route, 'rb') as file:
            data_list = []
            while True:
                try:
                    data = pickle.load(file)
                except EOFError:
                    break
                data_list.append(data)
        # construct lists for data frame
        title = []
        content = []
        date = []
        category = []
        for news in data_list[0]['return_object']['documents']:
            title.append(news['title'])
            content.append(news['content'])
            category.append(news['category'])
            date.append(news['published_at'][:10])  # 시간 조정이 필요하면 바꾸기
        # make lists as data frame
        news_data = pd.DataFrame([])
        news_data['date'] = date
        news_data['title'] = title
        news_data['content'] = content
        news_data['category'] = category
        news_data['date_tmp'] = news_data['date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').toordinal())
        return news_data

    def get_dataframe_from_list(self, data_names):
        news_data_list = []
        for data_name in data_names:
            news_data = self.get_dataframe(data_name)
            news_data_list.append(news_data)
        data = pd.DataFrame([])
        for news_data in news_data_list:
            data = data.append(news_data)
        data.reset_index(inplace=True)
        data.drop(['index'], axis=1, inplace=True)
        return data




# 2) 전처리
class PreprocessingText:
    def help(self):
        print("******PreprocessingText******")
        print("1) make_content_re(df['컬럼이름'](Series)) : 입력받은 열을 전처리 후 시리즈로 반환")
        print("2) add_noun_dict('list') : 명사 사전에 단어 추가")
        print("3) add_stopwords('list') : 불용어 사전에 단어 추가")
        print("4) tokenize(df['컬럼이름'](Series)) : 입력받은 열을 토큰화한 후 시리즈로 반환")
        print("5) change_similar_words(토큰화된 문서(Series), 유의어 사전(dictionary)) : 유의어 사전을 기반으로 문서 내 유의어를 대표어로 변환하고, 변환된 문서를 시리즈로 반환한다.")
        print("*****************************")

    def __init__(self):
        self.reg_reporter = re.compile('[가-힣]+\s[가-힣]*기자')  # 기자
        self.reg_email = re.compile('[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$')  # 이메일
        self.reg_eng = re.compile('[a-z]+')  # 소문자 알파벳, 이메일 제거용, 대문자는 남겨둔다
        self.reg_chi = re.compile("[\u4e00-\u9fff]+")  # 한자
        self.reg_sc = re.compile("·|…|◆+|◇+|▶+|●+|▲+|“|”|‘|’|\"|\'|\(|\)|\W+")  # 특수문자
        self.reg_date = re.compile('\d+일|\d+월|\d+년|\d+시|\d+분|\(현지시간\)|\(현지시각\)|\d+')  # 날짜,시간,숫자
        
        self.twitter_obj = Twitter()
        self.stopwords = []
        self.noun_list = []

    def preprocessing(self, doc):
        tmp = re.sub(self.reg_reporter, '', doc)
        tmp = re.sub(self.reg_email, '', tmp)
        tmp = re.sub(self.reg_eng, '', tmp)
        tmp = re.sub(self.reg_chi, '', tmp)
        tmp = re.sub(self.reg_sc, ' ', tmp)
        tmp = re.sub(self.reg_date, '', tmp)
        return tmp

    def make_content_re(self, data):
        pp_data = data.apply(self.preprocessing)
        return pp_data
    
    def add_noun_dict(self,noun_list):
        self.twitter_obj.add_dictionary(noun_list, 'Noun')
        self.noun_list.extend(noun_list)
        print("추가한 명사")
        print(noun_list)

    def add_stopwords(self,stopword_list):
        self.stopwords.extend(stopword_list)
        print("추가한 불용어")
        print(stopword_list)
        
    def change_similar_words(self, tokenized_docs, similar_words_dict):
        changed_docs = []
        for doc in tokenized_docs :
            changed_doc = []
            for word in doc:
                if word in similar_words_dict.keys():
                    changed_doc.append(similar_words_dict[word])
                else:
                    changed_doc.append(word)
            changed_docs.append(changed_doc)
        return changed_docs

    def tokenize(self, data):
        print('추가한 명사:',self.noun_list)
        print('불용어: ', self.stopwords)
        tokenized_doc = data.apply(lambda x: self.twitter_obj.nouns(x))
        tokenized_doc_without_stopwords = tokenized_doc.apply(
            lambda x: [item.lower() for item in x if item not in self.stopwords])
        tokenized_data = tokenized_doc_without_stopwords
        return pd.Series(tokenized_data)

# 3) 불용어, 유의어 처리
class GetSpecialWords:
    def help(self):
        print("******GetSimilarWords******")
        print("1) get_w2v_model(토큰화된 문서(Series),doc2vec 차원 크기(int)) : doc2vec 모델 학습")
        print("2) get_similar_words(단어(str)) : 유의어 출력")
        print("3) get_bow(토큰화된 문서(Series)) : bow 생성")
        print("4) get_stop_words(단어 출현 빈도 순위(int)) : 단어 출현 빈도 상위 n 개, 하위 n 개 출력")
        print("*****************************")

    def get_w2v_model(self, tokenized_doc, size=300, window=5, min_count=5, workers=4, sg=1):
        self.model = Word2Vec(sentences=tokenized_doc, size=size, window=window, min_count=min_count, workers=workers,
                              sg=sg)
        
    def get_similar_words(self, string):
        print('단어 : 유사도')
        for word, score in self.model.wv.most_similar(string):
            print(word)

    def get_bow(self, sentences):
        self.tmp_list = []
        for doc in sentences:
            self.tmp_list.extend(doc)
        self.word_count = pd.Series(self.tmp_list).value_counts()
        self.word_count_idx = list(self.word_count.index)

    def get_stop_words(self, number):
        stop_words_candi = self.word_count_idx[:number] + self.word_count_idx[-number:]
        for word in stop_words_candi:
            print(word) 
# 4) 문서 검색
class GetDocsFromQuery:
    def help(self):
        print("******GetDocsFromQuery******")
        print("1)set_query(검색어(str)) : 검색어 설정 ")
        print("2)select_news(데이터프레임(DataFrame),title) : 검색어를 포함한 문서를 시리즈로 반환, title = True 면 제목에 쿼리를 포함한 문서 인덱스 반환")
        print("*****************************")

    def set_query(self,query):
        self.query = query

    def select_news(self, df, title = False):
        selected_news = []
        if title == True :
             for idx in df.index.tolist():
                if self.query in df['title'].loc[idx]:
                    selected_news.append(idx)
        else :
            for idx in df.index.tolist():
                if self.query in df['content'].loc[idx]:
                    selected_news.append(idx)
        print(f"length of selected news: {len(selected_news)}")
        print(f"length of original data: {df.shape[0]}")
        return selected_news

# 4) 벡터화
class Vectorizer:
    def help(self):
        print("******Vectorizer******")
        print("1)get_tfidf_vec(토큰화된 문서(Series),단어 수(int)) : 문서를 tfidf 벡터(x) 와 단어(words)로 반환")
        print("2)get_doc2vec(토큰화된 문서(Series)) : doc2vec 벡터 반환")
        print("3)load_doc2vec_model(토큰화된 문서(Series),모델객체(word2vec_obj)): 저장된 모델로  doc2vec 벡터 반환")    
        print("*****************************")

    def get_tfidf_vec(self, query_doc, max_feat=None, min_df = 0, max_df = 1.0):
        query_doc = query_doc.apply(lambda x: ' '.join(x))
        obj = TfidfVectorizer(max_features=max_feat, min_df = min_df, max_df = max_df)  # max_features for lda
        x = obj.fit_transform(query_doc).toarray()
        words = np.array(obj.get_feature_names())
        return x, words

    def get_doc2vec(self, query_doc,
                    dm=1, dbow_words=1, window=8, vector_size=50, alpha=0.025,
                    seed=42, min_count=5, min_alpha=0.025, workers=4, hs=0, negative=10,
                    n_epochs=50, model_name='d2v.model'):
        tagged_data = [TaggedDocument(words=_d, tags=[str(i)]) for i, _d in enumerate(query_doc)]
        model = Doc2Vec(
            dm=dm,  # PV-DBOW => 0 / default 1
            dbow_words=dbow_words,  # w2v simultaneous with DBOW d2v / default 0
            window=window,  # distance between the predicted word and context words
            vector_size=vector_size,  # vector size
            alpha=alpha,  # learning-rate
            seed=seed,
            min_count=min_count,  # ignore with freq lower
            min_alpha=min_alpha,  # min learning-rate
            workers=workers,  # multi cpu
            hs=hs,  # hierarchical softmax / default 0
            negative=negative,  # negative sampling / default 5
        )
        model.build_vocab(tagged_data)
        print("corpus_count: ", model.corpus_count)

        for epoch in range(n_epochs):
            if epoch % 10 == 0:
                print('epoch: ', epoch)
            model.train(tagged_data,
                        total_examples=model.corpus_count,
                        epochs=model.epochs)
            model.alpha -= 0.0002  # decrease the learning rate
            model.min_alpha = model.alpha  # fix the learning rate, no decay
        model.save(model_name)
        print("Model Saved")
        model_loaded = Doc2Vec.load(model_name)
        print("Load Model")
        x_doc2vec = []
        for i in range(len(query_doc)):
            x_doc2vec.append(model_loaded.docvecs[i])
        x_doc2vec = np.array(x_doc2vec)
        return x_doc2vec

    def load_doc2vec_model(self, query_doc, model_name):
        print("Load Model")
        model_loaded = Doc2Vec.load(model_name)
        x_doc2vec = []
        for i in range(len(query_doc)):
            x_doc2vec.append(model_loaded.docvecs[i])
        x_doc2vec = np.array(x_doc2vec)
        return x_doc2vec

# 5) 시각화
class Get2DPlot:

    def help(self):
        print("******Get2DPlot******")
        print("1)get_2D_vec(벡터(ndarray),벡터 종류(string), 차원축소 방법(string)) : ")
        print("2 차원으로 차원축소된 벡터를 반환, 벡터 종류 = (tfidf, doc2vec) 차원축소 방법 = (TSNE,PCA)")
        print("2)get_cluster_labels(클러스터링 방법(string), min_samples(int), min_range(int), optimal_eps(boolean), eps(float)")
        print('''
            1)에서 받은 벡터를 군집화하고, 라벨 리스트를 반환, 
            optimal_eps = True 면 엡실론 최적화 함수 실행
            min_sample 은 DBSCAN min_sample과 동일, min_range는 평균 변화율 계산 구간의 너비
            ''')
        print("클러스터링 방법은 kmeans, DBSCAN 중 하나 선택. kmeans를 선택할 경우 클러스터 개수를 입력, DBSCAN 은 eps 입력")
        print("3)plot2D(): 2)에서 실행한 군집화 결과를 2차원으로 시각화")    
        print("*****************************")

    def __init__(self,learning_rate=200, random_state=10):
        self.learning_rate = learning_rate
        self.random_state = random_state

    def get_2D_vec(self, x, vec_kind = 'tfidf', reduction_method = 'TSNE'):
        self.reduction_method = reduction_method
        if vec_kind == 'tfidf' : self.x_scaled = x
        elif vec_kind == 'doc2vec' : self.x_scaled = StandardScaler().fit_transform(x)
        else:
            print('vec_kind 는 tfidf 혹은 doc2vec 만 가능하다. 이상한거 넣지 말긔')
            raise NotImplementedError 
        
        if self.reduction_method == 'TSNE':
            t_sne = TSNE(n_components=2, learning_rate=self.learning_rate, init='pca',
                         random_state=self.random_state)
            self.vec = t_sne.fit_transform(self.x_scaled)
        elif self.reduction_method == 'PCA':
            pca = PCA(n_components=2)
            self.vec = pca.fit_transform(self.x_scaled)
        return self.vec

    def get_best_eps(self, vector, min_samples=5, min_range = 4):        
        l = len(vector)
        nn = NearestNeighbors(n_neighbors=min_samples, metric='cosine').fit(vector)
        
        distances, indices = nn.kneighbors(vector)
        min_samples -= 1
        candi = sorted(distances[:, min_samples])
        rate1 = []
        for i in range(min_range, l):
            dy1 = candi[i - 1] - candi[i - min_range] 
            dy2 = candi[i] - candi[i - (min_range -1)]
            e = 10**(-6)
            rate1.append((dy2+e) / (dy1+e))
        index = rate1.index(max(rate1))
        best_eps = candi[index]
        #plt.figure(figsize = (10,5))
        #plt.plot(candi)
        #plt.axhline(y = best_eps, color = 'red')
        return best_eps

    def get_cluster_labels(self, optimal_eps = False, min_samples = 5, min_range = 4, eps =0.5, cluster_method = 'kmeans'):
        if optimal_eps == True:
            self.eps = self.get_best_eps(self.x_scaled, min_samples=min_samples, min_range = min_range)
        else : self.eps = eps
        self.cluster_method = cluster_method
          
        if self.cluster_method == 'kmeans':
            print('클러스터 개수를 입력하세요 :')
            self.n_clusters = int(input())
            cluster = KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit(self.x_scaled)
            self.cluster_labels = cluster.labels_
 
        elif self.cluster_method == 'DBSCAN':
            self.cluster_labels = DBSCAN(eps=self.eps, min_samples = min_samples, metric = 'cosine').fit_predict(self.x_scaled)
        
        elif self.cluster_method == 'OPTICS':
            opt = OPTICS(min_samples= min_samples, max_eps=0.9, metric='cosine')
            opt.fit(self.x_scaled)
            self.cluster_labels = opt.labels_           
        else :
            print('cluster method는  kmeans, DBSCAN, OPTICS 중에서만 골라주세용')
            raise NotImplementedError 
            
        vec_pd = np.c_[self.vec,self.cluster_labels]
        self.vec_pd = pd.DataFrame(vec_pd, columns=['x', 'y', 'labels'])
        
        return self.cluster_labels

    def plot2D(self):
        print(self.reduction_method, self.cluster_method)
        groups = self.vec_pd.groupby('labels')
        fig, ax = plt.subplots()
        for name, group in groups:
            ax.plot(group.x,
                    group.y,
                    marker='o',
                    linestyle='',
                    label=name)
        # ax.legend(fontsize=12, loc='upper left') # legend position

        plt.title('%s Plot of %s' % (self.reduction_method, self.cluster_method), fontsize=20)
        plt.xlabel('x', fontsize=14)
        plt.ylabel('y', fontsize=14)
        plt.show() 

class AnalyzingNewsData:
    def help(self):
        print("******AnalyzingNewsData******")
        print("1)get_n_data_per_cluster(라벨 리스트(list)) : 클러스터 별 문서 개수 반환")
        print("2)print_news_per_cluster(토큰화된 문서(Series),라벨 리스트(list)) : 클러스터 라벨별로 제목을 출력한다.")
        print("*****************************")


    def get_n_data_per_cluster(self, cluster_labels):
        tmp = pd.DataFrame(pd.Series(cluster_labels).value_counts(), columns=['counts'])
        tmp.index.name = 'cluster'
        return tmp

    def print_news_per_cluster(self, data, clusters, content='title'):
        for n in clusters:
            print('=' * 100, n)
            for i in range(len(data)):
                try:
                    print(data[data['labels'] == n][content].iloc[i])
                    print('*' * 100)
                except:
                    break
            print('=' * 100, n)
        return 0
    
class CleaningNoise:
    def help(self):
        print("******CleaningNoise******")
        print("1)count_topic(토큰화된문서(Series),단어(str)) : 각 문서당 특정 단어가 몇번 포함됐는지를 리스트로 출력")
        print("2)plot_hist(count_topic 에서 반환된 카운트리스트(list)) : 1 에서 반환된 리스트로 히스토그램을 그린다")
        print("3)get_proper_news(토큰화된문서(Series), count_topic 에서 반환된 카운트리스트(list), 기준치(int): 문서 내 특정 단어가 기준치 이상 나온 문서만 반환한다.")
        print("*****************************") 

    def count_topic(self, tokenized_doc, topic):
        counting = []
        for news in tokenized_doc:
            counting.append(news.count(topic))
        return counting

    def plot_hist(self,counting):
        counting_pd = pd.Series(counting)
        bins = len(counting_pd.value_counts())
        counting_pd.hist(figsize=(16, 12), bins=bins+20)
        plt.xlabel("counting", fontsize=14)
        plt.show()

    def get_proper_news(self, data, counting, threshold):
        select = np.array(counting) >= threshold
        print('number of total data:', len(data))
        print('number of proper data:', sum(select))
        return data[select].copy()

class GetKeyword:
        def help(self):
            print("******GetKeyword******")
            print("1)top_df(토큰화된 문서(Series)) : 문서에 포함된 단어의 DF를 계산하고, 단어에 따른 DF를 데이터프레임으로 반환한다.")
            print("2)top_tfidf(tfidf 벡터(np.ndarray), tfidf 벡터의 features(np.array), top_n(int)) : 각 문서별 tfidf가 가장 큰 단어 3개를 뽑고, 그 단어들의 빈도를 시리즈로 반환한다.")
            print("3)lda(tfidf 벡터(np.ndarray),tfidf 벡터의 features(np.array), 토픽수(int), 토픽당 출력할 단어 수(int)) : 토픽에 따라 선정된 키워드를 출력한다. 각 문서의 토픽 번호를 리스트로 반환한다.")
            print("4)get_issues_based_dataframe(self, 데이터프레임(dataframe), 주제별 키워드 리스트(list), 주제당 뽑고 싶은 기사 수(int) , 컬럼이름(str)")
            print("주제별 키워드 형식 : [[k11,k12,k12...],[k21,k22,...],...,[kn1,kn2]]")
            print("주제당 가장 bm25 score가 높은 기사를 뽑아서 데이터 프레임으로 반환한다")
            print("5)remove_na_category(데이터프레임(DataFrame)) : 카테고리가 있는 문서만 반환한다. 카테고리가 없는 문서를 제거 전/후 문서의 수를 출력한다.")
            print("6)select_category(카테고리(Series)) :카테고리 리스트 내에서 원소 한개를 선택해 반환한다.")
            print('''
            카테고리 원소 형식 : [cat_1>cat_2, cat_1>cat_2, cat_1>cat_2],[...],[...],...
            6-1) cat_1>cat2 에서 cat2 를 제거하고, 중복된 cat_1을 제거한다.
            6-2) cat_1의 빈도 분포를 출력한다.
            6-3) cat_1에서 가장 빈도가 높은 카테고리를 major_cat, 나머지를 빈도 순서대로 minor_cat에 저장한다.
            6-4) cat_list 는 category(input)의 원소일때,
                6-4-1) cat_list 의 길이가 1 이면 cat_list[0]을 카테고리로 선정
                6-4-2) cat_list 의 길이가 2 이상일 때
                    6-4-2-1) minor_cat 이 존재하지 않는다면 major_cat을 카테고리로 선정
                    6-4-2-2) minor_cat 이 존재한다면 minor_cat_list에서 가장 뒤에 있는 원소를 카테고리로 선정
            6-5) 선택된 카테고리 리스트를 시리즈로 반환한다.
            ''')
            print("7)get_news(이슈별로 분류된 데이터[cat[df,df..]], max_feat, min_df, max_df, n_keyword): 토픽 별 키워드를 추출하고, BM25가 높은 문서를 반환한다")
            print("*************************")

        def __init__(self):
            self.index = []
        
        def get_word_list(self, toknized_doc,make_set = True):
            word_list = []
            for words in toknized_doc:
                word_list = word_list.extent(words)
            if make_set == True : return list(set(word_list))
            else : return word_list
        
        def get_df(self, docs,word):
            n_docs = len(docs)
            cnt = 0
            for doc in docs:
                if word in doc:
                    cnt=cnt+1
                    continue
            df = cnt/n_docs
            return df
        
        def top_df(self, tokenized_doc):

            df_list = []
            n_docs = len(tokenized_doc)
            word_list = self.get_word_list(tokenized_doc,make_set = True)
            for word in word_list:
                df_list.append(self.get_df(tokenized_doc, word))
            return pd.DataFrame({'word':word_list,'df':df_list}).sort_values(by='df',ascending = False)
        
        def get_ranked_idx(self, x,top_n):
            ranks = (-x).argsort()
            top_n_idx = ranks[:top_n]
            return top_n_idx
           
        def idx2word(self, word,idx) : return(word[idx])
        
        def top_tfidf(self, tfidf_vec, word, top_n):
            top_n_idx = np.apply_along_axis(get_ranked_idx,1,tfidf_vec)
            top_n_word = np.apply_along_axis(lambda row : idx2word(word,row),1,top_n_idx)
            top_word_list = []
            for top_word in top_n_word:
                top_word_list.extend(list(top_word_list))
            return pd.Series(top_word_list).value_counts().sort_values(ascending = False)

        def lda(self, x, word, n_topics=5, no_top_words = 5, max_iter=100, learning_method='online', learning_offset=40, verbose = 1):
            lda_model = LDA(n_topics = n_topics, max_iter = max_iter, learning_method = learning_method, learning_offset = learning_offset, verbose = verbose).fit(x)
            x_lda = lda_model.transform(x) # 문서당 토픽 확률을 담은 배열

            print("*****주제별 키워드*******")
            ## 토픽 출력
            for topic_idx, topic in enumerate(lda_model.components_):
                print ("Topic %d:" % (topic_idx))
                print (" ".join([word[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

            lda_label = []
            for i in range(x.shape[0]):
                lda_label.append(x_lda[i,:])

            return lda_label

        def issue_based_search(self, tokenized_doc, issue, search, length):
            """
            :param tokenized_doc: 토큰화된 문서
            :param issue: 이슈 단어 리스트
            :param lenght: 뽑고 싶은 기사 수
            :return: 이슈와 관련된 문서의 인덱스를 출력
            """
            bm25_scores = search.get_scores(issue)
            tmp = []
            for i, score in enumerate(bm25_scores):
                tmp.append([i, score])
                sorted_bm25_scores = sorted(tmp, key=lambda x: x[1], reverse=True)
            selected = []
            for i, score in sorted_bm25_scores:
                if i not in self.index:
                    self.index.append(i)
                    selected.append(i)
                if len(selected) >= min([length, search.corpus_size]):
                    break
            selected = list(tokenized_doc.iloc[selected].index)
            return selected

        def get_issues_based_dataframe(self, data, issues, length=10, tokenized_column_name='tokenized_doc'):
            """
            :param data: 데이터(데이터프레임)
            :param issues: 이슈들
            :param length: 뽑고 싶은 기사수
            :param tokenized_column_name: 토큰화된 문서의 컬럼
            :return: 이슈 별 데이터프레임
            """
            search = bm25.BM25(data[tokenized_column_name])
            tmp_data = data.copy()
            issues_based_dataframe = []
            for issue in issues:
                selected = self.issue_based_search(tmp_data[tokenized_column_name], issue, search, length)
                issues_based_dataframe.append(tmp_data.loc[selected].copy())
            return issues_based_dataframe
        
        # 카테고리가 없는 문서를 제거한다.
        def remove_na_category(self,df):

            idx_na_cat = df[df['category'].apply(lambda x : 1 if x == [''] else 0)==1].index

            df_cat = df.drop(idx_na_cat).copy()
            print(f'n_docs : {df.shape[0]}')
            print(f'n_docs after remove na categories : {df_cat.shape[0]}')
            return df_cat
 

        def select_category(self,category):          
            # cat_2 제거, 중복된 cat_1 제거
            cat_1 = []
            cat_tmp = []
            for cat_list in category :
                tmp = []
                for cat in cat_list :
                    tmp.append(re.sub(r'\>\w+','',cat))
                cat_1.extend(list(set(tmp)))
                cat_tmp.append(list(set(tmp)))

            # 카테고리 원소 빈도 출력
            cat_1_table = pd.Series(cat_1).value_counts()
            # print(cat_1)
            # print(cat_tmp)
           # print(cat_1_table)
            print(len(cat_tmp))

            tmp = cat_1_table.index.tolist()
            major_cat = tmp[0]
            minor_cat_list = tmp[1:]

            print(f'major category : {major_cat}')
            print(f'minor categories : {minor_cat_list}')

            # 카테고리 원소 선택 
            selected_cat = []
            for cat in cat_tmp : 
                if(len(cat)==1) : selected_cat.append(cat[0])
                else :
                    for minor_cat in minor_cat_list:
                        if minor_cat in cat : 
                            selected_cat.append(minor_cat)
                            break
                        else : 
                            selected_cat.append(major_cat)
                            break
            return pd.Series(selected_cat)
                  
        def get_news(self,df, max_feat=1000, min_df=0.1, max_df=0.9, n_keyword=5):
            
            main_docs = pd.DataFrame()
            cat_list = df['cat_selected'].unique().tolist()
            
            for cat in cat_list:
                for topic in df[df['cat_selected'] == cat]['topic'].unique():

                    tmp = df[(df['cat_selected'] == cat) & (df['topic'] == topic)].copy() # topic 내 기사 가져오기
                    
                    vectorizer = Vectorizer()
                    x, words = vectorizer.get_tfidf_vec(tmp['tokenized_doc'])
                    
                    # 대표 벡터 추출
                    svd_model = TruncatedSVD(n_components=1, algorithm= 'randomized', n_iter = 100, random_state=123)
                    svd_model.fit(x)
                    
                    keyword_extractor = GetKeyword()
                    
                    keyword = []
                    keyword_score = svd_model.components_[0] 
                    
                    for i in keyword_score.argsort()[:-10:-1]:
                        keyword.append(words[i])
                    keyword_extractor.index = []
                    
                    main_doc = keyword_extractor.get_issues_based_dataframe(tmp,keyword,length = 1)[0]
                    main_docs = pd.concat([main_docs,main_doc])
                    
            return main_docs
            
                

def workflow_1(query, df_raw) :

    print('*****시작*****')
    print(f'query : {query}')
    import newsapp.newsight as ns
    import pandas as pd
    import re # 정규표현식
    import warnings # 경고 메시지 숨기기
    warnings.filterwarnings("ignore")

    print('*****카테고리, 문서, 제목 리스트로 바꾸기*****')
    df_raw['category'] = df_raw['category'].apply(lambda x : x.split(','))
    df_raw['tokenized_doc'] = df_raw['tokenized_doc'].apply(lambda x : x.split(','))
    df_raw['tokenized_title'] = df_raw['tokenized_title'].apply(lambda x : x.split(','))


    print('*****유효문서 추출 시작*****')
    selector = ns.GetDocsFromQuery() # 유효문서 추출 A
    pp = ns.PreprocessingText() # 전처리
    cleaner = ns.CleaningNoise() # 유효문서 추출 B

    # * 유효문서 추출 A
    # 제목에 검색어를 포함하는 문서만 추출
    # 제목에 '[\w+]'를 포함하는 문서 제외
    # 제목에 검색어를 포함하는 문서 인덱스 추출

    selector.set_query(query)
    idx_proper_title = selector.select_news(df_raw, title = True)
    df_proper_title = df_raw.loc[idx_proper_title]

    # [ET투자뉴스],[마켓인사이트] 등의 기사 제거
    # [단독], [속보], [2보] 등은 남겨야함
    idx_special_title = []
    regex = "\[\w{4,20}\]"
    for i in df_proper_title.index:
        if re.search(regex,df_proper_title.loc[i]['title']) is not None: idx_special_title.append(i)

    df_proper_title.drop(idx_special_title,inplace = True)

    query_counter = cleaner.count_topic(df_proper_title['tokenized_doc'],query)
    # print(query_counter)
    proper_idx = cleaner.get_proper_news(df_proper_title['tokenized_doc'],query_counter, 3).index.tolist()
    # print(len(proper_idx))
    df_proper = df_proper_title.loc[proper_idx].copy()

    # ## 문서별 카테고리 할당

    keyword_extractor = ns.GetKeyword() # 카테고리 할당 모듈

    print('*****카테고리 할당 시작*****')

    df_cat_proper = keyword_extractor.remove_na_category(df_proper).reset_index(drop=True) #인덱스 초기화 해야함
    cat_selected = keyword_extractor.select_category(df_cat_proper['category'])
    df_cat_proper['cat_selected'] = cat_selected

    threshold = 5
    rm_cat_list = df_cat_proper['cat_selected'].value_counts()[df_cat_proper['cat_selected'].value_counts()<threshold].index.tolist()
    for rm_cat in rm_cat_list:
        idx = df_cat_proper[df_cat_proper['cat_selected']==rm_cat].index.tolist()
        df_cat_proper.drop(idx,inplace = True)

    print('*****비지도 학습 시작****')
    vectorizer = ns.Vectorizer()
    cluster = ns.Get2DPlot()

    #  카테고리별 단어리스트와 TFIDF를 딕셔너리로 저장
    x_cat_dict = {} # 카테고리별 tfidf 벡터
    word_cat_dict = {} # 카테고리별 단어
    cat_list = df_cat_proper['cat_selected'].unique().tolist() # 카테고리 리스트

    for cat in cat_list:
        vec_obj = ns.Vectorizer()
        cat_docs = df_cat_proper[df_cat_proper['cat_selected'] == cat]['tokenized_doc']

        x_cat,word_cat = vec_obj.get_tfidf_vec(cat_docs)

        x_cat_dict[cat] = x_cat
        word_cat_dict[cat] = word_cat

    cluster_obj = ns.Get2DPlot()
    topic_label_dict = {}
    for category, tfidf_vec in x_cat_dict.items():
        cluster_obj = ns.Get2DPlot()

        print(f'****{category}****')

        vec_2d = cluster_obj.get_2D_vec(tfidf_vec,'tfidf','PCA')
        topic_label = cluster_obj.get_cluster_labels(True,cluster_method= 'OPTICS')
        #cluster_obj.plot2D()

        topic_label_dict[category] = topic_label

    # 카테고리별 토픽할당
    df_final = pd.DataFrame()
    for cat in cat_list:
        df_tmp = df_cat_proper[df_cat_proper['cat_selected'] == cat].copy()
        df_tmp['topic'] = topic_label_dict[cat]
        df_final = pd.concat([df_final,df_tmp])

    print('*****대표기사 추출*****')
    # ## 토픽 별 대표기사 추출
    df_main = keyword_extractor.get_news(df_final)
    print('*****끗*****')
    return (df_main, df_final)





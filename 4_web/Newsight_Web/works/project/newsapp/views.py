from django.shortcuts import render
from django.http import HttpResponse
from collections import OrderedDict
from .models import TotalDb
from django.views.decorators.csrf import csrf_protect
from .workflow1 import workflow_1
import pandas as pd
import newsapp.newsight as ns
import newsapp.visualize as vs
import matplotlib.pyplot as plt

@csrf_protect

def main(request):
    return render(request,'newsapp/main.html')

def query_result(request):
    data = TotalDb.objects.all()
    query = request.GET.get('q', '')
    if query:
        data_query = data.filter(query= query)
        data_selected = data_query.filter()

        data = pd.DataFrame(list(data_selected.values()))
        df_main, df_final = workflow_1(query, data)

        csv_path = (rf'C:\Users\student\Desktop\works\project\\newsapp\static\\')
        df_main.to_csv(csv_path + 'df_main.csv')
        df_final.to_csv(csv_path +  'df_final.csv')

        #카테고리 워드클라우드생성
        cat_list = df_final['cat_selected'].values.tolist()
        tmp_ab_cat_list = list(set(cat_list))
        cat_vol = []
        for i in range(len(tmp_ab_cat_list)):
            cat_vol.append(cat_list.count(tmp_ab_cat_list[i]))
        tmp_df = pd.DataFrame(data=cat_vol,index=tmp_ab_cat_list)
        final_list = tmp_df.sort_values(by=0, ascending=False).index
        vs.make_wordcloud(cat_list, 'cat_wc') #(데이터,파일명)

        #전체 시계열 그래프생성
        n_docs_date = df_final.groupby(['date'])['title'].count()
        plt.figure(figsize=(10, 6))
        n_docs_date.plot()
        plt.xticks(rotation=35)
        plt.xlabel('DATE')
        plt.ylabel('NEWS')
        plt.savefig(csv_path + '\images\\total_ts.jpg')

        return render(request, 'newsapp/category.html',{"query":query,"cat_list":final_list})
    return render(request, 'newsapp/main.html')

def category_result(request):
    # url c='value' 값 가져오기
    cat = request.GET.get('c', '')
    cat = vs.to_string(cat)

    # df_main, df_final csv파일 가져오기
    main, final = vs.read_data()

    # 카테고리 시계열 그래프 생성
    vs.cat_ts(main, final, cat)

    #대표기사 추출
    #csv_path = (rf'C:\Users\student\Desktop\works\project\\newsapp\static\\')
    id, date, title = vs.cat_news(main, cat)

    cat_df = zip(id, date, title)
    return render(request, 'newsapp/topic.html', {"cat":cat, "cat_df":cat_df})


def detail_news_result(request):
    # url id='value' 값 가져오기
    id = request.GET.get('id', '')
    id = vs.to_string(id)

    # df_final 데이터 받아오기
    main, final = vs.read_data()
    # 워드클라우드 생성
    img_name = vs.detail_wc(final, id)
    # id 값으로 detail news의 title, content 받아오기
    title, content, topic_title, topic_id = vs.detail_news(final, id)
    detail_df = zip(title, content)
    topic_df = zip(topic_title,topic_id)
    return render(request, 'newsapp/contents.html', {"img_name":img_name,"detail_df":detail_df,"topic_df":topic_df})

# def test(request):
#     candidates = TotalDb.objects.all()
#     content = candidates[0]
#     context = {'candidates': content}
#     return render(request, 'newsapp/main.html', context)

def test(request):
    return render(request, 'newsapp/test.html')







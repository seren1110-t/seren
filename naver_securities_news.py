# -*- coding: utf-8 -*-
import requests
from lxml import html
from datetime import datetime
import time
import pandas as pd
import re
import os
import pickle

from dotenv import load_dotenv  # ✅ 추가
load_dotenv()  # ✅ 환경 변수 로드

import openai  # ✅ dotenv 이후에 위치해야 함
openai.api_key = os.getenv("OPENAI_API_KEY")  # ✅ 바로 할당

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from pydantic import Field

HEADERS = {"User-Agent": "Mozilla/5.0"}

# ========== 1. 뉴스 크롤링 ==========

def get_last_page(date_str):
    url = f"https://finance.naver.com/news/mainnews.naver?date={date_str}"
    try:
        response = requests.get(url, headers=HEADERS, timeout=5)
        response.encoding = 'euc-kr'
        tree = html.fromstring(response.text)
        last_page_link = tree.xpath('//td[@class="pgRR"]/a')
        if last_page_link:
            match = re.search(r'page=(\d+)', last_page_link[0].get('href'))
            if match:
                return int(match.group(1))
    except:
        pass
    return 1

def get_news_list_on_page(date_str, page):
    url = f"https://finance.naver.com/news/mainnews.naver?date={date_str}&page={page}"
    try:
        response = requests.get(url, headers=HEADERS, timeout=5)
        response.encoding = 'euc-kr'
        tree = html.fromstring(response.text)
        news_items = tree.xpath('//*[@id="contentarea_left"]/div[2]/ul/li')
    except:
        return []

    news_list = []
    for item in news_items:
        try:
            title_a = item.xpath('./dl/dd[1]/a')
            wdate = item.xpath('.//span[@class="wdate"]/text()')
            if not title_a or not wdate:
                continue
            title = title_a[0].text.strip()
            href = "https://finance.naver.com" + title_a[0].get('href')
            news_datetime = datetime.strptime(wdate[0], "%Y-%m-%d %H:%M:%S")
            news_list.append({
                "title": title,
                "url": href,
                "datetime": news_datetime
            })
        except:
            continue
    return news_list

def get_news_body(news_url):
    try:
        response = requests.get(news_url, headers=HEADERS, timeout=5)
        response.encoding = 'euc-kr'
        if "top.location.href" in response.text:
            redirected_url = re.search(r"top\.location\.href='(.*?)'", response.text)
            if redirected_url:
                news_url = redirected_url.group(1)
                response = requests.get(news_url, headers=HEADERS, timeout=5)
                response.encoding = 'utf-8'
        tree = html.fromstring(response.text)
        xpath_id = tree.xpath('//*[@id="newsct_article"]')
        if xpath_id:
            return xpath_id[0].text_content().strip()
        xpath_class = tree.xpath('//*[contains(@class, "newsct_article_article_body")]')
        if xpath_class:
            return xpath_class[0].text_content().strip()
    except:
        return None
    return None

def scrape_news_until_cutoff_today(cutoff_hour=9):
    today_str = datetime.today().strftime("%Y-%m-%d")
    cutoff_time = datetime.strptime(f"{today_str} {cutoff_hour:02}:00:00", "%Y-%m-%d %H:%M:%S")
    last_page = get_last_page(today_str)
    all_news = []

    for page in range(last_page, 0, -1):
        news_list = get_news_list_on_page(today_str, page)
        if not news_list:
            continue

        early_news_found = False

        for news in news_list:
            if news["datetime"] > cutoff_time:
                continue

            early_news_found = True
            body = get_news_body(news["url"])
            if not body or len(body) < 100:
                continue

            all_news.append({
                "제목": news["title"],
                "URL": news["url"],
                "날짜": news["datetime"].strftime("%Y-%m-%d %H:%M:%S"),
                "본문": body
            })
            time.sleep(0.5)

        if early_news_found:
            continue

        if all(news["datetime"] > cutoff_time for news in news_list):
            break

    return all_news

def clean_korean_text(text):
    if pd.isnull(text):
        return ''
    text = re.sub(r'[\n\t\r]', ' ', text)
    text = re.sub(r'[^\uAC00-\uD7A3a-zA-Z0-9 .,?!]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ========== 2. LLM 확장 ==========

class CostTrackingChatOpenAI(ChatOpenAI):
    call_count: int = Field(default=0)
    total_tokens: int = Field(default=0)

    def generate(self, messages, **kwargs):
        response = super().generate(messages, **kwargs)
        usage = response.llm_output.get("token_usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        self.call_count += 1
        self.total_tokens += prompt_tokens + completion_tokens
        print(f"API 호출 횟수: {self.call_count}, 이번 호출 토큰: {prompt_tokens + completion_tokens}, 누적 토큰: {self.total_tokens}")
        return response


# ========== 3. 실행 메인 ==========

def main():
    news_data = scrape_news_until_cutoff_today(cutoff_hour=9)
    df = pd.DataFrame(news_data)
    df['제목'] = df['제목'].apply(clean_korean_text)
    df['본문'] = df['본문'].apply(clean_korean_text)
    df.dropna(subset=['제목', '본문'], inplace=True)
    df['내용'] = df['제목'] + '\n' + df['본문']
    df['내용'] = df['내용'].str.slice(0, 500)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': device}  # 'cpu'로 변경 가능
    )

    with open("bk_docs.pkl", "rb") as f:
        bk_docs = pickle.load(f)

    bk_faiss_db = FAISS.load_local("bk_faiss_index", embedding_model, allow_dangerous_deserialization=True)

    new_docs = []
    for idx, row in df.iterrows():
        metadata = {"일자": row["날짜"], "제목": row["제목"], "URL": row["URL"]}
        new_docs.append(Document(page_content=row["내용"], metadata=metadata))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=100)
    new_split_docs = text_splitter.split_documents(new_docs)
    bk_docs.extend(new_split_docs)

    bm25_retriever = BM25Retriever.from_documents(bk_docs)
    bm25_retriever.k = 5

    bk_faiss_db.add_documents(new_split_docs)
    faiss_retriever = bk_faiss_db.as_retriever(search_kwargs={"k": 5})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.6, 0.4],
        top_k=5
    )

    

    llm = CostTrackingChatOpenAI(
        model_name="gpt-4o",
        api_key=openai.api_key,
        temperature=0
    )

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=ensemble_retriever)

    with open("bk_docs.pkl", "wb") as f:
        pickle.dump(bk_docs, f)

    bk_faiss_db.save_local("bk_faiss_index")
    print("뉴스 업데이트 및 FAISS 인덱스 저장 완료!")

if __name__ == "__main__":
    main()

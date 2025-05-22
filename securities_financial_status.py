import asyncio
import aiohttp
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time
import sqlite3
from pykrx import stock
from datetime import timedelta

# ------------------- KOSPI 티커 수집 --------------------
def get_kospi_tickers():
    stocks = []
    headers = {'User-Agent': 'Mozilla/5.0'}
    for page in range(1, 21):
        url = f"https://finance.naver.com/sise/sise_market_sum.naver?sosok=0&page={page}"
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, "html.parser")
        rows = soup.select("table.type_2 tr")

        for row in rows:
            a_tag = row.select_one("a.tltle")
            if a_tag:
                name = a_tag.text.strip()
                href = a_tag['href']
                code = href.split('code=')[-1]
                stocks.append({"종목명": name, "종목코드": code})
    return pd.DataFrame(stocks)


# ------------------- 비동기 재무 정보 수집 --------------------
async def get_stock_report_async(code, session, semaphore):
    url = f"https://finance.naver.com/item/main.naver?code={code}"
    today = datetime.now().strftime("%Y-%m-%d")

    async with semaphore:
        try:
            async with session.get(url) as res:
                html = await res.text()
        except Exception as e:
            print(f"[{code}] 요청 오류: {e}")
            return None

    soup = BeautifulSoup(html, "html.parser")

    try:
        name = soup.select_one("div.wrap_company h2 a").text.strip()
        current_price = soup.select_one("p.no_today .blind").text.strip().replace(',', '')
    except:
        name = None
        current_price = None

    roe_최근, debt_최근, reserve_최근, per_최근, pbr_최근 = [None]*5
    roe_전기, debt_전기, reserve_전기, per_전기, pbr_전기 = [None]*5
    sales_최근, operating_최근, net_income_최근 = [None]*3
    sales_전기, operating_전기, net_income_전기 = [None]*3

    try:
        table = soup.select_one("#content > div.section.cop_analysis > div.sub_section > table")
        rows = table.select("tbody tr")

        def get_text(row_idx, td_idx):
            try:
                return rows[row_idx].select("td")[td_idx].text.strip()
            except:
                return None

        roe_최근, roe_전기 = get_text(5, 8), get_text(5, 7)
        debt_최근, debt_전기 = get_text(6, 8), get_text(6, 7)
        reserve_최근, reserve_전기 = get_text(8, 8), get_text(8, 7)
        per_최근, per_전기 = get_text(10, 8), get_text(10, 7)
        pbr_최근, pbr_전기 = get_text(12, 8), get_text(12, 7)

        sales_최근, sales_전기 = get_text(0, 8), get_text(0, 7)
        operating_최근, operating_전기 = get_text(1, 8), get_text(1, 7)
        net_income_최근, net_income_전기 = get_text(2, 8), get_text(2, 7)
    except Exception as e:
        print(f"[{code}] 재무지표 오류: {e}")

    # 뉴스 추출
    latest_news = []
    try:
        base_url = "https://finance.naver.com"
        news_items = soup.select("#content > div.section.new_bbs > div.sub_section.news_section > ul:nth-child(2) > li > span > a")
        for item in news_items[:3]:
            href = item.get("href")
            if href and href.startswith("/item/news_read.naver"):
                full_url = base_url + href
                latest_news.append(full_url)
    except Exception as e:
        print(f"[{code}] 뉴스 오류: {e}")

    return {
        "종목명": name,
        "티커": code,
        "작성일자": today,
        "현재가": current_price,
        "ROE_최근": roe_최근, "ROE_전기": roe_전기,
        "부채비율_최근": debt_최근, "부채비율_전기": debt_전기,
        "유보율_최근": reserve_최근, "유보율_전기": reserve_전기,
        "PER_최근": per_최근, "PER_전기": per_전기,
        "PBR_최근": pbr_최근, "PBR_전기": pbr_전기,
        "매출액_최근": sales_최근, "매출액_전기": sales_전기,
        "영업이익_최근": operating_최근, "영업이익_전기": operating_전기,
        "순이익_최근": net_income_최근, "순이익_전기": net_income_전기,
        "최신뉴스": latest_news
    }


# ------------------- 비동기 수집 실행 --------------------
async def collect_kospi_reports_async(limit=None):
    tickers_df = get_kospi_tickers()
    if limit:
        tickers_df = tickers_df.head(limit)

    semaphore = asyncio.Semaphore(10)
    async with aiohttp.ClientSession(headers={'User-Agent': 'Mozilla/5.0'}) as session:
        tasks = [get_stock_report_async(code, session, semaphore) for code in tickers_df['종목코드']]
        reports = await asyncio.gather(*tasks)

    reports = [r for r in reports if r]  # None 제거
    return pd.DataFrame(reports)


# ------------------- 가격 수집 및 DB 저장 --------------------
def collect_and_save_all():
    df = asyncio.run(collect_kospi_reports_async(limit=None))
    df = df.dropna()

    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)
    date_list = stock.get_market_ohlcv_by_date(
        start_date.strftime("%Y%m%d"),
        end_date.strftime("%Y%m%d"),
        "005930"
    ).index.strftime("%Y%m%d").tolist()

    price_df = pd.DataFrame()
    for date in date_list:
        try:
            daily_prices = stock.get_market_ohlcv_by_ticker(date, market="ALL")[["종가"]]
            daily_prices.columns = [date]
            price_df = pd.concat([price_df, daily_prices], axis=1)
        except:
            print(f"{date} 실패")
            continue

    merged_df = price_df.reset_index().merge(df, on="티커", how="inner")

    # 뉴스 리스트 문자열로
    if '최신뉴스' in merged_df.columns:
        merged_df['최신뉴스'] = merged_df['최신뉴스'].apply(
            lambda x: '\n'.join(x) if isinstance(x, list) else x
        )

    db_path = "financial_data.db"
    conn = sqlite3.connect(db_path)
    merged_df.to_sql('financial_data', conn, if_exists='replace', index=False)
    conn.close()

    print(f"✅ {db_path} 저장 완료")


# 실행
collect_and_save_all()

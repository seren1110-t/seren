# kospi_research_app.py
import streamlit as st
import pandas as pd
import sqlite3

st.set_page_config(page_title="📈 KOSPI Analyst AI", layout="wide")

@st.cache_data
def load_data(db_name="financial_data.db", table_name="financial_data"):
    conn = sqlite3.connect(db_name)
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

df = load_data()

st.sidebar.header("📂 필터 옵션")

# 초성 필터
initials = ['전체', 'ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
selected_initial = st.sidebar.selectbox("🔡 종목명 초성:", initials)

def get_initial(korean_char):
    ch_code = ord(korean_char) - ord('가')
    if 0 <= ch_code < 11172:
        cho = ch_code // 588
        return ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ'][cho]
    return ""

if selected_initial != "전체":
    df = df[df["종목명"].apply(lambda x: get_initial(x[0]) == selected_initial)]

# 텍스트 검색
search_term = st.sidebar.text_input("🔍 종목명 또는 티커 검색")
if search_term:
    df = df[df["종목명"].str.contains(search_term, case=False) | df["티커"].str.contains(search_term, case=False)]

종목_list = df["종목명"].tolist()

if not 종목_list:
    st.warning("❌ 조건에 맞는 종목이 없습니다.")
    st.stop()

선택한_종목 = st.sidebar.selectbox("📌 종목 선택:", 종목_list)
종목_df = df[df["종목명"] == 선택한_종목].iloc[0]

# ----------------------- 메인 컨텐츠 -----------------------
st.title(f"📊 {선택한_종목} ({종목_df['티커']}) 리서치 요약")

col1, col2 = st.columns(2)

with col1:
    st.metric("현재가", 종목_df["현재가"])
    st.metric("ROE (최근)", 종목_df["ROE_최근"])
    st.metric("PER (최근)", 종목_df["PER_최근"])
    st.metric("PBR (최근)", 종목_df["PBR_최근"])
    st.metric("부채비율", 종목_df["부채비율_최근"])

with col2:
    st.metric("유보율", 종목_df["유보율_최근"])
    st.metric("매출액", 종목_df["매출액_최근"])
    st.metric("영업이익", 종목_df["영업이익_최근"])
    st.metric("순이익", 종목_df["순이익_최근"])

# 그래프
st.markdown("### 📈 주가 추이")
price_cols = [col for col in df.columns if col.isdigit() and len(col) == 8]
price_series = 종목_df[price_cols].astype(float)
price_series.index = pd.to_datetime(price_cols, format='%Y%m%d')
chart_df = price_series.reset_index().rename(columns={'index': '날짜'})
st.line_chart(chart_df.set_index("날짜"))

# 뉴스
st.markdown("### 📰 최근 뉴스")
if isinstance(종목_df["최신뉴스"], str):
    for i, link in enumerate(종목_df["최신뉴스"].splitlines(), 1):
        st.markdown(f"{i}. [뉴스 링크]({link})")
else:
    st.info("최근 뉴스 없음")

# 향후 확장: LLM 질의 입력 박스
st.markdown("### 💬 AI 리서치 질의 (예: '이 종목의 PER이 높으면 어떤 해석이 가능해?')")
user_question = st.text_input("🧠 궁금한 점을 입력하세요:")
if user_question:
    st.write("🔎 (LangChain 질의 처리 영역 연결 예정)")
    # 향후 vectorstore 검색 + LLM 요약으로 연결 가능

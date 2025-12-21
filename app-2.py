import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

# -----------------------------------------------------------------------------
# 1. 页面配置
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Quant Sniper Pro", layout="wide", page_icon="⚡")

st.markdown("""
<style>
    .metric-card { background-color: #1e1e1e; border: 1px solid #333; padding: 15px; border-radius: 8px; text-align: center; }
    .scan-result { padding: 10px; border-radius: 5px; margin-bottom: 5px; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. 核心算法 (保持不变)
# -----------------------------------------------------------------------------
def get_swing_pivots(series, threshold=0.06):
    pivots = []
    last_pivot_price = series.iloc[0]
    last_pivot_date = series.index[0]
    last_pivot_type = 0 
    temp_extreme_price = series.iloc[0]
    temp_extreme_date = series.index[0]
    
    for date, price in series.items():
        if last_pivot_type == 0:
            if price > last_pivot_price * (1 + threshold):
                last_pivot_type = -1
                pivots.append({'date': last_pivot_date, 'price': last_pivot_price, 'type': -1})
                temp_extreme_price = price
                temp_extreme_date = date
            elif price < last_pivot_price * (1 - threshold):
                last_pivot_type = 1
                pivots.append({'date': last_pivot_date, 'price': last_pivot_price, 'type': 1})
                temp_extreme_price = price
                temp_extreme_date = date     
        elif last_pivot_type == -1: 
            if price > temp_extreme_price:
                temp_extreme_price = price
                temp_extreme_date = date
            elif price < temp_extreme_price * (1 - threshold):
                pivots.append({'date': temp_extreme_date, 'price': temp_extreme_price, 'type': 1})
                last_pivot_type = 1
                last_pivot_price = temp_extreme_price
                temp_extreme_price = price
                temp_extreme_date = date
        elif last_pivot_type == 1:
            if price < temp_extreme_price:
                temp_extreme_price = price
                temp_extreme_date = date
            elif price > temp_extreme_price * (1 + threshold):
                pivots.append({'date': temp_extreme_date, 'price': temp_extreme_price, 'type': -1})
                last_pivot_type = -1
                last_pivot_price = temp_extreme_price
                temp_extreme_price = price
                temp_extreme_date = date
    return pd.DataFrame(pivots)

def analyze_ticker(ticker, lookback="1y", threshold=0.06):
    try:
        df = yf.download(ticker, period=lookback, interval="1d", progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        if len(df) < 50: return None
        
        current_price = df['Close'].iloc[-1]
        
        pivots_df = get_swing_pivots(df['Close'], threshold=threshold)
        if len(pivots_df) < 3: return None
        
        # 倒序找结构
        for i in range(len(pivots_df)-3, -1, -1):
            pA = pivots_df.iloc[i]
            pB = pivots_df.iloc[i+1]
            pC = pivots_df.iloc[i+2]
            
            # 形态: A(低)->B(高)->C(低) 且 C > A
            if (pA['type'] == -1 and pB['type'] == 1 and pC['type'] == -1) and                (pB['price'] > pA['price'] and pC['price'] > pA['price']):
                
                wave_height = pB['price'] - pA['price']
                target_1618 = pC['price'] + wave_height * 1.618
                
                stop_loss = pA['price']
                risk = current_price - stop_loss
                potential = target_1618 - current_price
                rr = potential / risk if risk > 0 else 0
                
                # 信号判断
                signal = "WAIT"
                if current_price < stop_loss: signal = "STOP_OUT"
                elif current_price >= target_1618: signal = "TAKE_PROFIT"
                elif rr > 2.0 and current_price < pB['price']: signal = "BUY"
                
                return {
                    "ticker": ticker,
                    "price": current_price,
                    "signal": signal,
                    "rr": rr,
                    "target": target_1618,
                    "stop": stop_loss,
                    "data": df,
                    "pivots": (pA, pB, pC)
                }
        return None
    except:
        return None

# -----------------------------------------------------------------------------
# 3. 侧边栏模式选择
# -----------------------------------------------------------------------------
st.sidebar.header("🕹️ 模式选择")
mode = st.sidebar.radio("选择功能:", ["🔍 单股精细分析", "🚀 全市场批量扫描"])

# -----------------------------------------------------------------------------
# 模式 A: 单股分析 (原功能)
# -----------------------------------------------------------------------------
if mode == "🔍 单股精细分析":
    st.title("🔍 单股精细分析模式")
    ticker = st.sidebar.text_input("输入股票代码", value="TSLA").upper()
    threshold = st.sidebar.slider("灵敏度", 0.03, 0.15, 0.06, 0.01)
    
    if st.button("开始分析"):
        res = analyze_ticker(ticker, threshold=threshold)
        if res:
            pA, pB, pC = res['pivots']
            c1, c2, c3 = st.columns(3)
            c1.metric("当前价格", f"${res['price']:.2f}")
            c2.metric("盈亏比 (R/R)", f"{res['rr']:.2f}", delta="推荐" if res['rr']>2 else "观望")
            c3.metric("目标位", f"${res['target']:.2f}")
            
            color = "green" if res['signal'] == "BUY" else "orange"
            st.markdown(f"<h3 style='color:{color}'>当前信号: {res['signal']}</h3>", unsafe_allow_html=True)
            
            # 画图
            fig = go.Figure()
            df = res['data']
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
            fig.add_trace(go.Scatter(x=[pA['date'], pB['date'], pC['date']], y=[pA['price'], pB['price'], pC['price']], mode='lines+markers', line=dict(color='blue', dash='dash')))
            fig.add_hline(y=res['target'], line_color="green", annotation_text="1.618")
            fig.add_hline(y=res['stop'], line_color="red", annotation_text="Stop")
            fig.update_layout(template="plotly_dark", height=500, margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("未发现结构或数据不足。")

# -----------------------------------------------------------------------------
# 模式 B: 批量扫描 (新功能)
# -----------------------------------------------------------------------------
else:
    st.title("🚀 全市场机会扫描器")
    st.markdown("自动遍历列表，寻找符合 **ABC斐波那契 + 高盈亏比** 的标的。")
    
    # 默认列表
    default_tickers = "TSLA, NVDA, AAPL, AMD, MSFT, GOOG, META, AMZN, NFLX, COIN, MSTR, MARA, PLTR, BABA, PDD, QQQ, SPY, IWM"
    user_tickers = st.text_area("输入要扫描的股票池 (用逗号分隔)", value=default_tickers, height=100)
    scan_threshold = st.slider("扫描灵敏度", 0.04, 0.10, 0.06)
    
    if st.button("⚡ 开始全网扫描"):
        ticker_list = [t.strip().upper() for t in user_tickers.split(",") if t.strip()]
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, t in enumerate(ticker_list):
            status_text.text(f"正在分析 {t} ({i+1}/{len(ticker_list)})...")
            res = analyze_ticker(t, threshold=scan_threshold)
            
            if res and res['signal'] == "BUY":
                results.append(res)
            
            progress_bar.progress((i + 1) / len(ticker_list))
            time.sleep(0.1) # 防封控
            
        progress_bar.empty()
        status_text.empty()
        
        if len(results) > 0:
            st.success(f"扫描完成！发现 {len(results)} 个潜在机会：")
            
            # 转换为 DataFrame 展示
            scan_data = []
            for r in results:
                scan_data.append({
                    "股票代码": r['ticker'],
                    "现价": f"${r['price']:.2f}",
                    "盈亏比": round(r['rr'], 2),
                    "止损位": f"${r['stop']:.2f}",
                    "目标位": f"${r['target']:.2f}",
                    "入场日期": r['pivots'][2]['date'].strftime('%Y-%m-%d')
                })
            
            st.dataframe(pd.DataFrame(scan_data), use_container_width=True)
            
            # 展示详细图表
            st.markdown("---")
            st.subheader("📊 机会详情图表")
            for r in results:
                with st.expander(f"查看 {r['ticker']} (盈亏比: {r['rr']:.2f})"):
                    pA, pB, pC = r['pivots']
                    fig = go.Figure()
                    df = r['data']
                    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
                    fig.add_trace(go.Scatter(x=[pA['date'], pB['date'], pC['date']], y=[pA['price'], pB['price'], pC['price']], mode='lines+markers', line=dict(color='blue', dash='dash')))
                    fig.add_hline(y=r['target'], line_color="green", annotation_text="Target")
                    fig.add_hline(y=r['stop'], line_color="red", annotation_text="Stop")
                    fig.update_layout(template="plotly_dark", height=400, title=f"{r['ticker']} 斐波那契结构")
                    st.plotly_chart(fig)
        else:
            st.info("扫描完成，但当前列表中的股票暂无符合 '买入' 条件的结构。")

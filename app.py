import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# -----------------------------------------------------------------------------
# 页面配置
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Quant Sniper Pro", layout="wide", page_icon="📈")

st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e1e;
        border: 1px solid #333;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 核心算法：ZigZag + 斐波那契
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

def analyze_structure(df, threshold):
    # 如果数据太少，直接返回
    if len(df) < 50: return None
    
    pivots_df = get_swing_pivots(df['Close'], threshold=threshold)
    if len(pivots_df) < 3: return None
    
    # 倒序寻找符合 A->B->C 结构的波段
    for i in range(len(pivots_df)-3, -1, -1):
        pA = pivots_df.iloc[i]
        pB = pivots_df.iloc[i+1]
        pC = pivots_df.iloc[i+2]
        
        # 形态验证: 低(-1) -> 高(1) -> 低(-1)
        if (pA['type'] == -1 and pB['type'] == 1 and pC['type'] == -1):
            # 价格验证: 底底高 (C > A, B > A)
            if (pB['price'] > pA['price'] and pC['price'] > pA['price']):
                
                # 计算目标位
                wave_height = pB['price'] - pA['price']
                target_1618 = pC['price'] + wave_height * 1.618
                return pA, pB, pC, target_1618
            
    return None

# -----------------------------------------------------------------------------
# 侧边栏设置
# -----------------------------------------------------------------------------
st.sidebar.header("🛠️ 狙击参数设置")
ticker = st.sidebar.text_input("股票代码 (Ticker)", value="TSLA").upper()
threshold = st.sidebar.slider("波段灵敏度 (Threshold)", 0.03, 0.15, 0.06, 0.01, help="越大过滤越多噪音，越小波段越密集")
lookback = st.sidebar.selectbox("数据回溯时间", ["3mo", "6mo", "1y", "2y", "5y"], index=2)

if st.sidebar.button("🔄 刷新数据"):
    st.cache_data.clear()

# -----------------------------------------------------------------------------
# 主界面逻辑
# -----------------------------------------------------------------------------
st.title(f"🚀 {ticker} 量化实战指挥舱")

# 获取数据
with st.spinner('正在连接交易所数据...'):
    try:
        # 针对 yfinance 新版的 auto_adjust 逻辑调整
        df = yf.download(ticker, period=lookback, interval="1d", progress=False, auto_adjust=False)
        
        # 处理 MultiIndex 列名问题 (yfinance v0.2.x 常见问题)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        if len(df) == 0:
            st.error(f"❌ 无法获取 {ticker} 的数据，请检查拼写或网络。")
            st.stop()
            
        current_price = df['Close'].iloc[-1]
        prev_close = df['Close'].iloc[-2]
        daily_change = (current_price - prev_close) / prev_close * 100
        
    except Exception as e:
        st.error(f"数据下载出错: {e}")
        st.stop()

# 顶部指标卡
c1, c2, c3, c4 = st.columns(4)
c1.metric("当前价格", f"${current_price:.2f}", f"{daily_change:.2f}%")

# 运行核心分析
result = analyze_structure(df, threshold)

if result:
    pA, pB, pC, target_1618 = result
    
    # 策略计算
    stop_loss = pA['price']
    potential_profit = target_1618 - current_price
    risk = current_price - stop_loss
    
    # 防止分母为0
    if risk <= 0:
        rr_ratio = 0 
    else:
        rr_ratio = potential_profit / risk
    
    # 状态判定
    status_text = "等待观望 (WAIT)"
    status_color = "orange"
    
    if current_price < stop_loss:
        status_text = "❌ 已破位止损 (STOP OUT)"
        status_color = "#ff4b4b" # Red
    elif current_price >= target_1618:
        status_text = "💰 已达止盈位 (TAKE PROFIT)"
        status_color = "#09ab3b" # Green
    elif rr_ratio > 2.0 and current_price < pB['price']:
        status_text = "🟢 极佳买点 (BUY ZONE)"
        status_color = "#00FF00" # Bright Green
    elif rr_ratio > 1.5:
        status_text = "🟡 此时买入风险适中 (HOLD)"
        status_color = "#FFD700" # Gold
    elif rr_ratio < 1.0:
        status_text = "⚠️ 盈亏比极差 (HIGH RISK)"
        status_color = "#ff4b4b"

    # 显示策略指标
    c2.metric("止损位 (A点)", f"${stop_loss:.2f}", delta=f"{stop_loss-current_price:.2f}", delta_color="inverse")
    c3.metric("止盈目标 (1.618)", f"${target_1618:.2f}", delta=f"{target_1618-current_price:.2f}")
    c4.metric("盈亏比 (R/R)", f"{rr_ratio:.2f}", delta="> 2.0 优秀" if rr_ratio>2 else "一般")

    # 信号横幅
    st.markdown(f"""
    <div style="background-color: #262730; padding: 15px; border-radius: 10px; border-left: 10px solid {status_color}; margin-bottom: 20px;">
        <h3 style="color: {status_color}; margin:0;">信号: {status_text}</h3>
        <p style="margin:5px 0 0 0; color: #ccc; font-size: 14px;">
            结构识别: A点({pA['date'].date()}) ➔ B点({pB['date'].date()}) ➔ C点({pC['date'].date()})
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # -------------------------------------------------------------------------
    # 交互式绘图
    # -------------------------------------------------------------------------
    fig = go.Figure()

    # K线图
    fig.add_trace(go.Candlestick(x=df.index,
                    open=df['Open'], high=df['High'],
                    low=df['Low'], close=df['Close'],
                    name='Price'))

    # ABC 结构连线
    fig.add_trace(go.Scatter(x=[pA['date'], pB['date'], pC['date']], 
                             y=[pA['price'], pB['price'], pC['price']],
                             mode='lines+markers', name='Structure',
                             line=dict(color='blue', width=2, dash='dash'),
                             marker=dict(size=8, color='yellow', symbol='diamond')))

    # 目标线与止损线
    fig.add_hline(y=target_1618, line_dash="solid", line_color="green", annotation_text="Target 1.618")
    fig.add_hline(y=stop_loss, line_dash="dot", line_color="red", annotation_text="Stop Loss (A)")
    
    # 标记 B 点高位
    fig.add_hline(y=pB['price'], line_dash="dot", line_color="gray", annotation_text="Breakout (B)", opacity=0.5)

    fig.update_layout(
        title=f"{ticker} Fibonacci Structure Analysis",
        yaxis_title="Price (USD)",
        xaxis_title="Date",
        template="plotly_dark",
        height=600,
        showlegend=False,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    # 隐藏周末空缺
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # 交易计划文字版
    # -------------------------------------------------------------------------
    st.markdown("---")
    col_plan1, col_plan2 = st.columns(2)
    with col_plan1:
        st.info(f"""
        **📋 现货交易计划:**
        1. **买入条件:** 价格 > C点(${pC['price']:.2f}) 且 < B点(${pB['price']:.2f})。
        2. **硬止损:** 价格跌破 **${stop_loss:.2f}**。
        3. **第一减仓位:** ${target_1618:.2f} (卖出 50%)。
        """)
        
    with col_plan2:
        st.success(f"""
        **⚡ 期权博弈建议 (Options):**
        * **策略:** 牛市价差 (Bull Call Spread)
        * **买入腿:** Strike ${int(current_price)} Call
        * **卖出腿:** Strike ${int(target_1618)} Call
        * **期限:** 建议选择 **45天以上** 到期的合约。
        """)

else:
    st.warning(f"⚠️ 在 {ticker} 过去 {lookback} 的走势中，未检测到标准的 'A-B-C' 斐波那契结构。")
    st.markdown("建议：
 1. 尝试调整左侧的 **波段灵敏度**。
 2. 尝试切换 **数据回溯时间**。
 3. 换一个近期波动较大的股票。")
                
    
    # 即使没有结构，也画个简单的K线图给用户看
    fig_simple = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
    fig_simple.update_layout(template="plotly_dark", title=f"{ticker} Daily Chart", height=500)
    st.plotly_chart(fig_simple, use_container_width=True)

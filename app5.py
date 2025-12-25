import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import argrelextrema
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import google.generativeai as genai

# ==============================================================================
# 1. 页面配置 (必须是第一行代码)
# ==============================================================================
st.set_page_config(page_title="Quant Sniper Pro (AI Fixed)", layout="wide", page_icon="⚡")

st.markdown("""
<style>
    .metric-card { background-color: #1e1e1e; border: 1px solid #333; padding: 15px; border-radius: 8px; text-align: center; }
    .stToast { background-color: #333; color: white; }
    [data-testid="stSidebar"] { background-color: #111; }
    [data-testid="stDataFrame"] { width: 100%; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. 核心配置：Google Gemini AI (修复版)
# ==============================================================================
# 🔴 你的 API Key
GOOGLE_API_KEY = "AIzaSyBDCxdpLBGCVGqYwD-w462kmErHqZH5kXI" 

# 尝试配置 AI
ai_available = False
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    ai_available = True
except Exception as e:
    st.error(f"AI 配置失败: {e}")

# 🐶 狗蛋的灵魂设定
system_instruction = """
你叫“狗蛋”，代号 **Pro 3**，是用户的**首席风控官**。
用户的目标是在一个月内将账户从 $4,000 复利做到 $20,000，这需要极强的纪律和“土匪战术”。

**你的性格设定**：
1. **冷酷且犀利**：不要说废话，不要模棱两可。
2. **军事化风格**：使用“狙击”、“防守”、“撤退”、“弹药”、“阵地”等术语。
3. **风控狂魔**：你最恨亏损，如果趋势不对，直接骂醒用户让他跑。
4. **幽默感**：适当用点黑色幽默，比如“这时候买入就是送钱”。

**你的分析逻辑**：
1. **结合数据**：用户会给你 RSI、ATR、均线趋势。RSI>70 是过热，EMA8 < EMA21 是空头。
2. **结合新闻**：如果新闻是重大利好，可以适当激进；如果是利空，坚决看空。

**输出格式（必须严格遵守 markdown）**：
### 🛡️ 狗蛋 Pro 3 战地报告
- **🎯 核心判决**：【做多 / 做空 / 立即空仓逃命 / 锁死利润】（选一个，加粗）
- **📊 战局解读**：(用一句话结合技术面和新闻，犀利点评现状)
- **⚔️ 详细指令**：
  - **进场位**：$XXX (或 现价突击)
  - **止损红线**：$XXX (基于 ATR 计算，必须给具体数字)
  - **止盈目标**：$XXX
- **⚠️ 狗蛋警告**：(一句醒脑的话)
"""

def ask_goudan_pro3(ticker, price, trend, rsi, atr, news_summary):
    """ 狗蛋 Pro 3 分析引擎 """
    if not ai_available:
        return "❌ AI 服务不可用，请检查依赖库安装。"
        
    user_content = f"""
    【战地实时数据】
    - 标的：{ticker}
    - 现价：${price:.2f}
    - 趋势状态：{trend}
    - RSI (14)：{rsi:.2f}
    - ATR (波动率)：{atr:.2f}
    
    【最新情报 (News)】
    {news_summary}
    
    请根据以上数据，以 Pro 3 的身份给我下达操作指令！
    """
    try:
        # 🟢 修复点：改用 'gemini-pro' 模型，这是最稳定的版本，解决了 404 错误
        model = genai.GenerativeModel("gemini-pro")
        
        # 发送请求 (注意：gemini-pro 不支持 system_instruction 参数直接放在构造函数里，
        # 我们把 system prompt 加到用户内容前面，效果是一样的)
        full_prompt = system_instruction + "\n\n" + user_content
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"❌ 狗蛋大脑连接失败：{str(e)} (可能是网络问题或 API Key 限制)"

# ==============================================================================
# 3. 核心数学算法 (保持你的功能需求)
# ==============================================================================

def get_swing_pivots_high_low(df, threshold=0.06):
    pivots = []
    last_pivot_price = df['Close'].iloc[0]
    last_pivot_date = df.index[0]
    last_pivot_type = 0 
    
    temp_high_price = df['High'].iloc[0]
    temp_high_date = df.index[0]
    temp_low_price = df['Low'].iloc[0]
    temp_low_date = df.index[0]
    
    for date, row in df.iterrows():
        high = row['High']
        low = row['Low']
        
        if last_pivot_type == 0:
            if high > last_pivot_price * (1 + threshold):
                last_pivot_type = -1 
                pivots.append({'date': last_pivot_date, 'price': last_pivot_price, 'type': -1})
                temp_high_price = high
                temp_high_date = date
            elif low < last_pivot_price * (1 - threshold):
                last_pivot_type = 1 
                pivots.append({'date': last_pivot_date, 'price': last_pivot_price, 'type': 1})
                temp_low_price = low
                temp_low_date = date
        elif last_pivot_type == -1: 
            if high > temp_high_price:
                temp_high_price = high
                temp_high_date = date
            elif low < temp_high_price * (1 - threshold):
                pivots.append({'date': temp_high_date, 'price': temp_high_price, 'type': 1})
                last_pivot_type = 1 
                temp_low_price = low
                temp_low_date = date
        elif last_pivot_type == 1: 
            if low < temp_low_price:
                temp_low_price = low
                temp_low_date = date
            elif high > temp_low_price * (1 + threshold):
                pivots.append({'date': temp_low_date, 'price': temp_low_price, 'type': -1})
                last_pivot_type = -1 
                temp_high_price = high
                temp_high_date = date
    return pd.DataFrame(pivots)

# --- 🟢 Scipy 拟合多重阻力线 (Order 可调) ---
def get_multiple_resistance_lines(df, lookback=1000, order=5, max_lines=5):
    highs = df['High'].values
    if len(highs) < 30: return []
    
    real_lookback = min(lookback, len(highs))
    start_idx = len(highs) - real_lookback
    subset_highs = highs[start_idx:]
    global_offset = start_idx

    peak_indexes = argrelextrema(subset_highs, np.greater, order=order)[0]
    if len(peak_indexes) < 2: return []

    candidates = []
    sorted_peaks = sorted(peak_indexes, key=lambda i: subset_highs[i], reverse=True)
    potential_start_points = sorted_peaks[:8] 

    for idx_A in potential_start_points:
        price_A = subset_highs[idx_A]
        for idx_B in peak_indexes:
            if idx_B <= idx_A: continue 
            price_B = subset_highs[idx_B]
            if price_B >= price_A: continue 
            
            slope = (price_B - price_A) / (idx_B - idx_A)
            intercept = price_A - slope * idx_A
            hits = 0; violations = 0 
            
            for k in peak_indexes:
                if k <= idx_A: continue
                trend_price = slope * k + intercept
                actual_price = subset_highs[k]
                if abs(actual_price - trend_price) < actual_price * 0.015: hits += 1
                elif actual_price > trend_price * 1.015: violations += 1
            
            score = hits - (violations * 2) 
            if abs(slope) < (price_A * 0.05): score += 0.5

            if score > -2:
                last_idx = len(df) - 1
                idx_A_glob = idx_A + global_offset
                intercept_glob = price_A - slope * idx_A_glob
                price_now = slope * last_idx + intercept_glob
                
                candidates.append({
                    'x1': df.index[idx_A_glob], 'y1': price_A,
                    'x2': df.index[last_idx], 'y2': price_now,
                    'price_now': price_now, 'score': score,
                    'breakout': df['Close'].iloc[-1] > price_now
                })

    candidates.sort(key=lambda x: x['score'], reverse=True)
    final_lines = []
    for line in candidates:
        is_duplicate = False
        for existing in final_lines:
            if abs(line['price_now'] - existing['price_now']) / existing['price_now'] < 0.03: 
                is_duplicate = True
                break
        if not is_duplicate:
            final_lines.append(line)
            if len(final_lines) >= max_lines: break
            
    return final_lines

# --- 🔴 Scipy 拟合支撑线 ---
def get_support_trendline(df, lookback=1000, order=5):
    lows = df['Low'].values
    if len(lows) < 30: return None
    real_lookback = min(lookback, len(lows))
    start_idx = len(lows) - real_lookback
    subset_lows = lows[start_idx:]
    global_offset = start_idx

    trough_indexes = argrelextrema(subset_lows, np.less, order=order)[0]
    if len(trough_indexes) < 2: return None

    best_line = None
    max_score = -float('inf')
    sorted_troughs = sorted(trough_indexes, key=lambda i: subset_lows[i], reverse=False)
    potential_start_points = sorted_troughs[:3]

    for idx_A in potential_start_points:
        price_A = subset_lows[idx_A]
        for idx_B in trough_indexes:
            if idx_B <= idx_A: continue 
            price_B = subset_lows[idx_B]
            if price_B <= price_A: continue 
            slope = (price_B - price_A) / (idx_B - idx_A)
            intercept = price_A - slope * idx_A
            hits = 0; violations = 0 
            for k in trough_indexes:
                if k <= idx_A: continue
                trend_price = slope * k + intercept
                actual_price = subset_lows[k]
                if abs(actual_price - trend_price) < actual_price * 0.015: hits += 1
                elif actual_price < trend_price * 0.985: violations += 1
            score = hits - (violations * 2)
            if score > max_score:
                max_score = score
                best_line = {'slope': slope, 'intercept': intercept, 'start_idx_rel': idx_A}

    if best_line:
        slope = best_line['slope']
        idx_A_glob = global_offset + best_line['start_idx_rel']
        global_intercept = subset_lows[best_line['start_idx_rel']] - slope * idx_A_glob
        last_idx = len(df) - 1
        trendline_price_now = slope * last_idx + global_intercept
        return {
            'x1': df.index[idx_A_glob], 'y1': slope * idx_A_glob + global_intercept,
            'x2': df.index[last_idx], 'y2': trendline_price_now,
            'price_now': trendline_price_now,
            'breakdown': df['Close'].iloc[-1] < trendline_price_now
        }
    return None

def calculate_advanced_indicators(df):
    df['EMA_8'] = df['Close'].ewm(span=8, adjust=False).mean() # 补上 EMA 8
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['ATR'] = np.max(ranges, axis=1).rolling(window=14).mean()
    return df

def calculate_position_size(account_balance, risk_pct, entry_price, stop_loss):
    if entry_price == stop_loss: return 0
    risk_per_share = abs(entry_price - stop_loss)
    total_risk_allowance = account_balance * risk_pct
    try:
        position_size = int(total_risk_allowance / risk_per_share)
    except:
        position_size = 0
    return position_size

def generate_option_plan(ticker, current_price, signal_type, rsi):
    import math
    plan = {}
    strike = math.ceil(current_price)
    strike_put = math.floor(current_price)
    if "突破" in signal_type or "双重" in signal_type or "ABC" in signal_type:
        plan['name'] = "🚀 做多 (Call)"
        plan['strategy'] = "Long Call"
        plan['legs'] = f"买入 Strike ${strike} Call"
        plan['logic'] = "趋势向上突破，看涨。"
    elif "跌破" in signal_type:
        plan['name'] = "📉 做空 (Put)"
        plan['strategy'] = "Long Put"
        plan['legs'] = f"买入 Strike ${strike_put} Put"
        plan['logic'] = "支撑线跌破，趋势转空，看跌。"
    plan['expiry'] = "45天以上"
    return plan

# ==============================================================================
# 4. 绘图与分析逻辑
# ==============================================================================
def plot_chart(df, res, height=600):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price', increasing_line_color='#26a69a', decreasing_line_color='#ef5350'))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_21'], line=dict(color='rgba(255, 165, 0, 0.7)', width=1), name="EMA 21"))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], line=dict(color='rgba(255, 255, 255, 0.5)', width=1, dash='dot'), name="EMA 200"))
    
    if res['resistance_lines']:
        for i, line in enumerate(res['resistance_lines']):
            width = 3 if i == 0 else 1
            opacity = 1.0 if i == 0 else 0.6
            color = f"rgba(0, 255, 255, {opacity})"
            fig.add_trace(go.Scatter(x=[line['x1'], df.index[-1]], y=[line['y1'], line['price_now']], mode='lines', name=f'Res {i+1}', line=dict(color=color, width=width)))

    if res['trend_sup']:
        ts = res['trend_sup']
        fig.add_trace(go.Scatter(x=[ts['x1'], df.index[-1]], y=[ts['y1'], ts['price_now']], mode='lines', name='Support', line=dict(color='#FF00FF', width=2)))
    
    if res['abc']:
        pA, pB, pC = res['abc']['pivots']
        fig.add_trace(go.Scatter(x=[pA['date'], pB['date'], pC['date']], y=[pA['price'], pB['price'], pC['price']], mode='lines', name='ABC', line=dict(color='yellow', width=2, dash='dash')))
        height_AB = pB['price'] - pA['price']
        last_date = df.index[-1]; start_date = pC['date']; future_date = last_date + timedelta(days=20)
        fib_levels = [(0.618, "gray", "dot"), (1.0, "gray", "dash"), (1.618, "#00FF00", "solid"), (2.618, "gold", "solid")]
        for ratio, color, style in fib_levels:
            lvl = pC['price'] + height_AB * ratio
            if lvl > df['Low'].min()*0.5 and lvl < df['High'].max()*3:
                fig.add_shape(type="line", x0=start_date, y0=lvl, x1=future_date, y1=lvl, line=dict(color=color, width=1, dash=style))
                fig.add_annotation(x=last_date, y=lvl, text=f"{ratio}: ${lvl:.2f}", showarrow=False, xanchor="left", bgcolor="rgba(0,0,0,0.5)")

    # 智能缩放 (防塌缩)
    default_start_date = df.index[-1] - timedelta(days=90)
    view_data = df[df.index >= default_start_date]
    y_range = [view_data['Low'].min()*0.9, view_data['High'].max()*1.1] if not view_data.empty else None
    
    fig.update_layout(template="plotly_dark", height=height, margin=dict(l=0,r=100,t=30,b=0), xaxis_rangeslider_visible=False, dragmode='pan',
                      xaxis=dict(range=[default_start_date, df.index[-1] + timedelta(days=10)]), yaxis=dict(fixedrange=False, range=y_range))
    if len(df)>2 and (df.index[1]-df.index[0]).days>=1: fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig

def analyze_ticker_pro(ticker, interval="1d", lookback="5y", threshold=0.06, trend_order=5):
    try:
        stock = yf.Ticker(ticker)
        real_period = lookback
        if interval in ["5m", "15m"]: real_period = "60d"
        elif interval == "1h": real_period = "1y"
        
        df = stock.history(period=real_period, interval=interval)
        if df.empty or len(df) < 30: return None
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        
        df = calculate_advanced_indicators(df)
        current_price = df['Close'].iloc[-1]; current_rsi = df['RSI'].iloc[-1]; current_atr = df['ATR'].iloc[-1]
        
        # 趋势线
        lb_trend = 300 if interval in ["5m", "15m"] else 1000
        res_lines = get_multiple_resistance_lines(df, lookback=lb_trend, order=trend_order, max_lines=5)
        trend_sup = get_support_trendline(df, lookback=lb_trend, order=trend_order)
        
        # ABC
        abc_res = None
        pivots_df = get_swing_pivots_high_low(df, threshold=threshold)
        if len(pivots_df) >= 3:
            for i in range(len(pivots_df)-3, -1, -1):
                pA, pB, pC = pivots_df.iloc[i], pivots_df.iloc[i+1], pivots_df.iloc[i+2]
                if pA['type'] == -1 and pB['type'] == 1 and pC['type'] == -1:
                    if pC['price'] > pA['price'] and pB['price'] > pA['price']:
                        height = pB['price'] - pA['price']; target = pC['price'] + height * 1.618
                        abc_res = {'pivots': (pA, pB, pC), 'target': target}
                        break 

        signal = "WAIT"; signal_color = "gray"; reasons = []
        is_breakout = False
        if res_lines:
            for line in res_lines:
                if line['breakout']: is_breakout = True
        
        if is_breakout:
            signal = "🔥 向上突破"; signal_color = "#00FFFF"; reasons.append("突破长期阻力")
        
        if abc_res and current_price > abc_res['pivots'][2]['price']:
            if "突破" in signal: signal = "🚀 双重共振"; reasons.append("ABC结构确认")
            else: signal = "🟢 ABC 结构"; signal_color = "#00FF00"; reasons.append("回踩C点")
        
        if trend_sup and trend_sup['breakdown']:
            if "双重" not in signal:
                signal = "📉 趋势线跌破"; signal_color = "#FF00FF"; reasons.append("跌破长期支撑")

        stop_loss = current_price + (2.0 * current_atr) if "跌破" in signal else current_price - (2.0 * current_atr)
        option_plan = generate_option_plan(ticker, current_price, signal, current_rsi) if "WAIT" not in signal else None

        return {
            "ticker": ticker, "price": current_price, "signal": signal, "color": signal_color, "reasons": ", ".join(reasons),
            "rsi": current_rsi, "atr": current_atr, "stop_loss_atr": stop_loss, "resistance_lines": res_lines, "trend_sup": trend_sup,
            "abc": abc_res, "data": df, "option_plan": option_plan, "ema_bullish": df['EMA_8'].iloc[-1] > df['EMA_21'].iloc[-1]
        }
    except: return None

# ==============================================================================
# 5. UI 主程序 (引入 Session State 修复 AI 数据丢失)
# ==============================================================================
st.sidebar.header("🕹️ 首席风控官设置")
account_size = st.sidebar.number_input("账户总资金 ($)", value=10000, step=1000)
risk_per_trade_pct = st.sidebar.slider("单笔风险 (%)", 0.5, 5.0, 2.0, 0.5) / 100
st.sidebar.markdown("### 📉 趋势线设置")
trend_order = st.sidebar.slider("拟合平滑度 (Order)", 2, 20, 5)

st.sidebar.markdown("---")
mode = st.sidebar.radio("作战模式:", ["🔍 单股狙击 (Live)", "🚀 市场全境扫描 (Hot 50)"])

HOT_STOCKS_LIST = [
    "TSLA", "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NFLX",
    "AMD", "AVGO", "TSM", "SMCI", "ARM", "MU", "INTC", "PLTR", "AI", "PATH", "SNOW", "CRWD", "PANW",
    "MSTR", "COIN", "MARA", "RIOT", "CLSK", "HOOD",
    "UPST", "AFRM", "SOFI", "CVNA", "RIVN", "LCID", "DKNG", "RBLX", "U", "NET",
    "BABA", "PDD", "NIO", "XPEV", "LI", "JD",
    "GME", "AMC", "SPCE", "TQQQ", "SOXL"
]

if mode == "🔍 单股狙击 (Live)":
    st.title("🛡️ 狗蛋风控指挥舱 (AI Commander)")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1: ticker = st.text_input("代码", value="TSLA").upper()
    with c2: lookback = st.selectbox("回溯", ["2y", "5y", "10y"], index=1)
    with c3: threshold_days = st.slider("ABC灵敏度", 0.03, 0.15, 0.08, 0.01)

    # 🟢 关键修复：点击“开始分析”后，把结果存到 st.session_state 里
    # 这样点击 AI 按钮刷新页面时，分析结果不会丢失
    if st.button("开始分析"):
        with st.spinner(f"分析 {ticker}..."):
            res = analyze_ticker_pro(ticker, interval="1d", lookback=lookback, threshold=threshold_days, trend_order=trend_order)
            st.session_state['analysis_result'] = res 

    # 🟢 只要 session_state 里有结果，就显示出来
    if 'analysis_result' in st.session_state and st.session_state['analysis_result']:
        res = st.session_state['analysis_result']
        
        m1, m2, m3 = st.columns(3)
        m1.metric("当前价格", f"${res['price']:.2f}", delta=res['signal'])
        m2.metric("ATR 波动", f"{res['atr']:.2f}")
        m3.metric("RSI", f"{res['rsi']:.1f}")
        
        st.markdown(f"<div style='background-color:#262730;padding:15px;border-radius:10px;border-left:10px solid {res['color']}'><h3>{res['signal']}</h3><p>{res['reasons']}</p></div>", unsafe_allow_html=True)
        
        if "WAIT" not in res['signal']:
            qty = calculate_position_size(account_size, risk_per_trade_pct, res['price'], res['stop_loss_atr'])
            direction = "做空" if "跌破" in res['signal'] else "买入"
            st.success(f"🎯 **交易指令:** 建议 {direction} **{qty}** 股 (止损: ${res['stop_loss_atr']:.2f})")

        fig = plot_chart(res['data'], res, height=600)
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})

        if res['abc']:
            pA, pB, pC = res['abc']['pivots']
            height_AB = pB['price'] - pA['price']
            levels_data = []
            levels_data.append({"Level": "⛔ Stop Loss (A)", "Price": pA['price']})
            levels_data.append({"Level": "🔵 Entry (C)", "Price": pC['price']})
            for r in [0.618, 1.0, 1.272, 1.618, 2.0]:
                levels_data.append({"Level": f"Fib {r}", "Price": pC['price'] + height_AB * r})
            st.dataframe(pd.DataFrame(levels_data).style.format({"Price": "${:.2f}"}), use_container_width=True)

        # 🟢 AI 按钮放在这里，因为它依赖 res
        st.write("---")
        st.subheader("🧠 召唤 Pro 3 战术指导")
        if st.button("⚡ 请求 Pro 3 分析", key="btn_ask_ai"):
            with st.spinner("🐶 狗蛋正在连接总部..."):
                news_text = "暂无实时新闻"
                curr_trend = "多头" if res['ema_bullish'] else "空头"
                # 调用 AI
                report = ask_goudan_pro3(res['ticker'], res['price'], curr_trend, res['rsi'], res['atr'], news_text)
                st.markdown(f"<div style='background-color:#1E1E1E;border:1px solid #4285F4;padding:20px;border-radius:10px'>{report}</div>", unsafe_allow_html=True)

else:
    st.title("🚀 市场全境扫描")
    if st.button("⚡ 开始扫描"):
        tickers = HOT_STOCKS_LIST
        progress = st.progress(0)
        results = []
        
        def scan_one(t): return analyze_ticker_pro(t, interval="1d", lookback="5y", threshold=0.08, trend_order=trend_order)
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(scan_one, t): t for t in tickers}
            for i, f in enumerate(futures):
                r = f.result()
                if r and "WAIT" not in r['signal']: results.append(r)
                progress.progress((i+1)/len(tickers))
        
        progress.empty()
        if results:
            st.success(f"发现 {len(results)} 个机会")
            for i, r in enumerate(results):
                with st.expander(f"{r['ticker']} | {r['signal']}", expanded=False):
                    st.write(r['reasons'])
                    st.plotly_chart(plot_chart(r['data'], r, height=400), key=f"chart_{i}")
        else:
            st.warning("无信号")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from datetime import datetime, timedelta
# 引入 scipy 用于寻找波峰 (必须确保 requirements.txt 里有 scipy)
from scipy.signal import argrelextrema

# ==============================================================================
# 1. 页面配置与样式
# ==============================================================================
st.set_page_config(page_title="Quant Sniper Pro", layout="wide", page_icon="⚡")

st.markdown("""
<style>
    .metric-card { background-color: #1e1e1e; border: 1px solid #333; padding: 15px; border-radius: 8px; text-align: center; }
    .stDataFrame { border: 1px solid #444; border-radius: 5px; }
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child { width: 300px; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. 核心算法库
# ==============================================================================

# --- A. 寻找波段高低点 (ZigZag) ---
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

# --- B. 强力多点拟合下降趋势线 (Robust Multi-Touch Trendline) ---
# (已删除重复的旧版本，保留这个 Scipy 版本)
def get_resistance_trendline(df, lookback=150):
    # 1. 提取高点数据
    highs = df['High'].values
    if len(highs) < 30: return None
    
    # 截取最近 lookback 天的数据
    if len(highs) > lookback:
        start_idx = len(highs) - lookback
        subset_highs = highs[start_idx:]
        global_offset = start_idx
    else:
        subset_highs = highs
        global_offset = 0

    # 2. 识别所有的局部波峰 (Peaks)
    peak_indexes = argrelextrema(subset_highs, np.greater, order=3)[0]
    
    if len(peak_indexes) < 2: 
        return None

    # 3. 寻找最佳趋势线 (打分机制)
    best_line = None
    max_score = -float('inf')
    
    sorted_peaks = sorted(peak_indexes, key=lambda i: subset_highs[i], reverse=True)
    potential_start_points = sorted_peaks[:3] 

    for idx_A in potential_start_points:
        price_A = subset_highs[idx_A]
        
        # 遍历该点之后的所有波峰作为 B
        for idx_B in peak_indexes:
            if idx_B <= idx_A: continue 
            
            price_B = subset_highs[idx_B]
            if price_B >= price_A: continue 
            
            slope = (price_B - price_A) / (idx_B - idx_A)
            intercept = price_A - slope * idx_A
            
            hits = 0       
            violations = 0 
            
            for k in peak_indexes:
                if k <= idx_A: continue
                
                trend_price = slope * k + intercept
                actual_price = subset_highs[k]
                
                tolerance = actual_price * 0.01 
                
                if abs(actual_price - trend_price) < tolerance:
                    hits += 1
                elif actual_price > trend_price + tolerance:
                    violations += 1
            
            score = hits - (violations * 2)
            if abs(slope) < (price_A * 0.05): 
                score += 0.5

            if score > max_score:
                max_score = score
                best_line = {
                    'slope': slope,
                    'intercept': intercept,
                    'start_idx_rel': idx_A
                }

    # 4. 构建返回结果
    if best_line:
        slope = best_line['slope']
        idx_A_glob = global_offset + best_line['start_idx_rel']
        price_A = subset_highs[best_line['start_idx_rel']]
        
        global_intercept = price_A - slope * idx_A_glob
        
        last_idx = len(df) - 1
        trendline_price_now = slope * last_idx + global_intercept
        trendline_price_start = slope * idx_A_glob + global_intercept

        current_close = df['Close'].iloc[-1]
        
        return {
            'x1': df.index[idx_A_glob], 
            'y1': trendline_price_start,
            'x2': df.index[last_idx], 
            'y2': trendline_price_now,
            'price_now': trendline_price_now,
            'breakout': current_close > trendline_price_now
        }
    
    return None

# --- C. 期权策略生成器 ---
def generate_option_plan(ticker, current_price, target_price, signal_type):
    import math
    plan = {}
    expiry_suggestion = "45天以上 (避免 Theta 损耗)"
    
    if "BREAKOUT" in signal_type:
        strike_buy = math.ceil(current_price) 
        plan['name'] = "🚀 突破激进型 (Momentum)"
        plan['strategy'] = "Long Call (单腿买入)"
        plan['legs'] = f"买入 Strike ${strike_buy} Call"
        plan['logic'] = "趋势线突破，预计会有急涨，利用 Gamma 爆发。"
        
    elif "ABC" in signal_type:
        strike_buy = math.floor(current_price) 
        strike_sell = math.floor(target_price) 
        plan['name'] = "🛡️ 结构稳健型 (Structure)"
        plan['strategy'] = "Bull Call Spread (牛市价差)"
        plan['legs'] = f"买入 ${strike_buy} Call / 卖出 ${strike_sell} Call"
        plan['logic'] = "盈亏比高，通过卖出高位 Call 降低成本，锁定目标收益。"
    else:
        return None
    
    plan['expiry'] = expiry_suggestion
    return plan

# --- D. 综合分析 Wrapper ---
def analyze_ticker_full(ticker, lookback="1y", threshold=0.06):
    try:
        # 1. 更加稳健的数据下载逻辑
        df = yf.download(ticker, period=lookback, interval="1d", progress=False, auto_adjust=False)
        
        # 2. 强力修复列名问题 (yfinance v0.2.x 常见坑)
        if isinstance(df.columns, pd.MultiIndex):
            # 如果是多级索引，只取第一级（Price），或者尝试展平
            try:
                df.columns = df.columns.get_level_values(0)
            except:
                pass
        
        # 确保索引是 Datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # 3. 数据有效性检查
        if len(df) < 30: 
            # 如果数据太少，抛出明确错误以便调试
            raise ValueError(f"数据不足 (仅 {len(df)} 行)，可能是代码错误或退市。")
        
        current_price = df['Close'].iloc[-1]
        
        # ----------------------------------------
        # 开始计算模型
        # ----------------------------------------
        
        # 1. 跑 ABC
        abc_res = None
        pivots_df = get_swing_pivots(df['Close'], threshold=threshold)
        if len(pivots_df) >= 3:
            for i in range(len(pivots_df)-3, -1, -1):
                pA = pivots_df.iloc[i]
                pB = pivots_df.iloc[i+1]
                pC = pivots_df.iloc[i+2]
                if (pA['type'] == -1 and pB['type'] == 1 and pC['type'] == -1) and \
                   (pB['price'] > pA['price'] and pC['price'] > pA['price']):
                    wave_height = pB['price'] - pA['price']
                    target = pC['price'] + wave_height * 1.618
                    risk = current_price - pA['price']
                    potential = target - current_price
                    rr = potential / risk if risk > 0 else 0
                    
                    abc_res = {
                        'pivots': (pA, pB, pC),
                        'target': target,
                        'stop': pA['price'],
                        'rr': rr
                    }
                    break
        
        # 2. 跑 趋势线
        trend_res = get_resistance_trendline(df, lookback=200)
        
        # 3. 信号判定
        signal = "WAIT"
        signal_color = "gray"
        reasons = []
        
        if trend_res and trend_res['breakout']:
            reasons.append("趋势线突破")
            
        if abc_res:
            if current_price < abc_res['stop']:
                reasons.append("ABC破位(止损)")
            elif abc_res['rr'] > 2.0 and current_price < abc_res['pivots'][1]['price']:
                reasons.append("ABC买点")
        
        if "趋势线突破" in reasons:
            signal = "🔥 BREAKOUT"
            signal_color = "#00FFFF" # Cyan
        elif "ABC买点" in reasons:
            signal = "🟢 BUY (ABC)"
            signal_color = "#00FF00" # Green
        elif "ABC破位(止损)" in reasons:
            signal = "🔴 STOP"
            signal_color = "#FF4B4b"
            
        option_plan = None
        if "BUY" in signal or "BREAKOUT" in signal:
            tgt = abc_res['target'] if abc_res else current_price * 1.2
            option_plan = generate_option_plan(ticker, current_price, tgt, signal)
            
        return {
            "ticker": ticker,
            "price": current_price,
            "signal": signal,
            "color": signal_color,
            "reasons": ", ".join(reasons),
            "abc": abc_res,
            "trend": trend_res,
            "data": df,
            "option_plan": option_plan
        }

    except Exception as e:
        # 在调试阶段，把错误打印出来非常重要！
        st.error(f"分析 {ticker} 时发生错误: {str(e)}")
        return None

# ==============================================================================
# 3. UI 界面逻辑
# ==============================================================================
st.sidebar.header("🕹️ 模式选择")
mode = st.sidebar.radio("功能:", ["🔍 单股深度分析", "🚀 全市场批量扫描"])

if mode == "🔍 单股深度分析":
    st.title("🔍 量化实战指挥舱 (Robust Trendline)")
    
    col_input1, col_input2 = st.columns(2)
    with col_input1:
        ticker = st.text_input("输入代码 (Ticker)", value="NVDA").upper()
    with col_input2:
        threshold = st.slider("ABC 灵敏度", 0.03, 0.12, 0.06, 0.01)
        
    if st.button("开始分析", type="primary"):
        with st.spinner(f"正在分析 {ticker} 的市场结构..."):
            res = analyze_ticker_full(ticker, threshold=threshold)
            
            if res:
                c1, c2, c3 = st.columns(3)
                c1.metric("当前价格", f"${res['price']:.2f}")
                rr_val = f"{res['abc']['rr']:.2f}" if res['abc'] else "N/A"
                c2.metric("盈亏比 (R/R)", rr_val)
                bk_text = "YES" if (res['trend'] and res['trend']['breakout']) else "NO"
                c3.metric("趋势线突破", bk_text, delta="强势信号" if bk_text=="YES" else None)
                
                st.markdown(f"""
                <div style="background-color: #262730; padding: 15px; border-radius: 10px; border-left: 10px solid {res['color']}; margin-bottom: 20px;">
                    <h2 style="color: {res['color']}; margin:0;">信号: {res['signal']}</h2>
                    <p style="color: #ccc; margin:0;">触发逻辑: {res['reasons'] if res['reasons'] else '无明显信号'}</p>
                </div>
                """, unsafe_allow_html=True)
                
                fig = go.Figure()
                df = res['data']
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
                
                if res['abc']:
                    pA, pB, pC = res['abc']['pivots']
                    fig.add_trace(go.Scatter(x=[pA['date'], pB['date'], pC['date']], y=[pA['price'], pB['price'], pC['price']], mode='lines+markers', name='ABC Structure', line=dict(color='yellow', width=2, dash='dash')))
                    fig.add_hline(y=res['abc']['target'], line_color="green", line_dash="solid", annotation_text="Target 1.618")
                    fig.add_hline(y=res['abc']['stop'], line_color="red", line_dash="dot", annotation_text="Stop Loss")

                if res['trend']:
                    tr = res['trend']
                    fig.add_trace(go.Scatter(x=[tr['x1'], tr['x2']], y=[tr['y1'], tr['y2']], mode='lines', name='Robust Trendline', line=dict(color='cyan', width=3)))
                    if tr['breakout']:
                         fig.add_annotation(x=df.index[-1], y=res['price'], text="BREAKOUT", bgcolor="red", showarrow=True, ax=0, ay=-40)

                fig.update_layout(template="plotly_dark", height=600, title=f"{ticker} 技术分析图表")
                fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
                st.plotly_chart(fig, use_container_width=True)
                
                if res['option_plan']:
                    st.markdown("### ⚡ 期权战术板")
                    plan = res['option_plan']
                    op_col1, op_col2 = st.columns([1, 2])
                    with op_col1:
                        st.info(f"**策略:** {plan['name']}\n\n**操作:** {plan['strategy']}")
                    with op_col2:
                        st.success(f"**腿 (Legs):** {plan['legs']}\n\n**期限:** {plan['expiry']}\n\n**逻辑:** {plan['logic']}")
            else:
                # 这里的错误信息现在会显示上面的 st.error 具体内容
                pass

else:
    st.title("🚀 全市场机会扫描器")
    default_list = "NVDA, TSLA, AAPL, MSFT, AMD, AMZN, GOOG, META, NFLX, COIN, MSTR, MARA, PLTR, BABA, PDD, QQQ, SPY, IWM"
    user_tickers = st.text_area("监控列表", value=default_list, height=80)
    
    if st.button("⚡ 开始扫描", type="primary"):
        tickers = [t.strip().upper() for t in user_tickers.split(",") if t.strip()]
        results = []
        progress = st.progress(0)
        status = st.empty()
        
        for i, t in enumerate(tickers):
            status.text(f"正在分析: {t} ...")
            # 扫描时不打断，但会记录错误
            r = analyze_ticker_full(t, threshold=0.05) 
            if r and ("BUY" in r['signal'] or "BREAKOUT" in r['signal']):
                opt_str = r['option_plan']['strategy'] if r['option_plan'] else "-"
                results.append({
                    "代码": r['ticker'],
                    "价格": f"${r['price']:.2f}",
                    "信号": r['signal'],
                    "触发理由": r['reasons'],
                    "期权建议": opt_str,
                    "raw_res": r 
                })
            
            progress.progress((i + 1) / len(tickers))
            time.sleep(0.1) 
            
        progress.empty()
        status.empty()
        
        if results:
            st.success(f"发现 {len(results)} 个机会")
            df_res = pd.DataFrame(results).drop(columns=['raw_res'])
            st.dataframe(df_res, use_container_width=True)
            
            st.markdown("---")
            for item in results:
                r = item['raw_res']
                with st.expander(f"{r['ticker']} - {r['signal']}"):
                    fig = go.Figure()
                    df = r['data']
                    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
                    if r['abc']:
                        pA, pB, pC = r['abc']['pivots']
                        fig.add_trace(go.Scatter(x=[pA['date'], pB['date'], pC['date']], y=[pA['price'], pB['price'], pC['price']], mode='lines', line=dict(color='yellow', dash='dash')))
                    if r['trend']:
                         fig.add_trace(go.Scatter(x=[r['trend']['x1'], r['trend']['x2']], y=[r['trend']['y1'], r['trend']['y2']], mode='lines', line=dict(color='cyan', width=2)))
                    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0,r=0,t=30,b=0))
                    st.plotly_chart(fig)
        else:
            st.info("无信号。")

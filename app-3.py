import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import argrelextrema
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

# ==============================================================================
# 1. 页面配置与样式 (UI Configuration)
# ==============================================================================
st.set_page_config(page_title="Quant Sniper Pro (Price Levels)", layout="wide", page_icon="⚡")

st.markdown("""
<style>
    .metric-card { background-color: #1e1e1e; border: 1px solid #333; padding: 15px; border-radius: 8px; text-align: center; }
    /* 调整 Toast */
    .stToast { background-color: #333; color: white; }
    /* 侧边栏优化 */
    [data-testid="stSidebar"] { background-color: #111; }
    /* 表格样式 */
    [data-testid="stDataFrame"] { width: 100%; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. 核心数学算法 (Core Algorithms)
# ==============================================================================

def get_swing_pivots_high_low(df, threshold=0.06):
    """ [精度升级版] ZigZag 算法 """
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

def get_resistance_trendline(df, lookback=1000):
    """ 强力趋势线拟合 (使用 High) """
    highs = df['High'].values
    if len(highs) < 30: return None
    
    real_lookback = min(lookback, len(highs))
    start_idx = len(highs) - real_lookback
    subset_highs = highs[start_idx:]
    global_offset = start_idx

    peak_indexes = argrelextrema(subset_highs, np.greater, order=5)[0]
    if len(peak_indexes) < 2: return None

    best_line = None
    max_score = -float('inf')
    
    sorted_peaks = sorted(peak_indexes, key=lambda i: subset_highs[i], reverse=True)
    potential_start_points = sorted_peaks[:5]

    for idx_A in potential_start_points:
        price_A = subset_highs[idx_A]
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
                tolerance = actual_price * 0.02 
                
                if abs(actual_price - trend_price) < tolerance:
                    hits += 1
                elif actual_price > trend_price + tolerance:
                    violations += 1
            
            score = hits - (violations * 3) 
            if abs(slope) < (price_A * 0.05): score += 0.5

            if score > max_score:
                max_score = score
                best_line = {'slope': slope, 'intercept': intercept, 'start_idx_rel': idx_A}

    if best_line:
        slope = best_line['slope']
        idx_A_glob = global_offset + best_line['start_idx_rel']
        global_intercept = subset_highs[best_line['start_idx_rel']] - slope * idx_A_glob
        
        last_idx = len(df) - 1
        trendline_price_now = slope * last_idx + global_intercept
        
        return {
            'x1': df.index[idx_A_glob], 
            'y1': slope * idx_A_glob + global_intercept,
            'x2': df.index[last_idx], 
            'y2': trendline_price_now,
            'price_now': trendline_price_now,
            'breakout': df['Close'].iloc[-1] > trendline_price_now
        }
    return None

def calculate_advanced_indicators(df):
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['ATR'] = np.max(ranges, axis=1).rolling(window=14).mean()
    
    return df

def calculate_position_size(account_balance, risk_pct, entry_price, stop_loss):
    if entry_price <= stop_loss: return 0
    risk_per_share = entry_price - stop_loss
    total_risk_allowance = account_balance * risk_pct
    position_size = int(total_risk_allowance / risk_per_share)
    return position_size

def generate_option_plan(ticker, current_price, signal_type, rsi):
    import math
    plan = {}
    strike_buy = math.ceil(current_price)
    
    if "BREAKOUT" in signal_type or "ENTRY" in signal_type:
        if rsi > 70:
            plan['name'] = "⚠️ 风险过热保护"
            plan['strategy'] = "Debit Call Spread"
            plan['legs'] = f"买 ${strike_buy} / 卖 ${strike_buy+5} Call"
            plan['logic'] = "趋势向上但超买，用价差锁定利润并降低成本。"
        else:
            plan['name'] = "🚀 趋势爆发狙击"
            plan['strategy'] = "Long Call"
            plan['legs'] = f"买入 Strike ${strike_buy} Call"
            plan['logic'] = "ABC结构确认/趋势突破，动能充足，单腿做多。"
        plan['expiry'] = "45天以上"
    return plan

# ==============================================================================
# 3. 核心绘图系统 (Visual Engine)
# ==============================================================================
def plot_chart(df, res, height=600):
    fig = go.Figure()
    
    # 1. K 线
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], 
        name='Price',
        increasing_line_color='#26a69a', increasing_fillcolor='#26a69a', 
        decreasing_line_color='#ef5350', decreasing_fillcolor='#ef5350'
    ))
    
    # 2. EMA 均线
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_21'], line=dict(color='rgba(255, 165, 0, 0.7)', width=1), name="EMA 21"))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], line=dict(color='rgba(255, 255, 255, 0.5)', width=1, dash='dot'), name="EMA 200"))
    
    # 3. 趋势线
    if res['trend']:
        tr = res['trend']
        fig.add_trace(go.Scatter(
            x=[tr['x1'], tr['x2']], y=[tr['y1'], tr['y2']], 
            mode='lines', name='Res Trend', line=dict(color='cyan', width=2)
        ))

    # 4. 🔥 斐波那契战术地图 (带价格标注)
    if res['abc']:
        pA, pB, pC = res['abc']['pivots']
        
        # (A) ABC 连线
        fig.add_trace(go.Scatter(
            x=[pA['date'], pB['date'], pC['date']], 
            y=[pA['price'], pB['price'], pC['price']], 
            mode='lines', name='ABC Structure', 
            line=dict(color='yellow', width=2, dash='dash')
        ))
        
        # ABC 文字
        fig.add_trace(go.Scatter(
            x=[pA['date'], pB['date'], pC['date']], 
            y=[pA['price'], pB['price'], pC['price']], 
            mode='markers+text',
            text=[f"A<br>{pA['price']:.1f}", f"B<br>{pB['price']:.1f}", f"C<br>{pC['price']:.1f}"], 
            textposition=["bottom center", "top center", "bottom center"],
            textfont=dict(color="yellow", size=12, weight="bold"),
            marker=dict(size=10, color='yellow', symbol='diamond'),
            showlegend=False
        ))
        
        height_AB = pB['price'] - pA['price']
        
        # (B) 斐波那契拓展全家桶
        fib_levels = [
            (0.618, "gray", 1, "dot", "0.618"),
            (1.0, "gray", 1, "dash", "1.0 (AB=CD)"),
            (1.272, "gray", 1, "dot", "1.272"),
            (1.618, "#00FF00", 2, "solid", "🎯 1.618 Target"),
            (2.618, "gold", 2, "solid", "🚀 2.618 Target"),
            (3.618, "red", 1, "dot", "3.618"),
            (4.236, "red", 1, "dot", "4.236")
        ]
        
        last_date = df.index[-1]
        start_date = pC['date']
        future_date = last_date + timedelta(days=20) 
        
        for ratio, color, width, dash, label in fib_levels:
            lvl_price = pC['price'] + height_AB * ratio
            
            # 价格过滤
            if lvl_price > df['Low'].min() * 0.5 and lvl_price < df['High'].max() * 3:
                fig.add_shape(type="line", x0=start_date, y0=lvl_price, x1=future_date, y1=lvl_price,
                              line=dict(color=color, width=width, dash=dash))
                
                # 🟢 重点：在图上直接显示价格
                label_text = f"{label}: ${lvl_price:.2f}"
                
                fig.add_annotation(x=last_date, y=lvl_price, text=label_text, 
                                   showarrow=False, xanchor="left", yanchor="bottom",
                                   font=dict(color=color, size=11, family="Arial Black"),
                                   bgcolor="rgba(0,0,0,0.5)") # 加个背景色防止看不清

        # (C) 止损线
        fig.add_shape(type="line", x0=pA['date'], y0=pA['price'], x1=future_date, y1=pA['price'],
                      line=dict(color="red", width=2, dash="dot"))
        fig.add_annotation(x=pA['date'], y=pA['price'], text=f"⛔ STOP: ${pA['price']:.2f}", 
                           showarrow=True, arrowcolor="red", ax=0, ay=20)

    # 5. 动态止损线
    if 'stop_loss_atr' in res:
        fig.add_hline(y=res['stop_loss_atr'], line_color="#FF4B4B", line_dash="dot", annotation_text="ATR Stop")

    # 6. 默认缩放 3个月
    default_start_date = df.index[-1] - timedelta(days=90)
    
    fig.update_layout(
        template="plotly_dark", 
        height=height, 
        margin=dict(l=0,r=120,t=30,b=0), # 右侧留更多白给价格标签
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        dragmode='pan',
        xaxis=dict(
            range=[default_start_date, df.index[-1] + timedelta(days=10)], 
            type="date"
        ),
        yaxis=dict(fixedrange=False)
    )
    
    # 隐藏周末
    if len(df) > 2:
        diff = df.index[1] - df.index[0]
        if diff.days >= 1:
            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
            
    return fig

# ==============================================================================
# 4. 分析逻辑 (Controller)
# ==============================================================================
# ==============================================================================
# 4. 分析逻辑 (Controller) - 修复版
# ==============================================================================
def analyze_ticker_pro(ticker, interval="1d", lookback="5y", threshold=0.06):
    try:
        # 🟢 修正 1: 改用 Ticker 对象下载，解决多线程数据冲突/重复问题
        stock = yf.Ticker(ticker)
        
        # 处理时间映射
        real_period = lookback
        if interval in ["5m", "15m"]: real_period = "60d"
        elif interval == "1h": real_period = "1y"
        
        # 获取历史数据
        df = stock.history(period=real_period, interval=interval)
        
        # 🟢 修正 2: 数据清洗增强
        if df.empty or len(df) < 30: return None
        
        # 移除时区信息 (Plotly 有时会因为时区报错)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        # 统一列名 (yf.Ticker 返回的是 Title Case: Open, High...)
        # 确保不需要处理 MultiIndex，因为 .history() 返回的是单层索引
        
        # 2. 计算指标
        df = calculate_advanced_indicators(df)
        
        current_price = df['Close'].iloc[-1]
        current_rsi = df['RSI'].iloc[-1]
        current_atr = df['ATR'].iloc[-1]
        
        # 3. 寻找结构
        # (A) 趋势线
        lb_trend = 300 if interval in ["5m", "15m"] else 1000
        trend_res = get_resistance_trendline(df, lookback=lb_trend)
        
        # (B) ABC 结构
        abc_res = None
        pivots_df = get_swing_pivots_high_low(df, threshold=threshold)
        
        if len(pivots_df) >= 3:
            for i in range(len(pivots_df)-3, -1, -1):
                pA, pB, pC = pivots_df.iloc[i], pivots_df.iloc[i+1], pivots_df.iloc[i+2]
                if pA['type'] == -1 and pB['type'] == 1 and pC['type'] == -1:
                    if pC['price'] > pA['price'] and pB['price'] > pA['price']:
                        height = pB['price'] - pA['price']
                        target = pC['price'] + height * 1.618
                        abc_res = {'pivots': (pA, pB, pC), 'target': target}
                        break 

        # 4. 信号判定
        signal = "WAIT"
        signal_color = "gray"
        reasons = []
        
        is_breakout = trend_res and trend_res['breakout']
        
        if is_breakout:
            signal = "🔥 趋势线突破"
            signal_color = "#00FFFF"
            reasons.append("长期下降趋势线被突破")
            
        if abc_res:
            # 这里的逻辑稍微放宽，只要有结构就算，具体是否买入由人判断
            # 也可以加一个判定：价格是否在C点上方
            if current_price > abc_res['pivots'][2]['price']:
                if "突破" in signal:
                    signal = "🚀 双重共振买点"
                else:
                    signal = "🟢 ABC 结构确立"
                    signal_color = "#00FF00"
                reasons.append(f"回踩 C 点确认")

        stop_loss_atr = current_price - (2.0 * current_atr)
        option_plan = None
        if "突破" in signal or "ABC" in signal:
            option_plan = generate_option_plan(ticker, current_price, signal, current_rsi)

        return {
            "ticker": ticker,
            "price": current_price,
            "signal": signal,
            "color": signal_color,
            "reasons": ", ".join(reasons),
            "rsi": current_rsi,
            "atr": current_atr,
            "stop_loss_atr": stop_loss_atr,
            "trend": trend_res,
            "abc": abc_res,
            "data": df,
            "option_plan": option_plan
        }

    except Exception as e:
        # print(f"Error analyzing {ticker}: {e}") # 调试用
        return None
        


  

# ==============================================================================
# 5. UI 主程序
# ==============================================================================
st.sidebar.header("🕹️ 首席风控官设置")

account_size = st.sidebar.number_input("账户总资金 ($)", value=10000, step=1000)
risk_per_trade_pct = st.sidebar.slider("单笔风险 (%)", 0.5, 5.0, 2.0, 0.5) / 100

st.sidebar.markdown("---")
mode = st.sidebar.radio("作战模式:", ["🔍 单股狙击 (Live)", "🚀 市场全境扫描 (Hot 50)"])

HOT_STOCKS = ["TSLA", "NVDA", "PLTR", "MSTR", "COIN", "AMD", "META", "AMZN", "GOOG", "MSFT", "AAPL", "MARA", "RIOT", "CLSK", "NFLX"]

if mode == "🔍 单股狙击 (Live)":
    st.title("🛡️ 狗蛋风控指挥舱 (Precision)")
    
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        ticker = st.text_input("代码", value="TSLA").upper()
    with c2:
        lookback = st.selectbox("数据分析回溯", ["2y", "5y", "10y"], index=1)
    with c3:
        threshold_days = st.slider("结构灵敏度", 0.03, 0.15, 0.08, 0.01)

    with st.spinner(f"正在深度分析 {ticker} (High/Low Precision)..."):
        res = analyze_ticker_pro(ticker, interval="1d", lookback=lookback, threshold=threshold_days)
        
        if res:
            m1, m2, m3 = st.columns(3)
            m1.metric("当前价格", f"${res['price']:.2f}", delta=f"{res['signal']}")
            m2.metric("ATR 波动", f"{res['atr']:.2f}")
            m3.metric("RSI 情绪", f"{res['rsi']:.1f}")

            st.markdown(f"""
            <div style="background-color: #262730; padding: 15px; border-radius: 10px; border-left: 10px solid {res['color']}; margin-bottom: 20px;">
                <h3 style="color: {res['color']}; margin:0;">{res['signal']}</h3>
                <p style="color: #ccc; margin:0;">触发逻辑: {res['reasons']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if "ABC" in res['signal'] or "突破" in res['signal']:
                qty = calculate_position_size(account_size, risk_per_trade_pct, res['price'], res['stop_loss_atr'])
                st.success(f"🎯 **买入建议:** {qty} 股 (止损: ${res['stop_loss_atr']:.2f})")

            # 绘图
            fig = plot_chart(res['data'], res, height=600)
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})

            # 🟢 重点：在图表下方显示具体的点位价格表
            if res['abc']:
                pA, pB, pC = res['abc']['pivots']
                height_AB = pB['price'] - pA['price']
                
                # 计算关键点位列表
                levels_data = []
                # 基础点
                levels_data.append({"Level": "⛔ Stop Loss (A点)", "Price": pA['price'], "Type": "Risk"})
                levels_data.append({"Level": "🔵 Entry Support (C点)", "Price": pC['price'], "Type": "Entry"})
                # 斐波那契位
                fib_ratios = [0.618, 1.0, 1.272, 1.618, 2.0, 2.618, 3.618]
                for r in fib_ratios:
                    price = pC['price'] + height_AB * r
                    note = "🎯 TP1" if r==1.618 else "🚀 TP2" if r==2.618 else ""
                    levels_data.append({"Level": f"Fib {r} {note}", "Price": price, "Type": "Target"})
                
                df_levels = pd.DataFrame(levels_data)
                
                st.markdown("### 🔢 关键交易点位清单 (Key Levels)")
                st.dataframe(
                    df_levels.style.format({"Price": "${:.2f}"}),
                    use_container_width=True
                )

            if res['option_plan']:
                with st.expander("⚡ 查看期权建议", expanded=True):
                    p = res['option_plan']
                    st.info(f"**{p['name']}**: {p['legs']} | {p['logic']}")
        else:
            st.error("数据获取失败。")

else:
    st.title("🚀 市场全境扫描 (Hot 50)")
    tickers_input = st.text_area("监控列表", value=", ".join(HOT_STOCKS), height=100)
    
    if st.button("⚡ 开始扫描"):
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        progress_bar = st.progress(0)
        results = []
        
        def scan_one(t):
            return analyze_ticker_pro(t, interval="1d", lookback="5y", threshold=0.08)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(scan_one, t): t for t in tickers}
            for i, future in enumerate(futures):
                r = future.result()
                if r and ("ABC" in r['signal'] or "突破" in r['signal']):
                    results.append(r)
                progress_bar.progress((i + 1) / len(tickers))
        
        progress_bar.empty()
        
        if results:
            st.success(f"发现 {len(results)} 个机会")
            for i, r in enumerate(results):
                with st.expander(f"{r['ticker']} | ${r['price']:.2f} | {r['signal']}", expanded=False):
                    st.write(f"逻辑: {r['reasons']}")
                    
                    # 扫描模式也加上价格清单
                    if r['abc']:
                        pA, pB, pC = r['abc']['pivots']
                        h = pB['price'] - pA['price']
                        t1 = pC['price'] + h * 1.618
                        st.code(f"止损(A): ${pA['price']:.2f} | 目标(1.618): ${t1:.2f}")
                    
                    fig = plot_chart(r['data'], r, height=400)
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{i}", config={'scrollZoom': True})
        else:
            st.warning("暂无信号")

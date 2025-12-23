import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import argrelextrema
from concurrent.futures import ThreadPoolExecutor

# ==============================================================================
# 1. 页面配置与样式 (UI Configuration)
# ==============================================================================
st.set_page_config(page_title="Quant Sniper Pro (Final Ver)", layout="wide", page_icon="⚡")

st.markdown("""
<style>
    .metric-card { background-color: #1e1e1e; border: 1px solid #333; padding: 15px; border-radius: 8px; text-align: center; }
    .risk-alert { color: #ff4b4b; font-weight: bold; }
    .safe-zone { color: #00ff00; font-weight: bold; }
    /* 调整 Expander 样式 */
    .streamlit-expanderHeader { font-size: 16px; font-weight: bold; color: #e0e0e0; }
    /* 调整 Toast */
    .stToast { background-color: #333; color: white; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. 核心数学与指标库 (Core Engines)
# ==============================================================================

def calculate_advanced_indicators(df):
    """ 计算 TTM Squeeze, OBV, EMA, RSI, ATR """
    # 1. EMA 趋势系统
    df['EMA_8'] = df['Close'].ewm(span=8, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    
    # 2. OBV 资金流
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    # 3. RSI 情绪指标
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 4. ATR 波动率 (用于止损)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['ATR'] = np.max(ranges, axis=1).rolling(window=14).mean()
    
    # 5. TTM Squeeze (波动率挤压)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['BB_std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['SMA_20'] + (2.0 * df['BB_std'])
    df['BB_Lower'] = df['SMA_20'] - (2.0 * df['BB_std'])
    
    if 'ATR' in df.columns:
        df['KC_Upper'] = df['SMA_20'] + (1.5 * df['ATR'])
        df['KC_Lower'] = df['SMA_20'] - (1.5 * df['ATR'])
        df['Squeeze_On'] = (df['BB_Upper'] < df['KC_Upper']) & (df['BB_Lower'] > df['KC_Lower'])
    else:
        df['Squeeze_On'] = False

    return df

def calculate_position_size(account_balance, risk_pct, entry_price, stop_loss):
    """ 凯利公式简化版：仓位计算器 """
    if entry_price <= stop_loss: return 0
    risk_per_share = entry_price - stop_loss
    total_risk_allowance = account_balance * risk_pct
    position_size = int(total_risk_allowance / risk_per_share)
    return position_size

def get_swing_pivots(series, threshold=0.06):
    """ ZigZag 结构寻找 """
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

def get_resistance_trendline(df, lookback=150):
    """ 强力趋势线拟合 (Scipy) """
    highs = df['High'].values
    if len(highs) < 30: return None
    
    real_lookback = min(lookback, len(highs))
    start_idx = len(highs) - real_lookback
    subset_highs = highs[start_idx:]
    global_offset = start_idx

    peak_indexes = argrelextrema(subset_highs, np.greater, order=3)[0]
    if len(peak_indexes) < 2: return None

    best_line = None
    max_score = -float('inf')
    
    sorted_peaks = sorted(peak_indexes, key=lambda i: subset_highs[i], reverse=True)
    potential_start_points = sorted_peaks[:3] 

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
                tolerance = actual_price * 0.015
                
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

def generate_option_plan(ticker, current_price, signal_type, rsi, expiry_hint="短期"):
    import math
    plan = {}
    strike_buy = math.ceil(current_price)
    
    if "BREAKOUT" in signal_type or "ENTRY" in signal_type:
        if rsi > 70:
            plan['name'] = "⚠️ 风险警示 (RSI过热)"
            plan['strategy'] = "Debit Call Spread"
            plan['legs'] = f"买 ${strike_buy} / 卖 ${strike_buy+5} Call"
            plan['logic'] = "趋势向上但情绪过热，防止回调杀估值。"
        else:
            plan['name'] = "🚀 狙击 Call"
            plan['strategy'] = "Long Call"
            plan['legs'] = f"买入 Strike ${strike_buy} Call"
            plan['logic'] = "量价配合完美，动能充足，单腿买入博Gamma。"
        plan['expiry'] = expiry_hint
    return plan

# ==============================================================================
# 3. 核心绘图系统 (Visual Engine with Fibonacci)
# ==============================================================================
def plot_chart(df, res, height=600):
    fig = go.Figure()
    
    # 1. K 线
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], 
        name='Price'
    ))
    
    # 2. EMA 均线
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_8'], line=dict(color='orange', width=1), name="EMA 8"))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_21'], line=dict(color='purple', width=1), name="EMA 21"))
    
    # 3. 趋势线
    if res['trend']:
        tr = res['trend']
        fig.add_trace(go.Scatter(
            x=[tr['x1'], tr['x2']], y=[tr['y1'], tr['y2']], 
            mode='lines', name='Trendline', line=dict(color='cyan', width=2, dash='solid')
        ))

    # 4. 🔥 斐波那契战术地图 (Fibonacci & Structure) - 修复版
    if res['abc']:
        pA, pB, pC = res['abc']['pivots']
        
        # (A) 黄色虚线路径 A->B->C
        fig.add_trace(go.Scatter(
            x=[pA['date'], pB['date'], pC['date']], 
            y=[pA['price'], pB['price'], pC['price']], 
            mode='lines+markers', name='ABC Structure', 
            line=dict(color='yellow', width=2, dash='dash'),
            marker=dict(size=8, symbol='circle-open')
        ))
        
        # 计算高度与扩展位
        height_AB = pB['price'] - pA['price']
        
        # 定义斐波那契扩展位列表: (Ratio, Color, LineWidth, DashStyle, Label)
        fib_levels = [
            (0.618, "gray", 1, "dot", "Fib 0.618"),
            (1.0, "gray", 1, "dash", "Fib 1.0 (AB=CD)"),
            (1.272, "gray", 1, "dot", "Fib 1.272"),
            (1.618, "#00FF00", 2, "solid", "🎯 Fib 1.618 (TP1)"),
            (2.0, "gray", 1, "dot", "Fib 2.0"),
            (2.618, "gold", 2, "solid", "🚀 Fib 2.618 (TP2)"),
            (3.618, "red", 1, "dot", "Fib 3.618 (Max)")
        ]
        
        last_date = df.index[-1]
        start_date = pC['date']
        
        # (B) 循环绘制所有 Fib 线
        for ratio, color, width, style, label in fib_levels:
            lvl_price = pC['price'] + height_AB * ratio
            
            # 画线
            fig.add_shape(type="line", x0=start_date, y0=lvl_price, x1=last_date, y1=lvl_price,
                          line=dict(color=color, width=width, dash=style))
            # 画标签
            fig.add_annotation(x=last_date, y=lvl_price, text=label, 
                               showarrow=False, xanchor="left", yanchor="middle",
                               font=dict(color=color, size=10))

        # (C) 止损位 (Stop at A)
        fig.add_shape(type="line", x0=pA['date'], y0=pA['price'], x1=last_date, y1=pA['price'],
                      line=dict(color="red", width=1, dash="dot"))
        fig.add_annotation(x=pA['date'], y=pA['price'], text="STOP (A)", showarrow=True, arrowcolor="red", ax=0, ay=20)

    # 5. 动态止损线 (ATR)
    if 'stop_loss_atr' in res:
        fig.add_hline(y=res['stop_loss_atr'], line_color="#FF4B4B", line_dash="dot", annotation_text="ATR Stop")

    # 6. 布局优化 (加入 scrollZoom)
    fig.update_layout(
        template="plotly_dark", 
        height=height, 
        margin=dict(l=0,r=100,t=30,b=0), # 右侧留白给Fib标签
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        dragmode='pan' # 默认拖拽模式为平移，更适合触摸屏
    )
    
    # 隐藏周末 (仅日线)
    if len(df) > 2:
        diff = df.index[1] - df.index[0]
        if diff.days >= 1:
            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
            
    return fig

# ==============================================================================
# 4. 核心分析逻辑 (Brain)
# ==============================================================================
def analyze_ticker_pro(ticker, interval="1d", lookback="3mo", threshold=0.06):
    try:
        # 1. 数据下载
        real_period = lookback
        if interval in ["5m", "15m"]: real_period = "60d"
        elif interval == "1h": real_period = "1y"
        
        df = yf.download(ticker, period=real_period, interval=interval, progress=False, auto_adjust=False)
        
        if isinstance(df.columns, pd.MultiIndex):
            try: df.columns = df.columns.get_level_values(0)
            except: pass
        if len(df) < 30: return None
        if not isinstance(df.index, pd.DatetimeIndex): df.index = pd.to_datetime(df.index)
        
        # 2. 指标计算
        df = calculate_advanced_indicators(df)
        
        current_price = df['Close'].iloc[-1]
        current_rsi = df['RSI'].iloc[-1]
        current_atr = df['ATR'].iloc[-1]
        
        # 3. 模型运算
        # (A) 趋势线
        lb_trend = 300 if interval in ["5m", "15m"] else 150
        trend_res = get_resistance_trendline(df, lookback=lb_trend)
        
        # (B) ABC 结构
        abc_res = None
        # 如果是扫描模式，为了速度，ABC阈值固定；单股模式用传入的 threshold
        pivots_df = get_swing_pivots(df['Close'], threshold=threshold)
        if len(pivots_df) >= 3:
            # 简单寻找最近的一个有效 ABC
            for i in range(len(pivots_df)-3, len(pivots_df)-2):
                pA, pB, pC = pivots_df.iloc[i], pivots_df.iloc[i+1], pivots_df.iloc[i+2]
                if (pA['type'] == -1 and pB['type'] == 1 and pC['type'] == -1) and \
                   (pB['price'] > pA['price'] and pC['price'] > pA['price']):
                    height = pB['price'] - pA['price']
                    target = pC['price'] + height * 1.618
                    abc_res = {'pivots': (pA, pB, pC), 'target': target}

        # 4. 信号判定
        signal = "WAIT"
        signal_color = "gray"
        reasons = []
        
        is_breakout = trend_res and trend_res['breakout']
        is_squeeze_firing = (df['Squeeze_On'].iloc[-2] and not df['Squeeze_On'].iloc[-1])
        ema_bullish = df['EMA_8'].iloc[-1] > df['EMA_21'].iloc[-1]
        
        if is_breakout:
            if not ema_bullish:
                signal = "⚠️ 逆势突破"
                signal_color = "#FFA500"
                reasons.append("EMA空头排列")
            elif current_rsi > 75:
                signal = "⚠️ 超买突破"
                signal_color = "#FFFF00"
                reasons.append(f"RSI={current_rsi:.0f} 过热")
            else:
                signal = "🔥 SNIPER BREAKOUT"
                signal_color = "#00FFFF"
                reasons.append("趋势突破 + 均线多头")
                if is_squeeze_firing: reasons.append("Squeeze 爆发")
        
        # 5. 风控与期权
        stop_loss_atr = current_price - (2.0 * current_atr)
        
        option_plan = None
        if "SNIPER" in signal:
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
            "option_plan": option_plan,
            "ema_bullish": ema_bullish,
            "squeeze": "FIRING" if is_squeeze_firing else "ON" if df['Squeeze_On'].iloc[-1] else "OFF"
        }

    except Exception:
        return None

# ==============================================================================
# 5. UI 主程序 (Dashboard)
# ==============================================================================
st.sidebar.header("🕹️ 首席风控官设置")

# 资金管理
st.sidebar.markdown("### 💰 资金管理")
account_size = st.sidebar.number_input("账户总资金 ($)", value=10000, step=1000)
risk_per_trade_pct = st.sidebar.slider("单笔风险 (%)", 0.5, 5.0, 2.0, 0.5) / 100

st.sidebar.markdown("---")
mode = st.sidebar.radio("作战模式:", ["🔍 单股狙击 (Live)", "🚀 市场全境扫描 (Hot 50)"])

# 热门股池
HOT_STOCKS = [
    "TSLA", "NVDA", "PLTR", "MSTR", "COIN", "AMD", "META", "AMZN", "GOOG", "MSFT", "AAPL", 
    "MARA", "RIOT", "CLSK", "UPST", "AFRM", "SOFI", "AI", "SMCI", "AVGO", "TSM", 
    "NFLX", "CRM", "UBER", "ABNB", "HOOD", "DKNG", "RBLX", "NET", "CRWD", "PANW", 
    "GME", "AMC", "SPCE", "RIVN", "LCID", "NIO", "XPEV", "BABA", "PDD", "JD", 
    "TQQQ", "SOXL", "FNGU", "BITX"
]

if mode == "🔍 单股狙击 (Live)":
    st.title("🛡️ 狗蛋风控指挥舱 (Sniper Mode)")
    
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        ticker = st.text_input("代码 (Ticker)", value="TSLA").upper()
    with c2:
        interval = st.selectbox("K线周期", ["1d", "1h", "15m", "5m"], index=0)
    with c3:
        threshold_days = st.slider("结构灵敏度", 0.03, 0.12, 0.06, 0.01)

    with st.spinner(f"正在分析 {ticker} ..."):
        # 将输入参数传入
        res = analyze_ticker_pro(ticker, interval=interval, threshold=threshold_days)
        
        if res:
            # 1. 核心指标
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("价格", f"${res['price']:.2f}", delta=f"{res['signal']}")
            m2.metric("RSI (情绪)", f"{res['rsi']:.1f}", delta_color="inverse")
            m3.metric("ATR (波动)", f"{res['atr']:.2f}")
            m4.metric("EMA趋势", "🟢 多头" if res['ema_bullish'] else "🔴 空头")

            # 2. 信号横幅
            st.markdown(f"""
            <div style="background-color: #262730; padding: 15px; border-radius: 10px; border-left: 10px solid {res['color']}; margin-bottom: 20px;">
                <h3 style="color: {res['color']}; margin:0;">{res['signal']}</h3>
                <p style="color: #ccc; margin:0;">逻辑: {res['reasons']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # 3. 仓位建议
            if "SNIPER" in res['signal']:
                qty = calculate_position_size(account_size, risk_per_trade_pct, res['price'], res['stop_loss_atr'])
                st.success(f"🎯 **风控指令:** 建议买入 **{qty}** 股 (基于 {risk_per_trade_pct*100}% 风险，止损 ${res['stop_loss_atr']:.2f})")

            # 4. 强力绘图 (带完整斐波那契 + 缩放开启)
            fig = plot_chart(res['data'], res, height=600)
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})

            # 5. 期权战术
            if res['option_plan']:
                with st.expander("⚡ 查看期权战术板", expanded=True):
                    p = res['option_plan']
                    st.info(f"**{p['name']}**: {p['legs']} | {p['logic']}")
        else:
            st.error("数据获取失败，请检查代码或网络。")

else:
    # 批量扫描模式
    st.title("🚀 市场全境扫描 (Hot 50)")
    
    col_scan1, col_scan2 = st.columns([3, 1])
    with col_scan1:
        tickers_input = st.text_area("监控列表", value=", ".join(HOT_STOCKS), height=100)
    with col_scan2:
        st.write("")
        st.write("")
        start_scan = st.button("⚡ 开始全网扫描", type="primary")

    if start_scan:
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 线程池并发扫描
        def scan_one(t):
            return analyze_ticker_pro(t, interval="1d", lookback="6mo")

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(scan_one, t): t for t in tickers}
            for i, future in enumerate(futures):
                r = future.result()
                if r and ("SNIPER" in r['signal'] or "BREAKOUT" in r['signal']):
                    results.append(r)
                
                progress = (i + 1) / len(tickers)
                progress_bar.progress(progress)
                status_text.text(f"Scanning: {futures[future]} ({i+1}/{len(tickers)})")
        
        progress_bar.empty()
        status_text.empty()
        
        if results:
            st.success(f"🎯 扫描完成！发现 {len(results)} 个潜在机会")
            
            # 使用 enumerate 生成唯一 Key，修复 DuplicateElementId 错误
            for i, r in enumerate(results):
                label = f"{r['ticker']} | ${r['price']:.2f} | {r['signal']} | RSI: {r['rsi']:.1f}"
                
                with st.expander(label, expanded=False):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("EMA 趋势", "🟢 多头" if r['ema_bullish'] else "🔴 空头")
                    c2.metric("ATR 波动", f"{r['atr']:.2f}")
                    c3.metric("Squeeze", r['squeeze'])
                    
                    st.write(f"**触发逻辑:** {r['reasons']}")
                    
                    # 复用强力绘图函数
                    fig = plot_chart(r['data'], r, height=400)
                    
                    # 🔴 关键修复：加入 key 参数 + 缩放开启
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{r['ticker']}_{i}", config={'scrollZoom': True, 'displayModeBar': True})
                    
                    if r['option_plan']:
                        st.caption(f"💡 期权建议: {r['option_plan']['legs']}")
        else:
            st.warning("本次扫描未发现高胜率信号，市场可能处于震荡期。")

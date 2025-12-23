import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from scipy.signal import argrelextrema
from concurrent.futures import ThreadPoolExecutor

# ==============================================================================
# 1. 页面配置与样式 (UI Configuration)
# ==============================================================================
st.set_page_config(page_title="Quant Sniper Pro (Auto-Refresh)", layout="wide", page_icon="⚡")

st.markdown("""
<style>
    .metric-card { background-color: #1e1e1e; border: 1px solid #333; padding: 15px; border-radius: 8px; text-align: center; }
    .risk-alert { color: #ff4b4b; font-weight: bold; }
    .safe-zone { color: #00ff00; font-weight: bold; }
    /* 调整 Expander 的样式 */
    .streamlit-expanderHeader { font-size: 16px; font-weight: bold; color: #e0e0e0; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. 核心数学与指标库 (保持不变，已优化性能)
# ==============================================================================

def calculate_advanced_indicators(df):
    """ 计算 TTM Squeeze, OBV, EMA """
    # EMA
    df['EMA_8'] = df['Close'].ewm(span=8, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    
    # OBV
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    # RSI & ATR
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
    
    # TTM Squeeze
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
    if entry_price <= stop_loss: return 0
    risk_per_share = entry_price - stop_loss
    total_risk_allowance = account_balance * risk_pct
    position_size = int(total_risk_allowance / risk_per_share)
    return position_size

def get_resistance_trendline(df, lookback=150):
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
    
    # 限制起点数量以提高批量扫描速度
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
                tolerance = actual_price * 0.015 # 稍微放宽容差
                
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
            plan['name'] = "⚠️ 风险警示"
            plan['strategy'] = "Debit Call Spread"
            plan['legs'] = f"买 ${strike_buy} / 卖 ${strike_buy+5} Call"
            plan['logic'] = "RSI过热，禁止裸买。通过价差降低成本。"
        else:
            plan['name'] = "🚀 狙击 Call"
            plan['strategy'] = "Long Call"
            plan['legs'] = f"买入 Strike ${strike_buy} Call"
            plan['logic'] = "趋势突破，动能强劲。"
        plan['expiry'] = expiry_hint
    return plan

# ==============================================================================
# 3. 核心分析逻辑 (支持并发与自动刷新)
# ==============================================================================
def analyze_ticker_pro(ticker, interval="1d", lookback="3mo"):
    try:
        # 1. 数据下载 (适配 yfinance)
        real_period = lookback
        if interval in ["5m", "15m"]: real_period = "60d"
        elif interval == "1h": real_period = "1y"
        
        df = yf.download(ticker, period=real_period, interval=interval, progress=False, auto_adjust=False)
        
        if isinstance(df.columns, pd.MultiIndex):
            try: df.columns = df.columns.get_level_values(0)
            except: pass
        if len(df) < 30: return None
        
        # 2. 指标计算
        df = calculate_advanced_indicators(df)
        
        current_price = df['Close'].iloc[-1]
        current_rsi = df['RSI'].iloc[-1]
        current_atr = df['ATR'].iloc[-1]
        
        # 3. 趋势线分析
        lb_trend = 300 if interval in ["5m", "15m"] else 150
        trend_res = get_resistance_trendline(df, lookback=lb_trend)
        
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
                reasons.append("EMA空头排列，假突破概率大")
            elif current_rsi > 75:
                signal = "⚠️ 超买突破"
                signal_color = "#FFFF00"
                reasons.append(f"RSI={current_rsi:.0f} 过热")
            else:
                signal = "🔥 SNIPER BREAKOUT"
                signal_color = "#00FFFF"
                reasons.append("趋势突破 + 均线多头")
                if is_squeeze_firing: reasons.append("Squeeze 爆发")
        
        # 动态止损
        stop_loss_atr = current_price - (2.0 * current_atr)
        
        # 期权建议
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
            "data": df,
            "option_plan": option_plan,
            "ema_bullish": ema_bullish,
            "squeeze": "FIRING" if is_squeeze_firing else "ON" if df['Squeeze_On'].iloc[-1] else "OFF"
        }

    except Exception as e:
        return None

# ==============================================================================
# 4. UI 界面逻辑
# ==============================================================================
st.sidebar.header("🕹️ 首席风控官设置")

# 资金管理
st.sidebar.markdown("### 💰 资金管理")
account_size = st.sidebar.number_input("账户总资金 ($)", value=10000, step=1000)
risk_per_trade_pct = st.sidebar.slider("单笔风险 (%)", 0.5, 5.0, 2.0, 0.5) / 100

st.sidebar.markdown("---")
mode = st.sidebar.radio("作战模式:", ["🔍 单股狙击 (自动刷新)", "🚀 全市场批量扫描 (Hot 50)"])

# ----------------- 热门股池 (Hardcoded for Speed) -----------------
HOT_STOCKS = [
    "TSLA", "NVDA", "PLTR", "MSTR", "COIN", "AMD", "META", "AMZN", "GOOG", "MSFT", "AAPL", 
    "MARA", "RIOT", "CLSK", "UPST", "AFRM", "SOFI", "AI", "SMCI", "AVGO", "TSM", 
    "NFLX", "CRM", "NOW", "SNOW", "DDOG", "UBER", "ABNB", "HOOD", "DKNG", "RBLX", 
    "NET", "CRWD", "PANW", "ZS", "GME", "AMC", "SPCE", "RIVN", "LCID", "NIO", 
    "XPEV", "BABA", "PDD", "JD", "BIDU", "TQQQ", "SOXL", "FNGU", "BITX"
]

if mode == "🔍 单股狙击 (自动刷新)":
    st.title("🛡️ 狗蛋风控指挥舱 (Live Sniper)")
    
    # 布局输入控件
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        # 这里的输入更改会自动触发 Rerun
        ticker = st.text_input("代码 (Ticker)", value="TSLA").upper()
    with c2:
        interval = st.selectbox("K线周期", ["1d", "1h", "15m", "5m"], index=0)
    with c3:
        # 滑块拖动也会自动 Rerun，无需按钮
        threshold = st.slider("趋势灵敏度", 100, 300, 150, 10, help="Lookback Days")

    # 直接执行分析，不等待按钮
    with st.spinner(f"正在分析 {ticker} ..."):
        # 将 threshold 映射为 lookback
        res = analyze_ticker_pro(ticker, interval=interval, lookback=f"{threshold}d" if interval=="1d" else "3mo")
        
        if res:
            # 1. 核心数据
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("价格", f"${res['price']:.2f}", delta=f"{res['signal']}")
            m2.metric("RSI (情绪)", f"{res['rsi']:.1f}", delta_color="inverse")
            m3.metric("ATR (波动)", f"{res['atr']:.2f}")
            m4.metric("EMA趋势", "🟢 多头" if res['ema_bullish'] else "🔴 空头")

            # 2. 信号横幅
            st.markdown(f"""
            <div style="background-color: #262730; padding: 15px; border-radius: 10px; border-left: 10px solid {res['color']}; margin-bottom: 20px;">
                <h3 style="color: {res['color']}; margin:0;">{res['signal']}</h3>
                <p style="color: #ccc; margin:0;">{res['reasons']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # 3. 仓位建议
            if "SNIPER" in res['signal']:
                qty = calculate_position_size(account_size, risk_per_trade_pct, res['price'], res['stop_loss_atr'])
                st.success(f"🎯 **风控指令:** 建议买入 **{qty}** 股 (基于 {risk_per_trade_pct*100}% 风险，止损 ${res['stop_loss_atr']:.2f})")

            # 4. 图表
            fig = go.Figure()
            df = res['data']
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
            
            # 画趋势线
            if res['trend']:
                tr = res['trend']
                fig.add_trace(go.Scatter(x=[tr['x1'], tr['x2']], y=[tr['y1'], tr['y2']], mode='lines', name='Resistance', line=dict(color='cyan', width=2)))
            
            # 画 EMA
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_8'], mode='lines', name='EMA 8', line=dict(color='orange', width=1)))
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_21'], mode='lines', name='EMA 21', line=dict(color='purple', width=1)))
            
            fig.update_layout(template="plotly_dark", height=550, margin=dict(l=0,r=0,t=30,b=0))
            if interval == "1d":
                fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
            st.plotly_chart(fig, use_container_width=True)

            # 5. 期权计划
            if res['option_plan']:
                with st.expander("⚡ 查看期权战术 (Option Plan)", expanded=True):
                    p = res['option_plan']
                    st.info(f"**{p['name']}**: {p['legs']} | {p['logic']}")
        else:
            st.error("数据获取失败，请检查代码拼写或网络。")

else:
    # 批量扫描模式
    st.title("🚀 市场全境扫描 (Hot 50)")
    
    col_scan1, col_scan2 = st.columns([3, 1])
    with col_scan1:
        # 默认填入热门股
        tickers_input = st.text_area("监控列表 (已预设 2025 热门股)", value=", ".join(HOT_STOCKS), height=100)
    with col_scan2:
        st.write("")
        st.write("")
        start_scan = st.button("⚡ 开始全网扫描", type="primary")

    if start_scan:
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 使用线程池并发扫描，提高速度
        def scan_one(t):
            return analyze_ticker_pro(t, interval="1d", lookback="6mo")

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(scan_one, t): t for t in tickers}
            for i, future in enumerate(futures):
                r = future.result()
                if r and ("SNIPER" in r['signal'] or "BREAKOUT" in r['signal']):
                    results.append(r)
                
                # 更新进度
                progress = (i + 1) / len(tickers)
                progress_bar.progress(progress)
                status_text.text(f"正在扫描: {futures[future]} ({i+1}/{len(tickers)})")
        
        progress_bar.empty()
        status_text.empty()
        
        if results:
            st.success(f"🎯 扫描完成！发现 {len(results)} 个潜在机会")
            
            # 遍历结果，生成可折叠的详细卡片
            if results:
            st.success(f"🎯 扫描完成！发现 {len(results)} 个潜在机会")
            
            # ⬇️⬇️⬇️ 这里的循环是改动点 ⬇️⬇️⬇️
            # 我们加上 enumerate(results) 来获取索引 i，确保 key 绝对唯一
            for i, r in enumerate(results):
                # 标题栏显示关键信息
                label = f"{r['ticker']} | ${r['price']:.2f} | {r['signal']} | RSI: {r['rsi']:.1f}"
                
                with st.expander(label, expanded=False):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("EMA 趋势", "🟢 多头" if r['ema_bullish'] else "🔴 空头")
                    c2.metric("ATR 波动", f"{r['atr']:.2f}")
                    c3.metric("Squeeze", r['squeeze'])
                    
                    st.write(f"**触发逻辑:** {r['reasons']}")
                    
                    # 绘制迷你图表
                    fig = go.Figure()
                    df = r['data'][-60:] # 只画最近60天
                    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
                    
                    if r['trend']:
                        tr = r['trend']
                        fig.add_trace(go.Scatter(x=[tr['x1'], tr['x2']], y=[tr['y1'], tr['y2']], mode='lines', line=dict(color='cyan')))
                        
                    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_8'], line=dict(color='orange', width=1), name="EMA8"))
                    
                    fig.update_layout(template="plotly_dark", height=350, margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False)
                    
                    # 🔴 关键修复：加入了 key 参数
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{r['ticker']}_{i}")
                    
                    if r['option_plan']:
                        st.caption(f"💡 期权建议: {r['option_plan']['legs']}")

        else:
            st.warning("本次扫描未发现高胜率信号，建议休息或调整监控列表。")

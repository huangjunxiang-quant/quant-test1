import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from datetime import datetime, timedelta
from scipy.signal import argrelextrema

# ==============================================================================
# 1. 页面配置与样式 (UI Configuration)
# ==============================================================================
st.set_page_config(page_title="Quant Sniper Pro (Risk Control Edition)", layout="wide", page_icon="⚡")

st.markdown("""
<style>
    .metric-card { background-color: #1e1e1e; border: 1px solid #333; padding: 15px; border-radius: 8px; text-align: center; }
    .risk-alert { color: #ff4b4b; font-weight: bold; }
    .safe-zone { color: #00ff00; font-weight: bold; }
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child { width: 300px; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. 核心数学与指标库 (Core Math & Indicators)
# ==============================================================================

# --- A. 基础指标计算 (RSI & ATR) ---
def calculate_indicators(df):
    # 1. RSI (相对强弱指标) - 14周期
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 2. ATR (平均真实波幅) - 用于动态止损
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    return df

# --- B. 资金管理计算器 (Money Management) ---
def calculate_position_size(account_balance, risk_pct, entry_price, stop_loss):
    """
    基于账户总风险计算仓位
    """
    if entry_price <= stop_loss: return 0
    risk_per_share = entry_price - stop_loss
    total_risk_allowance = account_balance * risk_pct
    # 向下取整，保守计算
    position_size = int(total_risk_allowance / risk_per_share)
    return position_size

# --- C. 寻找波段高低点 (ZigZag) ---
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

# --- D. 强力多点拟合下降趋势线 (Robust Multi-Touch Trendline) ---
def get_resistance_trendline(df, lookback=150):
    highs = df['High'].values
    if len(highs) < 30: return None
    
    # 动态调整 lookback，防止数据越界
    real_lookback = min(lookback, len(highs))
    start_idx = len(highs) - real_lookback
    subset_highs = highs[start_idx:]
    global_offset = start_idx

    peak_indexes = argrelextrema(subset_highs, np.greater, order=3)[0]
    if len(peak_indexes) < 2: return None

    best_line = None
    max_score = -float('inf')
    
    sorted_peaks = sorted(peak_indexes, key=lambda i: subset_highs[i], reverse=True)
    potential_start_points = sorted_peaks[:4] # 稍微放宽起点搜索范围

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
                # 动态容差：价格越高容差越大
                tolerance = actual_price * 0.01 
                
                if abs(actual_price - trend_price) < tolerance:
                    hits += 1
                elif actual_price > trend_price + tolerance:
                    violations += 1
            
            # 这里的打分机制倾向于惩罚突破(violations)，奖励触碰(hits)
            score = hits - (violations * 3) 
            if abs(slope) < (price_A * 0.05): score += 0.5

            if score > max_score:
                max_score = score
                best_line = {
                    'slope': slope,
                    'intercept': intercept,
                    'start_idx_rel': idx_A
                }

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

# --- E. 期权策略生成器 (含风控逻辑) ---
def generate_option_plan(ticker, current_price, target_price, signal_type, rsi, expiry_hint="短期"):
    import math
    plan = {}
    
    if "BREAKOUT" in signal_type or "ENTRY" in signal_type:
        strike_buy = math.ceil(current_price) 
        
        # 狗蛋风控逻辑：如果 RSI 过高，不建议裸买 Call
        if rsi > 70:
            plan['name'] = "⚠️ 风险提示"
            plan['strategy'] = "Wait / Debit Spread"
            plan['legs'] = "RSI过热，禁止裸买Call。考虑价差或观望。"
            plan['logic'] = "虽然突破，但超买严重，容易回撤。"
            plan['expiry'] = "观望"
        else:
            plan['name'] = "🚀 狙击模式 (Sniper)"
            plan['strategy'] = "Long Call"
            plan['legs'] = f"买入 Strike ${strike_buy} Call"
            plan['logic'] = "趋势突破且RSI健康，动能爆发。"
            plan['expiry'] = expiry_hint
            
    elif "ABC" in signal_type:
        strike_buy = math.floor(current_price) 
        strike_sell = math.floor(target_price) 
        plan['name'] = "🛡️ 结构战法 (Structure)"
        plan['strategy'] = "Bull Call Spread"
        plan['legs'] = f"买 ${strike_buy} / 卖 ${strike_sell} Call"
        plan['logic'] = "利用ABC结构，锁定盈亏比，规避波动率风险。"
        plan['expiry'] = "30天以上"
    else:
        return None
    
    return plan

# ==============================================================================
# 3. 核心分析逻辑 (支持盘中)
# ==============================================================================
def analyze_ticker_pro(ticker, interval="1d", lookback="3mo", threshold=0.06):
    try:
        # 1. 动态确定数据获取长度 (yfinance 限制)
        real_period = lookback
        if interval == "1m": real_period = "7d"
        elif interval in ["5m", "15m"]: real_period = "60d"
        elif interval == "1h": real_period = "730d" # yfinance max for 1h
        
        # 2. 获取数据
        df = yf.download(ticker, period=real_period, interval=interval, progress=False, auto_adjust=False)
        
        # 修复列名
        if isinstance(df.columns, pd.MultiIndex):
            try: df.columns = df.columns.get_level_values(0)
            except: pass
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        if len(df) < 30: 
            raise ValueError(f"数据不足 (仅 {len(df)} 行)，请检查代码或市场状态。")
        
        # 3. 计算风控指标
        df = calculate_indicators(df)
        
        current_price = df['Close'].iloc[-1]
        current_rsi = df['RSI'].iloc[-1]
        current_atr = df['ATR'].iloc[-1]
        
        # 4. 跑模型
        # (A) ABC 结构
        abc_res = None
        # 如果是盘中，减少 ABC 误判，提高阈值或仅在日线级别跑
        # 这里为了演示，仍然跑，但建议主要看日线
        pivots_df = get_swing_pivots(df['Close'], threshold=threshold)
        if len(pivots_df) >= 3:
            # 简化逻辑：取最后3个点
            for i in range(len(pivots_df)-3, len(pivots_df)-2):
                pA, pB, pC = pivots_df.iloc[i], pivots_df.iloc[i+1], pivots_df.iloc[i+2]
                if (pA['type'] == -1 and pB['type'] == 1 and pC['type'] == -1) and \
                   (pB['price'] > pA['price'] and pC['price'] > pA['price']):
                    wave_height = pB['price'] - pA['price']
                    target = pC['price'] + wave_height * 1.618
                    risk_dist = current_price - pA['price']
                    potential = target - current_price
                    rr = potential / risk_dist if risk_dist > 0 else 0
                    abc_res = {'pivots': (pA, pB, pC), 'target': target, 'stop': pA['price'], 'rr': rr}
        
        # (B) 趋势线
        # 盘中数据多，lookback 放大
        lb_trend = 300 if interval in ["5m", "15m"] else 150
        trend_res = get_resistance_trendline(df, lookback=lb_trend)
        
        # 5. 信号与风控判定
        signal = "WAIT"
        signal_color = "gray"
        reasons = []
        
        # 动态止损 (ATR Based) - 狗蛋核心风控
        # 如果是日线，2倍ATR；如果是盘中，1.5倍ATR
        atr_mult = 2.0 if interval == "1d" else 1.5
        stop_loss_atr = current_price - (atr_mult * current_atr)

        is_breakout = trend_res and trend_res['breakout']
        is_abc_buy = abc_res and abc_res['rr'] > 2.0 and current_price < abc_res['pivots'][1]['price']

        if is_breakout:
            if current_rsi > 75:
                signal = "⚠️ 假突破预警"
                signal_color = "#FFA500" # Orange
                reasons.append(f"RSI过热 ({current_rsi:.1f})，需回踩确认")
            elif current_atr < (current_price * 0.005):
                signal = "⚠️ 弱势突破"
                signal_color = "#FFFF00" # Yellow
                reasons.append("波动率过低，动能存疑")
            else:
                signal = "🔥 SNIPER BREAKOUT"
                signal_color = "#00FFFF" # Cyan
                reasons.append("放量突破 + 指标健康")
        elif is_abc_buy:
            signal = "🟢 BUY (ABC)"
            signal_color = "#00FF00"
            reasons.append("ABC结构确立，盈亏比优秀")
            
        # 6. 生成期权建议
        option_plan = None
        if "BUY" in signal or "SNIPER" in signal:
            tgt = abc_res['target'] if abc_res else current_price * 1.15
            # 盘中操作建议短期期权，日线建议长期
            exp = "本周/下周 (短期)" if interval in ["5m", "15m", "1h"] else "45天+"
            option_plan = generate_option_plan(ticker, current_price, tgt, signal, current_rsi, exp)

        return {
            "ticker": ticker,
            "price": current_price,
            "signal": signal,
            "color": signal_color,
            "reasons": ", ".join(reasons),
            "rsi": current_rsi,
            "atr": current_atr,
            "stop_loss_atr": stop_loss_atr,
            "abc": abc_res,
            "trend": trend_res,
            "data": df,
            "option_plan": option_plan
        }

    except Exception as e:
        st.error(f"分析失败: {str(e)}")
        return None

# ==============================================================================
# 4. UI 界面逻辑 (Dashboard)
# ==============================================================================
st.sidebar.header("🕹️ 首席风控官设置")

# 资金管理模块
st.sidebar.markdown("### 💰 资金管理 (Money Management)")
account_size = st.sidebar.number_input("账户总资金 ($)", value=10000, step=1000)
risk_per_trade_pct = st.sidebar.slider("单笔最大亏损 (%)", 0.5, 5.0, 2.0, 0.5) / 100

st.sidebar.markdown("---")
mode = st.sidebar.radio("功能模式:", ["🔍 单股狙击 (Sniper Mode)", "🚀 批量扫描 (Scanner)"])

if mode == "🔍 单股狙击 (Sniper Mode)":
    st.title("🛡️ 狗蛋风控指挥舱 (Risk Control Center)")
    
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        ticker = st.text_input("代码 (Ticker)", value="TSLA").upper()
    with c2:
        # 支持盘中周期
        interval = st.selectbox("K线周期", ["1d", "1h", "15m", "5m"], index=0)
    with c3:
        threshold = st.slider("结构灵敏度", 0.03, 0.10, 0.06, 0.01)

    if st.button("🚀 启动分析", type="primary"):
        with st.spinner(f"正在接入交易所数据 ({interval})..."):
            res = analyze_ticker_pro(ticker, interval=interval, threshold=threshold)
            
            if res:
                # ----------------- 核心指标区 -----------------
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("当前价格", f"${res['price']:.2f}")
                
                # RSI 颜色逻辑
                rsi_val = res['rsi']
                rsi_delta = "超买" if rsi_val > 70 else "超卖" if rsi_val < 30 else "正常"
                rsi_color = "normal" if 30 <= rsi_val <= 70 else "inverse"
                col_m2.metric("RSI (情绪)", f"{rsi_val:.1f}", delta=rsi_delta, delta_color=rsi_color)
                
                col_m3.metric("ATR (波动)", f"{res['atr']:.2f}")
                
                # 推荐止损
                col_m4.metric("🛡️ 推荐硬止损", f"${res['stop_loss_atr']:.2f}", delta="基于ATR")

                # ----------------- 信号区 -----------------
                st.markdown(f"""
                <div style="background-color: #262730; padding: 20px; border-radius: 10px; border-left: 10px solid {res['color']}; margin: 20px 0;">
                    <h2 style="color: {res['color']}; margin:0;">信号: {res['signal']}</h2>
                    <p style="color: #ccc; margin-top:5px; font-size: 16px;">逻辑: {res['reasons'] if res['reasons'] else '等待市场确认...'}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # ----------------- 资金仓位建议 (狗蛋核心) -----------------
                if "ENTRY" in res['signal'] or "BUY" in res['signal'] or "BREAKOUT" in res['signal']:
                    qty = calculate_position_size(account_size, risk_per_trade_pct, res['price'], res['stop_loss_atr'])
                    risk_amt = account_size * risk_per_trade_pct
                    
                    st.markdown("### 💰 首席风控官·仓位指令")
                    cc1, cc2 = st.columns([2, 1])
                    with cc1:
                        if qty > 0:
                            st.success(f"🎯 **建议最大仓位:** 正股 **{qty}** 股")
                            st.caption(f"*计算逻辑: 总资金 ${account_size} × 风险 {risk_per_trade_pct*100}% = 最大亏损 ${risk_amt:.0f}。单股止损距离 ${res['price'] - res['stop_loss_atr']:.2f}。*")
                        else:
                            st.error("❌ **禁止开仓:** 止损空间太窄或风险过大！")
                
                # ----------------- 图表区 -----------------
                fig = go.Figure()
                df = res['data']
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
                
                # 画 ABC
                if res['abc']:
                    pA, pB, pC = res['abc']['pivots']
                    fig.add_trace(go.Scatter(x=[pA['date'], pB['date'], pC['date']], y=[pA['price'], pB['price'], pC['price']], mode='lines+markers', name='Structure', line=dict(color='yellow', dash='dash')))
                    fig.add_hline(y=res['abc']['target'], line_color="green", annotation_text="Target")

                # 画趋势线
                if res['trend']:
                    tr = res['trend']
                    fig.add_trace(go.Scatter(x=[tr['x1'], tr['x2']], y=[tr['y1'], tr['y2']], mode='lines', name='Trendline', line=dict(color='cyan', width=2)))

                # 画动态止损线 (只画最后一段)
                fig.add_hline(y=res['stop_loss_atr'], line_color="#FF4B4B", line_dash="dot", annotation_text=f"Hard Stop ${res['stop_loss_atr']:.2f}")

                fig.update_layout(template="plotly_dark", height=600, title=f"{ticker} ({interval}) 风控分析图")
                
                # 如果是日线，隐藏周末空缺；分钟线暂时不隐藏以防报错
                if interval == "1d":
                    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
                
                st.plotly_chart(fig, use_container_width=True)

                # ----------------- 期权战术板 -----------------
                if res['option_plan']:
                    st.markdown("### ⚡ 期权作战计划")
                    plan = res['option_plan']
                    # 只有当信号是正向且非警告时，才显示绿色
                    color_call = "red" if "警告" in plan['name'] else "green"
                    
                    op_c1, op_c2 = st.columns(2)
                    with op_c1:
                         st.info(f"**策略:** {plan['strategy']}\n\n**战术:** {plan['name']}")
                    with op_c2:
                         st.markdown(f"""
                         <div style="padding:10px; border:1px solid {color_call}; border-radius:5px;">
                         <strong>腿 (Legs):</strong> {plan['legs']}<br>
                         <strong>到期 (Expiry):</strong> {plan['expiry']}<br>
                         <strong>逻辑:</strong> {plan['logic']}
                         </div>
                         """, unsafe_allow_html=True)

else:
    # 批量扫描模式 (简化版)
    st.title("🚀 市场全境扫描 (Scanner)")
    default_list = "TSLA, NVDA, AAPL, AMD, AMZN, GOOG, META, MSFT, COIN, MSTR, PLTR"
    tickers_input = st.text_area("监控列表 (逗号分隔)", value=default_list)
    
    if st.button("⚡ 开始扫描"):
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        results = []
        progress = st.progress(0)
        
        for i, t in enumerate(tickers):
            # 扫描模式默认用日线，速度快
            r = analyze_ticker_pro(t, interval="1d")
            if r:
                # 只有出现特定信号才加入列表
                if "BUY" in r['signal'] or "BREAKOUT" in r['signal']:
                    results.append({
                        "代码": t,
                        "价格": r['price'],
                        "信号": r['signal'],
                        "RSI": f"{r['rsi']:.1f}",
                        "ATR止损": f"${r['stop_loss_atr']:.2f}"
                    })
            progress.progress((i + 1) / len(tickers))
            time.sleep(0.1)
            
        if results:
            st.success(f"扫描完成，发现 {len(results)} 个潜在机会")
            st.dataframe(pd.DataFrame(results))
        else:
            st.info("扫描完成，暂无符合高胜率模型的标的。")

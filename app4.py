import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import argrelextrema
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

# ==============================================================================
# 1. 页面配置与样式
# ==============================================================================
st.set_page_config(page_title="Quant Sniper Pro (Fitting Control)", layout="wide", page_icon="⚡")

st.markdown("""
<style>
    .metric-card { background-color: #1e1e1e; border: 1px solid #333; padding: 15px; border-radius: 8px; text-align: center; }
    .stToast { background-color: #333; color: white; }
    [data-testid="stSidebar"] { background-color: #111; }
    [data-testid="stDataFrame"] { width: 100%; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. 核心数学算法 (你指定的 Scipy 拟合版 + 参数化)
# ==============================================================================

def get_swing_pivots_high_low(df, threshold=0.06):
    """ [精度版] ZigZag 算法 (High/Low) """
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

# --- 🟢 阻力趋势线 (Scipy 拟合版 - 使用你提供的逻辑) ---
def get_resistance_trendline(df, lookback=1000, order=5):
    # 1. 提取高点数据
    highs = df['High'].values
    if len(highs) < 30: return None
    
    real_lookback = min(lookback, len(highs))
    start_idx = len(highs) - real_lookback
    subset_highs = highs[start_idx:]
    global_offset = start_idx

    # 2. 识别所有的局部波峰 (Peaks)
    # 🟢 这里使用了传入的 order 参数，让你可以控制
    peak_indexes = argrelextrema(subset_highs, np.greater, order=order)[0]
    
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
                
                # 误差容忍度: 1.5% (稍微放宽一点适应5年长周期)
                tolerance = actual_price * 0.015 
                
                if abs(actual_price - trend_price) < tolerance:
                    hits += 1
                elif actual_price > trend_price + tolerance:
                    violations += 1
            
            # 评分公式 (保持你喜欢的逻辑)
            score = hits - (violations * 2)
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

# --- 🔴 支撑趋势线 (Scipy 拟合版 - 对称逻辑) ---
def get_support_trendline(df, lookback=1000, order=5):
    lows = df['Low'].values
    if len(lows) < 30: return None
    
    real_lookback = min(lookback, len(lows))
    start_idx = len(lows) - real_lookback
    subset_lows = lows[start_idx:]
    global_offset = start_idx

    # 使用 np.less 找波谷
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
                elif actual_price < trend_price * 0.985: violations += 1 # 跌破算violation
            
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
            'x1': df.index[idx_A_glob], 
            'y1': slope * idx_A_glob + global_intercept,
            'x2': df.index[last_idx], 
            'y2': trendline_price_now,
            'price_now': trendline_price_now,
            'breakdown': df['Close'].iloc[-1] < trendline_price_now
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
    
    if "突破" in signal_type or "双重共振" in signal_type or "ABC" in signal_type:
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
    
    # 3. 阻力线 (看多 - 青色)
    if res['trend_res']:
        tr = res['trend_res']
        fig.add_trace(go.Scatter(
            x=[tr['x1'], df.index[-1]], 
            y=[tr['y1'], tr['price_now']], 
            mode='lines', name='Resistance', line=dict(color='cyan', width=2)
        ))

    # 4. 支撑线 (看空 - 紫色)
    if res['trend_sup']:
        ts = res['trend_sup']
        fig.add_trace(go.Scatter(
            x=[ts['x1'], df.index[-1]], 
            y=[ts['y1'], ts['price_now']], 
            mode='lines', name='Support', line=dict(color='#FF00FF', width=2)
        ))

    # 5. 斐波那契战术地图
    if res['abc']:
        pA, pB, pC = res['abc']['pivots']
        
        # 连线
        fig.add_trace(go.Scatter(
            x=[pA['date'], pB['date'], pC['date']], 
            y=[pA['price'], pB['price'], pC['price']], 
            mode='lines', name='ABC', 
            line=dict(color='yellow', width=2, dash='dash')
        ))
        
        # 标注
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
        
        # 扩展位 + 价格标注
        fib_levels = [
            (0.618, "gray", 1, "dot", "0.618"),
            (1.0, "gray", 1, "dash", "1.0"),
            (1.618, "#00FF00", 2, "solid", "🎯 1.618"),
            (2.618, "gold", 2, "solid", "🚀 2.618")
        ]
        
        last_date = df.index[-1]
        start_date = pC['date']
        future_date = last_date + timedelta(days=20) 
        
        for ratio, color, width, dash, label in fib_levels:
            lvl_price = pC['price'] + height_AB * ratio
            if lvl_price > df['Low'].min() * 0.5 and lvl_price < df['High'].max() * 3:
                fig.add_shape(type="line", x0=start_date, y0=lvl_price, x1=future_date, y1=lvl_price,
                              line=dict(color=color, width=width, dash=dash))
                
                label_text = f"{label}: ${lvl_price:.2f}"
                fig.add_annotation(x=last_date, y=lvl_price, text=label_text, 
                                   showarrow=False, xanchor="left", yanchor="bottom",
                                   font=dict(color=color, size=11, family="Arial Black"),
                                   bgcolor="rgba(0,0,0,0.5)")

    # 6. 默认缩放
    default_start_date = df.index[-1] - timedelta(days=90)
    fig.update_layout(
        template="plotly_dark", height=height, margin=dict(l=0,r=100,t=30,b=0),
        xaxis_rangeslider_visible=False, hovermode="x unified", dragmode='pan',
        xaxis=dict(range=[default_start_date, df.index[-1] + timedelta(days=10)], type="

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from concurrent.futures import ThreadPoolExecutor

# ==============================================================================
# 1. 页面配置 (UX 升级)
# ==============================================================================
st.set_page_config(page_title="Quant Sniper Pro (UX Enhanced)", layout="wide", page_icon="⚡")

st.markdown("""
<style>
    .metric-card { background-color: #1e1e1e; border: 1px solid #333; padding: 15px; border-radius: 8px; text-align: center; }
    /* 优化侧边栏样式 */
    [data-testid="stSidebar"] { background-color: #181818; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. 核心算法库 (Core Logic)
# ==============================================================================

def calculate_indicators(df):
    """ 计算基础指标: EMA, RSI, ATR """
    df['EMA_8'] = df['Close'].ewm(span=8, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    
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

def get_swing_pivots(series, threshold_pct=0.05):
    """ 
    ZigZag 核心算法 (支持灵敏度调节) 
    threshold_pct: 波动阈值，0.05 代表 5% 的波动才算一个拐点
    """
    pivots = []
    last_pivot_price = series.iloc[0]
    last_pivot_date = series.index[0]
    last_pivot_type = 0 # 0=init, 1=low, -1=high
    
    temp_extreme_price = series.iloc[0]
    temp_extreme_date = series.index[0]
    
    for date, price in series.items():
        if last_pivot_type == 0:
            if price > last_pivot_price * (1 + threshold_pct):
                last_pivot_type = -1 # High
                pivots.append({'date': last_pivot_date, 'price': last_pivot_price, 'type': -1}) # 记录前一个Low? 不，这里简化逻辑
                # 重新初始化
                last_pivot_price = price
                last_pivot_date = date
                pivots = [{'date': date, 'price': price, 'type': -1}] # 第一个点确立
                temp_extreme_price = price
            elif price < last_pivot_price * (1 - threshold_pct):
                last_pivot_type = 1 # Low
                last_pivot_price = price
                last_pivot_date = date
                pivots = [{'date': date, 'price': price, 'type': 1}]
                temp_extreme_price = price
                
        elif last_pivot_type == -1: # 当前寻找 Low
            if price > temp_extreme_price: # 更高的高点
                temp_extreme_price = price
                temp_extreme_date = date
                # 更新当前的高点
                pivots[-1] = {'date': date, 'price': price, 'type': -1}
            elif price < temp_extreme_price * (1 - threshold_pct): # 跌破阈值，确认 High，开始找 Low
                pivots.append({'date': temp_extreme_date, 'price': temp_extreme_price, 'type': -1}) # 确保High被锁定(虽然已经更新过)
                # 这里的逻辑需要修正：上面的 elif 更新了最后一个点，这里应该 append 新的 Low 候选
                # 简单写法：
                pivots.append({'date': date, 'price': price, 'type': 1})
                last_pivot_type = 1
                temp_extreme_price = price
                temp_extreme_date = date

        elif last_pivot_type == 1: # 当前寻找 High
            if price < temp_extreme_price: # 更低的低点
                temp_extreme_price = price
                temp_extreme_date = date
                pivots[-1] = {'date': date, 'price': price, 'type': 1}
            elif price > temp_extreme_price * (1 + threshold_pct):
                pivots.append({'date': date, 'price': price, 'type': -1})
                last_pivot_type = -1
                temp_extreme_price = price
                temp_extreme_date = date

    return pd.DataFrame(pivots)

def get_swing_pivots_simple(series, threshold_pct=0.05):
    """ 更稳定的 ZigZag 实现 """
    ut = 1 + threshold_pct
    dt = 1 - threshold_pct
    
    pivots = pd.Series(0, index=series.index)
    last_pivot = series.iloc[0]
    trend = 0 # 1 up, -1 down
    
    peak_val = last_pivot
    peak_date = series.index[0]
    trough_val = last_pivot
    trough_date = series.index[0]
    
    pivot_list = []
    
    for date, price in series.items():
        if trend == 0:
            if price > last_pivot * ut:
                trend = 1
                peak_val = price; peak_date = date
                trough_val = last_pivot; trough_date = series.index[0] # 假设起点是低点
                pivot_list.append({'date': trough_date, 'price': trough_val, 'type': 1})
            elif price < last_pivot * dt:
                trend = -1
                trough_val = price; trough_date = date
                peak_val = last_pivot; peak_date = series.index[0]
                pivot_list.append({'date': peak_date, 'price': peak_val, 'type': -1})
        
        elif trend == 1: # 上升趋势中
            if price > peak_val:
                peak_val = price
                peak_date = date
            elif price < peak_val * dt: # 回调确认，高点成立
                pivot_list.append({'date': peak_date, 'price': peak_val, 'type': -1})
                trend = -1
                trough_val = price
                trough_date = date
                
        elif trend == -1: # 下降趋势中
            if price < trough_val:
                trough_val = price
                trough_date = date
            elif price > trough_val * ut: # 反弹确认，低点成立
                pivot_list.append({'date': trough_date, 'price': trough_val, 'type': 1})
                trend = 1
                peak_val = price
                peak_date = date
                
    # 加上最后一个极值点
    if trend == 1:
        pivot_list.append({'date': peak_date, 'price': peak_val, 'type': -1})
    else:
        pivot_list.append({'date': trough_date, 'price': trough_val, 'type': 1})
        
    return pd.DataFrame(pivot_list)

# ==============================================================================
# 3. 绘图引擎 (UX 核心升级)
# ==============================================================================
def plot_interactive_chart(df, pivots, ticker, height=600):
    fig = go.Figure()
    
    # 1. K线图
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Price'
    ))
    
    # 2. 均线
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_8'], line=dict(color='orange', width=1), name='EMA 8'))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_21'], line=dict(color='purple', width=1), name='EMA 21'))
    
    # 3. 动态绘制 ABC 结构 & 斐波那契
    # 寻找最近的一个有效 ABC: Low -> High -> Higher Low
    if len(pivots) >= 3:
        # 取最后三个点
        pC = pivots.iloc[-1]
        pB = pivots.iloc[-2]
        pA = pivots.iloc[-3]
        
        # 验证是否是上涨结构 (Low A -> High B -> Higher Low C)
        # 或者仅仅是最近的三个转折点，我们都画出来供参考
        
        # 画 ZigZag 连线
        fig.add_trace(go.Scatter(
            x=[pA['date'], pB['date'], pC['date']],
            y=[pA['price'], pB['price'], pC['price']],
            mode='lines+markers',
            name='Structure (ZigZag)',
            line=dict(color='yellow', width=2, dash='dash'),
            marker=dict(size=8)
        ))
        
        # 计算斐波那契拓展 (Extension) based on A-B leg, projected from C
        # Logic: Target = C + (B - A) * Ratio
        if pB['price'] > pA['price']: # 确保 A->B 是上涨段
            wave_height = pB['price'] - pA['price']
            
            # 定义关键点位
            levels = [0.618, 1.0, 1.618, 2.618]
            colors = ['gray', 'white', '#00FF00', 'gold']
            labels = ['0.618', '1.0 (等距)', '🎯 1.618 (止盈)', '🚀 2.618 (极值)']
            
            last_date = df.index[-1]
            start_date = pC['date']
            
            for i, fib in enumerate(levels):
                price_level = pC['price'] + wave_height * fib
                
                # 画延伸线
                fig.add_shape(type="line",
                    x0=start_date, y0=price_level, x1=last_date, y1=price_level,
                    line=dict(color=colors[i], width=1 if fib!=1.618 else 2, dash="dot" if fib!=1.618 else "solid")
                )
                # 画标签
                fig.add_annotation(
                    x=last_date, y=price_level, text=labels[i],
                    showarrow=False, xanchor="left", font=dict(color=colors[i])
                )
                
            # 止损位 (A点)
            fig.add_shape(type="line", x0=pA['date'], y0=pA['price'], x1=last_date, y1=pA['price'],
                         line=dict(color="red", width=1, dash="dot"))
            fig.add_annotation(x=pA['date'], y=pA['price'], text="⛔ STOP (A)", showarrow=False, xanchor="left", font=dict(color="red"))

    # 4. UX 交互设置 (关键修复)
    fig.update_layout(
        template="plotly_dark",
        height=height,
        title=f"{ticker} 结构分析图",
        xaxis_rangeslider_visible=True, # ✅ 开启底部拖动条
        dragmode='pan', # ✅ 默认鼠标动作为平移
        margin=dict(r=100), # 右侧留白给标签
        hovermode='x unified'
    )
    
    # 移除 rangebreaks 以保证拖动流畅性 (你可以根据喜好开启，但可能会卡顿)
    # fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])]) 
    
    return fig

# ==============================================================================
# 4. 主分析逻辑
# ==============================================================================
def analyze_data(ticker, interval, lookback_days, sensitivity):
    # 1. 下载数据
    period_map = {"1d": "2y", "1h": "1y", "15m": "60d", "5m": "30d"}
    period = period_map.get(interval, "1y")
    
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    if len(df) < 30: return None, None
    
    # 2. 计算指标
    df = calculate_indicators(df)
    
    # 3. 计算 ZigZag (传入 sensitivity)
    pivots = get_swing_pivots_simple(df['Close'], threshold_pct=sensitivity)
    
    return df, pivots

# ==============================================================================
# 5. UI 界面
# ==============================================================================
st.sidebar.header("🎛️ 控制台")

# 模式选择
mode = st.sidebar.radio("模式", ["🔍 单股精细分析", "🚀 批量扫描 (Beta)"])

if mode == "🔍 单股精细分析":
    st.title("🛡️ 狗蛋交易作战系统 (Pro UX)")
    
    # 第一行：输入与基础设置
    c1, c2 = st.columns([1, 1])
    with c1:
        ticker = st.text_input("股票代码", value="TSLA").upper()
    with c2:
        interval = st.selectbox("周期", ["1d", "1h", "15m", "5m"], index=0)
        
    # 第二行：灵敏度滑块 (这才是你想要的)
    st.write("---")
    st.markdown("### 🌊 结构设置")
    c3, c4 = st.columns([3, 1])
    with c3:
        # 这里的 key 保证了滑块拖动时会自动刷新
        sensitivity = st.slider("波段灵敏度 (ZigZag Sensitivity)", 
                                min_value=0.01, max_value=0.15, value=0.06, step=0.01,
                                help="数值越小，捕捉的波动越细微；数值越大，只看大趋势。")
    with c4:
        st.info(f"当前阈值: {sensitivity*100:.0f}%")

    # 执行分析
    if ticker:
        df, pivots = analyze_data(ticker, interval, 365, sensitivity)
        
        if df is not None:
            current_price = df['Close'].iloc[-1]
            rsi = df['RSI'].iloc[-1]
            
            # 显示核心数据
            m1, m2, m3 = st.columns(3)
            m1.metric("现价", f"${current_price:.2f}")
            m2.metric("RSI", f"{rsi:.1f}", delta="过热" if rsi>70 else "正常")
            m3.metric("ATR (波动)", f"{df['ATR'].iloc[-1]:.2f}")
            
            # 绘制可拖动图表
            fig = plot_interactive_chart(df, pivots, ticker)
            st.plotly_chart(fig, use_container_width=True)
            
            # Pivot 数据表 (可选)
            with st.expander("查看波段点位数据 (Pivots Data)"):
                st.dataframe(pivots.sort_values(by='date', ascending=False).head(10))
                
        else:
            st.error("数据加载失败，请检查代码。")

else:
    st.title("🚀 批量扫描 (保留功能)")
    st.info("批量扫描模式逻辑同上，为保证流畅性，请在单股模式调试好灵敏度后再使用。")
    # ... (批量代码可复用 analyze_data 函数)

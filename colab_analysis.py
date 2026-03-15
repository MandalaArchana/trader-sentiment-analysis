import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── STYLE ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f1117', 'axes.facecolor': '#1a1d2e',
    'axes.edgecolor': '#2d3154', 'axes.labelcolor': '#c8cce8',
    'axes.titlecolor': '#e8eaf6', 'xtick.color': '#8b90b8',
    'ytick.color': '#8b90b8', 'text.color': '#c8cce8',
    'grid.color': '#2d3154', 'grid.alpha': 0.5,
    'axes.titlesize': 13, 'axes.labelsize': 11,
    'legend.facecolor': '#1a1d2e', 'legend.edgecolor': '#2d3154',
    'legend.fontsize': 10,
})
COLORS = {
    'Extreme Fear': '#e74c3c', 'Fear': '#e67e22',
    'Neutral': '#f1c40f', 'Greed': '#2ecc71', 'Extreme Greed': '#1abc9c',
}
SENT_ORDER = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']

# ══════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════════
trader = pd.read_csv("https://drive.google.com/uc?id=1IAfLZwu6rJzyWKgBToqwSmmVYU6VbjVs")
fg     = pd.read_csv("https://drive.google.com/uc?id=1PgQC0tO8XN-wqkNyghWc_-mnrYv_nhSf")

# ══════════════════════════════════════════════════════════════════
# STEP 2 — CLEAN & PREPARE
# ══════════════════════════════════════════════════════════════════

# Parse trader date from 'Timestamp IST' (format: 02-12-2024 22:50)
trader['date'] = pd.to_datetime(trader['Timestamp IST'], dayfirst=True, errors='coerce').dt.normalize()

# Parse fear/greed date
fg['date'] = pd.to_datetime(fg['date'])
fg = fg.rename(columns={'classification': 'sentiment'})

# Derived columns
trader['net_pnl'] = trader['Closed PnL'] - trader['Fee']
trader['is_win']  = trader['Closed PnL'] > 0
trader['Side_clean'] = trader['Side'].str.upper().str.strip()

# Merge on date
df = trader.merge(fg[['date', 'sentiment', 'value']], on='date', how='left')
df = df.dropna(subset=['sentiment'])
df['sentiment'] = pd.Categorical(df['sentiment'], categories=SENT_ORDER, ordered=True)

print("=== DATA LOADED ===")
print(f"Total trades after merge : {len(df):,}")
print(f"Date range               : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"Unique accounts          : {df['Account'].nunique():,}")
print(f"Unique coins             : {df['Coin'].nunique()}")
print(f"\nSentiment distribution:\n{df['sentiment'].value_counts().to_string()}")
print(f"\nClosed PnL stats:\n{df['Closed PnL'].describe().round(2).to_string()}")
print(f"\nMissing values:\n{df[['Closed PnL','Fee','Size USD','sentiment']].isnull().sum().to_string()}")

# ══════════════════════════════════════════════════════════════════
# STEP 3 — KEY STATISTICS
# ══════════════════════════════════════════════════════════════════
print("\n=== KEY STATISTICS BY SENTIMENT ===")

print("\n[1] Trade Count:")
print(df.groupby('sentiment', observed=True).size().to_string())

print("\n[2] Avg Net PnL per Trade (USD):")
print(df.groupby('sentiment', observed=True)['net_pnl'].mean().round(4).to_string())

print("\n[3] Median Net PnL per Trade (USD):")
print(df.groupby('sentiment', observed=True)['net_pnl'].median().round(4).to_string())

print("\n[4] Win Rate (%):")
print((df.groupby('sentiment', observed=True)['is_win'].mean()*100).round(2).to_string())

print("\n[5] Avg Trade Size (USD):")
print(df.groupby('sentiment', observed=True)['Size USD'].mean().round(2).to_string())

print("\n[6] Total Volume (USD):")
print(df.groupby('sentiment', observed=True)['Size USD'].sum().round(0).to_string())

print("\n[7] Avg Fee per Trade (USD):")
print(df.groupby('sentiment', observed=True)['Fee'].mean().round(4).to_string())

print("\n[8] Buy/Sell Ratio:")
ratio = df.groupby('sentiment', observed=True).apply(
    lambda x: round((x['Side_clean']=='BUY').sum() / max((x['Side_clean']=='SELL').sum(),1), 3)
)
print(ratio.to_string())

print("\n[9] Top 10 Most Traded Coins:")
print(df['Coin'].value_counts().head(10).to_string())

print("\n[10] Avg PnL by Coin (top 10 by volume):")
top_coins = df['Coin'].value_counts().head(10).index
print(df[df['Coin'].isin(top_coins)].groupby('Coin')['net_pnl'].mean().round(4).sort_values().to_string())

# Statistical test
print("\n[11] Kruskal-Wallis Test — PnL across sentiment groups:")
groups = [df[df['sentiment']==s]['net_pnl'].dropna().values for s in SENT_ORDER]
groups = [g for g in groups if len(g) > 10]
stat, pval = stats.kruskal(*groups)
print(f"    H = {stat:.4f},  p = {pval:.6f}")
print(f"    → {'SIGNIFICANT difference across sentiment groups' if pval < 0.05 else 'No significant difference'} (α=0.05)")

# ══════════════════════════════════════════════════════════════════
# FIGURE 1 — OVERVIEW DASHBOARD
# ══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor('#0f1117')
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# 1a — Sentiment pie
ax1 = fig.add_subplot(gs[0, 0])
sc = df['sentiment'].value_counts().reindex(SENT_ORDER).dropna()
wedges, texts, autotexts = ax1.pie(
    sc.values, labels=sc.index, colors=[COLORS[s] for s in sc.index],
    autopct='%1.1f%%', startangle=90, pctdistance=0.8,
    wedgeprops={'linewidth':1.5,'edgecolor':'#0f1117'})
for t in texts:   t.set_fontsize(9);  t.set_color('#c8cce8')
for at in autotexts: at.set_fontsize(8); at.set_color('white'); at.set_fontweight('bold')
ax1.set_title('Market Sentiment\nDistribution of Trades', fontweight='bold')

# 1b — Volume by sentiment
ax2 = fig.add_subplot(gs[0, 1])
vol = df.groupby('sentiment', observed=True)['Size USD'].sum() / 1e6
bars = ax2.bar(vol.index, vol.values,
               color=[COLORS[s] for s in vol.index], edgecolor='#0f1117')
ax2.set_title('Total Trade Volume\nby Sentiment (USD M)', fontweight='bold')
ax2.set_ylabel('Volume (USD Millions)'); ax2.tick_params(axis='x', rotation=30)
for bar, val in zip(bars, vol.values):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
             f'${val:.0f}M', ha='center', va='bottom', fontsize=8, color='white')

# 1c — Avg Net PnL by sentiment
ax3 = fig.add_subplot(gs[0, 2])
avg_pnl = df.groupby('sentiment', observed=True)['net_pnl'].mean()
ax3.bar(avg_pnl.index, avg_pnl.values,
        color=['#e74c3c' if v < 0 else '#2ecc71' for v in avg_pnl.values],
        edgecolor='#0f1117')
ax3.axhline(0, color='#8b90b8', linewidth=1, linestyle='--', alpha=0.6)
ax3.set_title('Avg Net PnL per Trade\nby Sentiment (USD)', fontweight='bold')
ax3.set_ylabel('Avg Net PnL (USD)'); ax3.tick_params(axis='x', rotation=30)

# 1d — Win rate
ax4 = fig.add_subplot(gs[1, 0])
wr = df.groupby('sentiment', observed=True)['is_win'].mean() * 100
ax4.bar(wr.index, wr.values,
        color=[COLORS[s] for s in wr.index], edgecolor='#0f1117')
ax4.axhline(50, color='white', linewidth=1, linestyle='--', alpha=0.6, label='50% baseline')
ax4.set_title('Win Rate by Sentiment (%)', fontweight='bold')
ax4.set_ylabel('Win Rate (%)'); ax4.tick_params(axis='x', rotation=30); ax4.legend()

# 1e — BUY vs SELL
ax5 = fig.add_subplot(gs[1, 1])
ss = df.groupby(['sentiment', 'Side_clean'], observed=True).size().unstack(fill_value=0)
x = np.arange(len(SENT_ORDER)); w = 0.35
for col, color, offset in [('BUY','#2ecc71',-w/2),('SELL','#e74c3c',w/2)]:
    if col in ss.columns:
        ax5.bar(x+offset, ss.reindex(SENT_ORDER)[col].fillna(0), w,
                label=col, color=color, edgecolor='#0f1117')
ax5.set_xticks(x); ax5.set_xticklabels(SENT_ORDER, rotation=30, fontsize=8)
ax5.set_title('BUY vs SELL Count\nby Sentiment', fontweight='bold')
ax5.set_ylabel('Number of Trades'); ax5.legend()

# 1f — PnL boxplot
ax6 = fig.add_subplot(gs[1, 2])
groups_plot = [df[df['sentiment']==s]['net_pnl'].clip(-500,500).dropna().values for s in SENT_ORDER]
bp = ax6.boxplot(groups_plot, labels=SENT_ORDER, patch_artist=True,
                 medianprops={'color':'white','linewidth':2},
                 whiskerprops={'color':'#8b90b8'}, capprops={'color':'#8b90b8'},
                 flierprops={'marker':'o','markersize':2,'alpha':0.2,'markerfacecolor':'#8b90b8'})
for patch, sent in zip(bp['boxes'], SENT_ORDER):
    patch.set_facecolor(COLORS[sent]); patch.set_alpha(0.7)
ax6.axhline(0, color='white', linewidth=0.8, linestyle='--', alpha=0.5)
ax6.set_title('PnL Distribution by Sentiment\n(clipped ±$500)', fontweight='bold')
ax6.set_ylabel('Net PnL (USD)'); ax6.tick_params(axis='x', rotation=30)

# 1g — Fear/Greed timeline
ax7 = fig.add_subplot(gs[2, :])
fg_s = fg.sort_values('date')
ax7.fill_between(fg_s['date'], fg_s['value'], alpha=0.25, color='#7c83e0')
ax7.plot(fg_s['date'], fg_s['value'], color='#9ea8f5', linewidth=0.8)
ax7.axhline(25, color='#e74c3c', linewidth=0.8, linestyle='--', alpha=0.6, label='Extreme Fear (<25)')
ax7.axhline(75, color='#2ecc71', linewidth=0.8, linestyle='--', alpha=0.6, label='Extreme Greed (>75)')
ax7.set_title('Bitcoin Fear & Greed Index — 2018 to 2025', fontweight='bold')
ax7.set_ylabel('Index Value (0–100)'); ax7.set_ylim(0,100); ax7.legend()

fig.suptitle('Trader Behavior vs Market Sentiment — Overview Dashboard',
             fontsize=16, fontweight='bold', color='white', y=1.01)
plt.savefig('fig1_overview.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.show()
print("Fig1 saved")

# ══════════════════════════════════════════════════════════════════
# FIGURE 2 — PERFORMANCE DEEP DIVE
# ══════════════════════════════════════════════════════════════════
fig2, axes = plt.subplots(2, 3, figsize=(20, 12))
fig2.patch.set_facecolor('#0f1117')

# 2a — Avg trade size
ax = axes[0,0]
avg_size = df.groupby('sentiment', observed=True)['Size USD'].mean()
ax.bar(avg_size.index, avg_size.values,
       color=[COLORS[s] for s in avg_size.index], edgecolor='#0f1117')
ax.set_title('Avg Trade Size (USD)\nby Sentiment', fontweight='bold')
ax.set_ylabel('Avg Size (USD)'); ax.tick_params(axis='x', rotation=30)

# 2b — Trade count per unique account by sentiment
ax = axes[0,1]
trades_per_acc = df.groupby(['sentiment','Account'], observed=True).size().reset_index(name='count')
avg_trades = trades_per_acc.groupby('sentiment', observed=True)['count'].mean()
ax.bar(avg_trades.index, avg_trades.values,
       color=[COLORS[s] for s in avg_trades.index], edgecolor='#0f1117')
ax.set_title('Avg Trades per Account\nby Sentiment', fontweight='bold')
ax.set_ylabel('Avg Trades per Account'); ax.tick_params(axis='x', rotation=30)

# 2c — Top 10 coins
ax = axes[0,2]
top10 = df['Coin'].value_counts().head(10)
ax.barh(top10.index[::-1], top10.values[::-1], color='#7c83e0', edgecolor='#0f1117')
ax.set_title('Top 10 Most Traded Coins', fontweight='bold')
ax.set_xlabel('Number of Trades')

# 2d — Avg PnL by coin
ax = axes[1,0]
coin_pnl = df[df['Coin'].isin(df['Coin'].value_counts().head(10).index)] \
             .groupby('Coin')['net_pnl'].mean().sort_values()
ax.barh(coin_pnl.index, coin_pnl.values,
        color=['#e74c3c' if v < 0 else '#2ecc71' for v in coin_pnl.values],
        edgecolor='#0f1117')
ax.axvline(0, color='white', linewidth=0.8, linestyle='--', alpha=0.5)
ax.set_title('Avg Net PnL by Coin\n(Top 10 by volume)', fontweight='bold')
ax.set_xlabel('Avg Net PnL (USD)')

# 2e — Fee burden by sentiment
ax = axes[1,1]
fee_s = df.groupby('sentiment', observed=True)['Fee'].mean()
ax.bar(fee_s.index, fee_s.values,
       color=[COLORS[s] for s in fee_s.index], edgecolor='#0f1117')
ax.set_title('Avg Fee per Trade\nby Sentiment (USD)', fontweight='bold')
ax.set_ylabel('Avg Fee (USD)'); ax.tick_params(axis='x', rotation=30)

# 2f — PnL heatmap (FG value bucket × sentiment)
ax = axes[1,2]
df['fg_bucket'] = pd.cut(df['value'], bins=[0,20,40,60,80,100],
                          labels=['0-20','21-40','41-60','61-80','81-100'])
heat = df.groupby(['fg_bucket','sentiment'], observed=True)['net_pnl'].mean().unstack()
heat = heat.reindex(columns=SENT_ORDER)
sns.heatmap(heat, ax=ax, cmap='RdYlGn', center=0, linewidths=0.5,
            linecolor='#0f1117', annot=True, fmt='.2f',
            cbar_kws={'label':'Avg Net PnL (USD)'})
ax.set_title('Avg PnL Heatmap\n(FG Value Bucket × Sentiment)', fontweight='bold')
ax.set_xlabel('Sentiment'); ax.set_ylabel('FG Index Value Bucket')

fig2.suptitle('Trader Performance Deep Dive', fontsize=16, fontweight='bold', color='white')
plt.tight_layout()
plt.savefig('fig2_performance.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.show()
print("Fig2 saved")

# ══════════════════════════════════════════════════════════════════
# FIGURE 3 — BEHAVIORAL PATTERNS & STRATEGY INSIGHTS
# ══════════════════════════════════════════════════════════════════
fig3, axes = plt.subplots(2, 3, figsize=(20, 12))
fig3.patch.set_facecolor('#0f1117')

# 3a — Buy/Sell ratio
ax = axes[0,0]
ratio = df.groupby('sentiment', observed=True).apply(
    lambda x: (x['Side_clean']=='BUY').sum() / max((x['Side_clean']=='SELL').sum(),1)
)
ax.bar(ratio.index, ratio.values,
       color=['#2ecc71' if v>=1 else '#e74c3c' for v in ratio.values],
       edgecolor='#0f1117')
ax.axhline(1, color='white', linewidth=1, linestyle='--', alpha=0.6, label='Equal ratio')
ax.set_title('Buy/Sell Ratio\nby Sentiment', fontweight='bold')
ax.set_ylabel('BUY / SELL Ratio'); ax.tick_params(axis='x', rotation=30); ax.legend()

# 3b — FG index rolling MA
ax = axes[0,1]
fg_s = fg.sort_values('date').set_index('date')
ax.plot(fg_s.index, fg_s['value'], color='#8b90b8', linewidth=0.6, alpha=0.4, label='Daily')
ax.plot(fg_s.index, fg_s['value'].rolling(30).mean(),
        color='#f1c40f', linewidth=2, label='30-day MA')
ax.fill_between(fg_s.index, 0, 25, alpha=0.07, color='#e74c3c', label='Extreme Fear zone')
ax.fill_between(fg_s.index, 75, 100, alpha=0.07, color='#2ecc71', label='Extreme Greed zone')
ax.set_title('Fear/Greed Index\n30-day Moving Average', fontweight='bold')
ax.set_ylabel('Index Value'); ax.set_ylim(0,100); ax.legend(fontsize=8)

# 3c — Cumulative PnL per sentiment (real data, first 1000 trades each)
ax = axes[0,2]
for sent in SENT_ORDER:
    sub = df[df['sentiment']==sent].sort_values('date').head(1000)
    if len(sub) > 10:
        ax.plot(sub['net_pnl'].cumsum().values, label=sent,
                color=COLORS[sent], linewidth=1.5, alpha=0.85)
ax.axhline(0, color='white', linewidth=0.8, linestyle='--', alpha=0.4)
ax.set_title('Cumulative PnL Trajectory\n(First 1000 trades per sentiment)', fontweight='bold')
ax.set_xlabel('Trade Number'); ax.set_ylabel('Cumulative Net PnL (USD)'); ax.legend(fontsize=8)

# 3d — PnL distribution violin plot
ax = axes[1,0]
violin_data = [df[df['sentiment']==s]['net_pnl'].clip(-300,300).dropna().values for s in SENT_ORDER]
parts = ax.violinplot(violin_data, positions=range(len(SENT_ORDER)),
                      showmedians=True, showextrema=True)
for i, (pc, sent) in enumerate(zip(parts['bodies'], SENT_ORDER)):
    pc.set_facecolor(COLORS[sent]); pc.set_alpha(0.6)
parts['cmedians'].set_color('white'); parts['cmedians'].set_linewidth(2)
ax.set_xticks(range(len(SENT_ORDER)))
ax.set_xticklabels(SENT_ORDER, rotation=30, fontsize=8)
ax.axhline(0, color='white', linewidth=0.8, linestyle='--', alpha=0.4)
ax.set_title('PnL Violin Plot by Sentiment\n(clipped ±$300)', fontweight='bold')
ax.set_ylabel('Net PnL (USD)')

# 3e — Monthly trade frequency stacked bar
ax = axes[1,1]
df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
monthly = df.groupby(['month','sentiment'], observed=True).size().unstack(fill_value=0)
monthly = monthly.reindex(columns=SENT_ORDER, fill_value=0)
bottom = np.zeros(len(monthly))
for sent in SENT_ORDER:
    if sent in monthly.columns:
        ax.bar(monthly.index, monthly[sent], bottom=bottom,
               color=COLORS[sent], label=sent, width=20, edgecolor='none')
        bottom += monthly[sent].values
ax.set_title('Monthly Trade Frequency\nby Sentiment (Stacked)', fontweight='bold')
ax.set_xlabel('Month'); ax.set_ylabel('Number of Trades'); ax.legend(fontsize=8)

# 3f — Top 10 traders by total PnL
ax = axes[1,2]
top_traders = df.groupby('Account')['net_pnl'].sum().sort_values(ascending=False).head(10)
short_labels = [f"{a[:6]}...{a[-4:]}" for a in top_traders.index]
ax.barh(short_labels[::-1], top_traders.values[::-1],
        color=['#2ecc71' if v>=0 else '#e74c3c' for v in top_traders.values[::-1]],
        edgecolor='#0f1117')
ax.axvline(0, color='white', linewidth=0.8, linestyle='--', alpha=0.5)
ax.set_title('Top 10 Traders\nby Total Net PnL', fontweight='bold')
ax.set_xlabel('Total Net PnL (USD)')

fig3.suptitle('Behavioral Patterns & Strategic Insights',
              fontsize=16, fontweight='bold', color='white')
plt.tight_layout()
plt.savefig('fig3_insights.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.show()
print("Fig3 saved")

print("\n✓ All 3 figures saved. Download them from the Colab file panel (folder icon on left).")

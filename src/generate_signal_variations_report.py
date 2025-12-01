#!/usr/bin/env python3
"""
Signal Variations Analysis Report Generator

This script generates a comprehensive report analyzing the results from test_signal_variations.py
It compares walk-forward analysis with traditional strategies across all tickers.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
import os

# Import configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_config import TICKERS, RESULTS_VARIATIONS_DIR

class SignalVariationsReportGenerator:
    def __init__(self):
        self.results_dir = Path(RESULTS_VARIATIONS_DIR)
        self.report_dir = self.results_dir / "reports"
        self.report_dir.mkdir(exist_ok=True)
        
        self.all_results = {}
        self.summary_stats = {}
        
    def load_all_results(self):
        """Load all signal variation results for analysis."""
        print("Loading signal variation results...")
        
        for ticker in TICKERS:
            comparison_file = self.results_dir / f"{ticker}_signal_variations_comparison.csv"
            detailed_file = self.results_dir / f"{ticker}_walk_forward_detailed.csv"
            selections_file = self.results_dir / f"{ticker}_strategy_selections.csv"
            
            if comparison_file.exists():
                self.all_results[ticker] = {
                    'comparison': pd.read_csv(comparison_file),
                    'detailed': pd.read_csv(detailed_file) if detailed_file.exists() else None,
                    'selections': pd.read_csv(selections_file) if selections_file.exists() else None
                }
                print(f"✅ Loaded {ticker} results")
            else:
                print(f"❌ Missing {ticker} results")
        
        print(f"Total loaded: {len(self.all_results)} tickers")
        
    def analyze_walk_forward_performance(self):
        """Analyze walk-forward performance across all tickers."""
        wf_performance = {}
        
        for ticker, data in self.all_results.items():
            df = data['comparison']
            
            # Get walk-forward results
            wf_row = df[df['Strategy'].str.contains('Walk-Forward', na=False)]
            bh_row = df[df['Strategy'] == 'Buy & Hold']
            
            if not wf_row.empty and not bh_row.empty:
                wf = wf_row.iloc[0]
                bh = bh_row.iloc[0]
                
                wf_performance[ticker] = {
                    'WF_CAGR': wf['CAGR'] * 100,
                    'WF_Sharpe': wf['Sharpe'],
                    'WF_MaxDD': wf['MaxDD'] * 100,
                    'WF_Volatility': wf['Volatility'] * 100,
                    'BH_CAGR': bh['CAGR'] * 100,
                    'BH_Sharpe': bh['Sharpe'],
                    'BH_MaxDD': bh['MaxDD'] * 100,
                    'CAGR_Ratio': wf['CAGR'] / bh['CAGR'] if bh['CAGR'] != 0 else 0,
                    'Sharpe_Diff': wf['Sharpe'] - bh['Sharpe'],
                    'Risk_Reduction': (abs(bh['MaxDD']) - abs(wf['MaxDD'])) / abs(bh['MaxDD']) * 100,
                    'Final_Equity_WF': wf['Final_Equity'],
                    'Final_Equity_BH': bh['Final_Equity']
                }
        
        return pd.DataFrame(wf_performance).T
    
    def analyze_strategy_selection_patterns(self):
        """Analyze which strategies are selected most often."""
        strategy_counts = {}
        
        for ticker, data in self.all_results.items():
            if data['selections'] is not None:
                selections = data['selections']
                strategy_freq = selections['Strategy'].value_counts()
                strategy_counts[ticker] = strategy_freq
        
        return strategy_counts
    
    def generate_performance_charts(self):
        """Generate performance comparison charts."""
        wf_perf = self.analyze_walk_forward_performance()
        
        if wf_perf.empty:
            print("No data available for charts")
            return
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Walk-Forward vs Buy & Hold Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. CAGR Comparison
        ax1 = axes[0, 0]
        x_pos = np.arange(len(wf_perf))
        width = 0.35
        
        ax1.bar(x_pos - width/2, wf_perf['WF_CAGR'], width, label='Walk-Forward', color='crimson', alpha=0.7)
        ax1.bar(x_pos + width/2, wf_perf['BH_CAGR'], width, label='Buy & Hold', color='navy', alpha=0.7)
        ax1.set_xlabel('Tickers')
        ax1.set_ylabel('CAGR (%)')
        ax1.set_title('CAGR Comparison')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(wf_perf.index, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Sharpe Ratio Comparison
        ax2 = axes[0, 1]
        ax2.bar(x_pos - width/2, wf_perf['WF_Sharpe'], width, label='Walk-Forward', color='crimson', alpha=0.7)
        ax2.bar(x_pos + width/2, wf_perf['BH_Sharpe'], width, label='Buy & Hold', color='navy', alpha=0.7)
        ax2.set_xlabel('Tickers')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.set_title('Sharpe Ratio Comparison')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(wf_perf.index, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Risk vs Return Scatter
        ax3 = axes[1, 0]
        ax3.scatter(wf_perf['WF_Volatility'], wf_perf['WF_CAGR'], 
                   color='crimson', s=100, alpha=0.7, label='Walk-Forward')
        ax3.scatter(wf_perf['WF_Volatility'], wf_perf['BH_CAGR'], 
                   color='navy', s=100, alpha=0.7, label='Buy & Hold')
        
        for i, ticker in enumerate(wf_perf.index):
            ax3.annotate(ticker, (wf_perf['WF_Volatility'].iloc[i], wf_perf['WF_CAGR'].iloc[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax3.set_xlabel('Volatility (%)')
        ax3.set_ylabel('CAGR (%)')
        ax3.set_title('Risk vs Return Profile')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance Ratios
        ax4 = axes[1, 1]
        ratios_data = wf_perf[['CAGR_Ratio', 'Sharpe_Diff']].copy()
        ratios_data['CAGR_Ratio'] = (ratios_data['CAGR_Ratio'] - 1) * 100  # Convert to percentage difference
        
        x_pos = np.arange(len(ratios_data))
        ax4.bar(x_pos, ratios_data['CAGR_Ratio'], color='green', alpha=0.7, label='CAGR Diff (%)')
        ax4_twin = ax4.twinx()
        ax4_twin.plot(x_pos, ratios_data['Sharpe_Diff'], color='orange', marker='o', linewidth=2, label='Sharpe Diff')
        
        ax4.set_xlabel('Tickers')
        ax4.set_ylabel('CAGR Difference (%)', color='green')
        ax4_twin.set_ylabel('Sharpe Difference', color='orange')
        ax4.set_title('Walk-Forward vs Buy & Hold Differences')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(ratios_data.index, rotation=45)
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save chart
        chart_path = self.report_dir / "performance_analysis_charts.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
    
    def generate_markdown_report(self):
        """Generate a comprehensive markdown report."""
        wf_perf = self.analyze_walk_forward_performance()
        strategy_patterns = self.analyze_strategy_selection_patterns()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# Signal Variations Analysis Report

**Generated:** {timestamp}  
**Analysis Period:** 2000-2025  
**Methodology:** Walk-Forward Analysis (2-year training, 6-month testing windows)

---

## Executive Summary

This report analyzes the performance of walk-forward analysis compared to buy & hold strategy across {len(wf_perf)} major stocks. The walk-forward approach uses a rigorous methodology that avoids look-ahead bias by training on historical data and testing on future periods.

### Key Findings

"""
        
        if not wf_perf.empty:
            # Calculate summary statistics
            avg_wf_cagr = wf_perf['WF_CAGR'].mean()
            avg_bh_cagr = wf_perf['BH_CAGR'].mean()
            avg_sharpe_improvement = wf_perf['Sharpe_Diff'].mean()
            avg_risk_reduction = wf_perf['Risk_Reduction'].mean()
            
            winners = (wf_perf['CAGR_Ratio'] > 1).sum()
            total = len(wf_perf)
            
            report += f"""
- **Average Walk-Forward CAGR:** {avg_wf_cagr:.1f}%
- **Average Buy & Hold CAGR:** {avg_bh_cagr:.1f}%
- **Walk-Forward Outperforms:** {winners}/{total} stocks ({winners/total*100:.0f}%)
- **Average Sharpe Improvement:** {avg_sharpe_improvement:+.2f}
- **Average Risk Reduction:** {avg_risk_reduction:+.1f}%

---

## Detailed Performance Analysis

### Performance by Stock

| Ticker | WF CAGR | BH CAGR | WF Sharpe | BH Sharpe | Risk Reduction | Performance |
|--------|---------|---------|-----------|-----------|----------------|-------------|
"""
            
            for ticker in wf_perf.index:
                row = wf_perf.loc[ticker]
                performance = "🟢 WIN" if row['CAGR_Ratio'] > 1 else "🔴 LOSS"
                report += f"| {ticker} | {row['WF_CAGR']:.1f}% | {row['BH_CAGR']:.1f}% | {row['WF_Sharpe']:.2f} | {row['BH_Sharpe']:.2f} | {row['Risk_Reduction']:+.1f}% | {performance} |\n"
        
        # Strategy Selection Analysis
        report += f"""
---

## Strategy Selection Patterns

The walk-forward analysis dynamically selects the best-performing strategy for each period. Here's how often each strategy was chosen:

"""
        
        # Aggregate strategy selections across all tickers
        all_strategies = {}
        total_periods = 0
        
        for ticker, strategies in strategy_patterns.items():
            total_periods += strategies.sum()
            for strategy, count in strategies.items():
                if strategy in all_strategies:
                    all_strategies[strategy] += count
                else:
                    all_strategies[strategy] = count
        
        # Sort strategies by frequency
        sorted_strategies = sorted(all_strategies.items(), key=lambda x: x[1], reverse=True)
        
        for strategy, count in sorted_strategies:
            percentage = (count / total_periods) * 100
            report += f"- **{strategy}:** {count} periods ({percentage:.1f}%)\n"
        
        # Individual ticker analysis
        report += f"""
---

## Individual Ticker Analysis

### Best Performers (Walk-Forward)
"""
        
        if not wf_perf.empty:
            # Sort by CAGR ratio (WF vs BH)
            best_performers = wf_perf.nlargest(3, 'CAGR_Ratio')
            
            for ticker in best_performers.index:
                row = best_performers.loc[ticker]
                report += f"""
#### {ticker}
- **Walk-Forward CAGR:** {row['WF_CAGR']:.1f}% vs **Buy & Hold:** {row['BH_CAGR']:.1f}%
- **Sharpe Improvement:** {row['Sharpe_Diff']:+.2f} ({row['WF_Sharpe']:.2f} vs {row['BH_Sharpe']:.2f})
- **Risk Reduction:** {row['Risk_Reduction']:+.1f}% (MaxDD: {row['WF_MaxDD']:.1f}% vs {row['BH_MaxDD']:.1f}%)
- **Final Equity:** ${row['Final_Equity_WF']:,.0f} vs ${row['Final_Equity_BH']:,.0f}
"""
        
        # Methodology explanation
        report += f"""
---

## Methodology

### Walk-Forward Analysis Process

1. **Training Phase:** Use 2 years of historical data to test 9 different signal strategies
2. **Selection:** Choose the strategy with the highest Sharpe ratio during training
3. **Testing:** Apply the selected strategy to the next 6 months of data
4. **Rolling:** Move forward 6 months and repeat the process
5. **Aggregation:** Combine all testing periods to calculate overall performance

### Strategy Options Tested

- **Original (≥2 signals):** Buy when at least 2 of 4 MA signals are active
- **Short-term only (5,20):** Use only fast MA crossover
- **Medium-term only (10,50):** Use only medium MA crossover  
- **Long-term only (50,200):** Use only slow MA crossover
- **Short OR Long:** Buy when either short OR long signal is active
- **Short AND Medium:** Buy when both short AND medium signals are active
- **Long AND VeryLong:** Buy when both long signals are active
- **≥3 of 4 signals:** Buy when at least 3 of 4 signals are active
- **All 4 signals:** Buy only when all 4 signals are active

### Key Advantages

- **No Look-Ahead Bias:** Uses only historical data available at each point in time
- **Adaptive:** Changes strategy based on market conditions
- **Realistic:** Simulates actual trading decisions
- **Robust:** Tested across multiple market cycles and conditions

---

## Conclusions

"""
        
        if not wf_perf.empty:
            best_ticker = wf_perf.loc[wf_perf['CAGR_Ratio'].idxmax()]
            worst_ticker = wf_perf.loc[wf_perf['CAGR_Ratio'].idxmin()]
            
            report += f"""
### Overall Assessment

The walk-forward analysis demonstrates varying effectiveness across different stocks:

**Strengths:**
- Provides realistic, implementable trading results
- Often improves risk-adjusted returns (Sharpe ratio)
- Reduces maximum drawdown in most cases
- Adapts to changing market conditions

**Best Case:** {best_ticker.name} achieved {best_ticker['WF_CAGR']:.1f}% CAGR vs {best_ticker['BH_CAGR']:.1f}% buy & hold

**Challenging Case:** {worst_ticker.name} achieved {worst_ticker['WF_CAGR']:.1f}% CAGR vs {worst_ticker['BH_CAGR']:.1f}% buy & hold

### Recommendations

1. **Use for mature, cyclical stocks** where technical analysis is more effective
2. **Focus on risk-adjusted returns** rather than pure CAGR
3. **Consider combining** with fundamental analysis for better results
4. **Avoid for highly volatile growth stocks** where momentum strategies might work better

---

*This report was generated automatically from walk-forward analysis results.*
"""
        
        return report
    
    def save_report(self, report_content):
        """Save the report to a markdown file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.report_dir / f"Signal_Variations_Report_{timestamp}.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return report_path
    
    def generate_full_report(self):
        """Generate complete analysis report."""
        print("🔍 Generating Signal Variations Analysis Report...")
        print("="*60)
        
        # Load data
        self.load_all_results()
        
        if not self.all_results:
            print("❌ No results found. Please run test_signal_variations.py first.")
            return None
        
        # Generate visualizations
        print("📊 Creating performance charts...")
        chart_path = self.generate_performance_charts()
        print(f"✅ Charts saved to: {chart_path}")
        
        # Generate report
        print("📄 Generating markdown report...")
        report_content = self.generate_markdown_report()
        report_path = self.save_report(report_content)
        print(f"✅ Report saved to: {report_path}")
        
        print("\n🎉 Report generation complete!")
        print(f"📁 Report directory: {self.report_dir}")
        
        return {
            'report_path': report_path,
            'chart_path': chart_path,
            'report_dir': self.report_dir
        }

def main():
    """Main execution function."""
    generator = SignalVariationsReportGenerator()
    results = generator.generate_full_report()
    
    if results:
        print(f"\n📖 To view the report, open: {results['report_path']}")
        print(f"📊 To view charts, open: {results['chart_path']}")

if __name__ == "__main__":
    main()
"""
Visualization and Dashboard Module

Provides comprehensive business intelligence dashboards and visualizations
for vendor analytics, risk assessment, and lending portfolio analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')


class DashboardGenerator:
    """Generates comprehensive business intelligence dashboards."""
    
    def __init__(self, figsize: Tuple[int, int] = (20, 16)):
        """Initialize dashboard generator with default styling."""
        plt.style.use('default')
        sns.set_palette("husl")
        self.figsize = figsize
        self.setup_styling()
    
    def setup_styling(self):
        """Set up consistent styling for all plots."""
        plt.rcParams['figure.figsize'] = self.figsize
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 9
    
    def create_score_distribution_plot(self, ax, metrics_df: pd.DataFrame):
        """Create lending score distribution histogram."""
        scores = metrics_df['lending_score']
        n, bins, patches = ax.hist(scores, bins=8, color='skyblue', alpha=0.7, edgecolor='black')
        ax.axvline(scores.mean(), color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {scores.mean():.1f}')
        ax.axvline(50, color='green', linestyle='-', linewidth=3, label='Loan Threshold: 50')
        ax.set_title('Lending Score Distribution', fontweight='bold')
        ax.set_xlabel('Lending Score')
        ax.set_ylabel('Number of Vendors')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def create_risk_distribution_pie(self, ax, metrics_df: pd.DataFrame):
        """Create risk category distribution pie chart."""
        risk_counts = metrics_df['risk_category'].value_counts()
        colors = ['#2E8B57', '#FFD700', '#FF6347', '#8B0000']  # Green, Yellow, Red, Dark Red
        wedges, texts, autotexts = ax.pie(
            risk_counts.values, 
            labels=[cat.replace('_', ' ') for cat in risk_counts.index], 
            autopct='%1.1f%%', 
            colors=colors[:len(risk_counts)], 
            startangle=90
        )
        ax.set_title('Risk Category Distribution', fontweight='bold')
    
    def create_engagement_activity_scatter(self, ax, metrics_df: pd.DataFrame):
        """Create engagement vs activity scatter plot."""
        scatter = ax.scatter(
            metrics_df['posts_per_week'], 
            metrics_df['avg_views_per_post'], 
            c=metrics_df['lending_score'], 
            s=metrics_df['recommended_loan_etb']/500, 
            alpha=0.7, 
            cmap='RdYlGn', 
            edgecolors='black'
        )
        ax.set_xlabel('Posts per Week')
        ax.set_ylabel('Avg Views per Post')
        ax.set_title('Activity vs Engagement\n(Size = Loan Amount)', fontweight='bold')
        plt.colorbar(scatter, ax=ax, label='Lending Score')
        ax.grid(True, alpha=0.3)
    
    def create_price_analysis_plot(self, ax, metrics_df: pd.DataFrame):
        """Create price analysis by vendor."""
        vendors_with_prices = metrics_df[metrics_df['avg_price_etb'] > 0]
        if not vendors_with_prices.empty:
            colors = ['green' if score >= 50 else 'red' 
                     for score in vendors_with_prices['lending_score']]
            bars = ax.bar(range(len(vendors_with_prices)), vendors_with_prices['avg_price_etb'], 
                         color=colors, alpha=0.7, edgecolor='black')
            ax.set_xticks(range(len(vendors_with_prices)))
            ax.set_xticklabels([title[:12] + '...' if len(title) > 12 else title 
                               for title in vendors_with_prices['channel_title']], rotation=45)
            ax.set_title('Average Price Points by Vendor', fontweight='bold')
            ax.set_ylabel('Average Price (ETB)')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No Price Data Available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title('Price Analysis - No Data', fontweight='bold')
    
    def create_diversity_impact_scatter(self, ax, metrics_df: pd.DataFrame):
        """Create product diversity vs lending score scatter."""
        scatter = ax.scatter(
            metrics_df['product_diversity'], 
            metrics_df['lending_score'], 
            s=metrics_df['total_views']/100, 
            alpha=0.6, 
            c='purple', 
            edgecolors='black'
        )
        ax.set_xlabel('Product Diversity (# of product types)')
        ax.set_ylabel('Lending Score')
        ax.set_title('Business Diversification Impact\n(Size = Total Views)', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def create_loan_amounts_bar(self, ax, metrics_df: pd.DataFrame):
        """Create recommended loan amounts horizontal bar chart."""
        loan_data = metrics_df.sort_values('recommended_loan_etb', ascending=True)
        colors = [
            loan_data.loc[idx, 'risk_color'].replace('🟢', 'green').replace('🟡', 'yellow')
            .replace('🔴', 'red').replace('⚫', 'black') for idx in loan_data.index
        ]
        bars = ax.barh(range(len(loan_data)), loan_data['recommended_loan_etb']/1000,
                      color=colors, alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(loan_data)))
        ax.set_yticklabels([title[:12] + '...' if len(title) > 12 else title 
                           for title in loan_data['channel_title']])
        ax.set_xlabel('Recommended Loan (Thousands ETB)')
        ax.set_title('Loan Amount Recommendations', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def create_score_components_bar(self, ax, metrics_df: pd.DataFrame):
        """Create score components breakdown bar chart."""
        score_components = ['score_market_reach', 'score_activity_level', 'score_diversification', 
                           'score_viability', 'score_consistency']
        component_names = ['Market Reach\n(30%)', 'Activity\n(25%)', 'Diversification\n(20%)', 
                          'Viability\n(15%)', 'Consistency\n(10%)']
        avg_scores = [metrics_df[comp].mean() for comp in score_components]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
        bars = ax.bar(component_names, avg_scores, color=colors, alpha=0.8)
        ax.set_title('Average Score Components', fontweight='bold')
        ax.set_ylabel('Average Score')
        ax.grid(True, alpha=0.3)
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{height:.1f}',
                   ha='center', va='bottom', fontweight='bold')
    
    def create_category_performance_bar(self, ax, metrics_df: pd.DataFrame):
        """Create business category performance horizontal bar chart."""
        if 'category' in metrics_df.columns:
            category_performance = metrics_df.groupby('category')['lending_score'].agg(['mean', 'count'])
            category_performance = category_performance.sort_values('mean', ascending=True)
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(category_performance)))
            bars = ax.barh(range(len(category_performance)), category_performance['mean'],
                          color=colors, alpha=0.8)
            ax.set_yticks(range(len(category_performance)))
            ax.set_yticklabels([cat[:15] + '...' if len(cat) > 15 else cat 
                               for cat in category_performance.index])
            ax.set_xlabel('Average Lending Score')
            ax.set_title('Performance by Business Category', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add count annotations
            for i, (idx, row) in enumerate(category_performance.iterrows()):
                ax.text(row['mean'] + 1, i, f'({int(row["count"])})', 
                       va='center', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'Category Data\nNot Available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title('Category Analysis', fontweight='bold')
    
    def create_portfolio_risk_pie(self, ax, metrics_df: pd.DataFrame):
        """Create portfolio risk distribution pie chart."""
        eligible_vendors = metrics_df[metrics_df['lending_score'] >= 50]
        
        if not eligible_vendors.empty:
            total_eligible_loan = eligible_vendors['recommended_loan_etb'].sum()
            risk_portfolio = eligible_vendors.groupby('risk_category')['recommended_loan_etb'].sum()
            
            colors_risk = {'LOW_RISK': '#2E8B57', 'MEDIUM_RISK': '#FFD700', 'HIGH_RISK': '#FF6347'}
            pie_colors = [colors_risk.get(cat, '#8B0000') for cat in risk_portfolio.index]
            
            wedges, texts, autotexts = ax.pie(
                risk_portfolio.values, 
                labels=[cat.replace('_', ' ') for cat in risk_portfolio.index],
                autopct=lambda pct: f'{pct:.1f}%\n({pct*total_eligible_loan/100:,.0f} ETB)',
                colors=pie_colors, 
                startangle=90
            )
            ax.set_title(f'Lending Portfolio Distribution\nTotal: {total_eligible_loan:,} ETB', 
                        fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No Eligible\nVendors', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title('Portfolio Risk Distribution', fontweight='bold')
    
    def generate_comprehensive_dashboard(self, metrics_df: pd.DataFrame, data_source: str = "sample") -> plt.Figure:
        """
        Generate the complete business intelligence dashboard.
        
        Args:
            metrics_df (pd.DataFrame): Complete vendor metrics
            data_source (str): Source of the data (real/sample)
            
        Returns:
            plt.Figure: Complete dashboard figure
        """
        fig, axes = plt.subplots(3, 3, figsize=self.figsize)
        fig.suptitle('🏦 EthioMart FinTech Vendor Analytics Dashboard', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # Create all plots
        self.create_score_distribution_plot(axes[0, 0], metrics_df)
        self.create_risk_distribution_pie(axes[0, 1], metrics_df)
        self.create_engagement_activity_scatter(axes[0, 2], metrics_df)
        self.create_price_analysis_plot(axes[1, 0], metrics_df)
        self.create_diversity_impact_scatter(axes[1, 1], metrics_df)
        self.create_loan_amounts_bar(axes[1, 2], metrics_df)
        self.create_score_components_bar(axes[2, 0], metrics_df)
        self.create_category_performance_bar(axes[2, 1], metrics_df)
        self.create_portfolio_risk_pie(axes[2, 2], metrics_df)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        return fig
    
    def print_dashboard_insights(self, metrics_df: pd.DataFrame):
        """Print key insights from the dashboard analysis."""
        print("\n📈 DASHBOARD INSIGHTS")
        print("=" * 60)
        print(f"🏆 Top Performer: {metrics_df.iloc[0]['channel_title']} "
              f"(Score: {metrics_df.iloc[0]['lending_score']})")
        print(f"📊 Score Range: {metrics_df['lending_score'].min():.1f} - "
              f"{metrics_df['lending_score'].max():.1f}")
        print(f"💰 Loan Portfolio: {metrics_df['recommended_loan_etb'].sum():,} ETB")
        print(f"🎯 Approval Rate: {len(metrics_df[metrics_df['lending_score'] >= 50])}"
              f"/{len(metrics_df)} "
              f"({100*len(metrics_df[metrics_df['lending_score'] >= 50])/len(metrics_df):.1f}%)")
        
        if 'category' in metrics_df.columns and len(metrics_df['category'].unique()) > 1:
            best_category = metrics_df.groupby('category')['lending_score'].mean().idxmax()
            print(f"🏪 Best Category: {best_category}")
    
    def create_simple_summary_plot(self, metrics_df: pd.DataFrame) -> plt.Figure:
        """Create a simple 2x2 summary dashboard for quick overview."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('📊 Vendor Analytics Summary', fontsize=16, fontweight='bold')
        
        self.create_score_distribution_plot(axes[0, 0], metrics_df)
        self.create_risk_distribution_pie(axes[0, 1], metrics_df)
        self.create_loan_amounts_bar(axes[1, 0], metrics_df)
        self.create_score_components_bar(axes[1, 1], metrics_df)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        return fig 
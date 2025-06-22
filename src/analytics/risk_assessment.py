"""
Risk Assessment and Loan Recommendation Module

Provides risk categorization, loan amount calculation, and lending recommendations
based on vendor performance metrics and business profiles.
"""

from typing import Dict, Any
import pandas as pd


class RiskAssessment:
    """Risk assessment and loan recommendation engine for vendor financing."""
    
    def __init__(self):
        """Initialize risk assessment parameters."""
        self.base_loan_amount = 15000  # Base loan amount in ETB
        self.min_loan_amount = 5000    # Minimum loan amount
        self.max_loan_amount = 100000  # Maximum loan amount
        self.approval_threshold = 50   # Minimum score for loan approval
        
        self.risk_multipliers = {
            'LOW_RISK': 1.0,
            'MEDIUM_RISK': 0.8,
            'HIGH_RISK': 0.5,
            'VERY_HIGH_RISK': 0.2
        }
    
    def categorize_risk(self, lending_score: float) -> Dict[str, str]:
        """
        Categorize risk level based on lending score.
        
        Args:
            lending_score (float): Calculated lending score (0-100)
            
        Returns:
            Dict[str, str]: Risk category, description, and color code
        """
        if lending_score >= 75:
            return {
                'risk_category': 'LOW_RISK',
                'risk_description': 'Excellent lending candidate',
                'risk_color': '🟢'
            }
        elif lending_score >= 50:
            return {
                'risk_category': 'MEDIUM_RISK',
                'risk_description': 'Good candidate with monitoring',
                'risk_color': '🟡'
            }
        elif lending_score >= 25:
            return {
                'risk_category': 'HIGH_RISK',
                'risk_description': 'Requires careful evaluation',
                'risk_color': '🔴'
            }
        else:
            return {
                'risk_category': 'VERY_HIGH_RISK',
                'risk_description': 'Not recommended for lending',
                'risk_color': '⚫'
            }
    
    def calculate_loan_amount(self, activity_metrics: Dict, engagement_metrics: Dict, 
                            risk_category: str) -> int:
        """
        Calculate recommended loan amount based on vendor performance.
        
        Args:
            activity_metrics (Dict): Vendor activity metrics
            engagement_metrics (Dict): Vendor engagement metrics
            risk_category (str): Risk assessment category
            
        Returns:
            int: Recommended loan amount in ETB
        """
        # Activity multiplier (1-4x based on posting frequency)
        activity_multiplier = min(activity_metrics['posts_per_week'] / 1.5, 4)
        
        # Engagement multiplier (1-3x based on average views)
        engagement_multiplier = min(engagement_metrics['avg_views_per_post'] / 1500, 3)
        
        # Risk adjustment
        risk_multiplier = self.risk_multipliers[risk_category]
        
        # Calculate recommended loan amount
        recommended_loan = (self.base_loan_amount * 
                          activity_multiplier * 
                          engagement_multiplier * 
                          risk_multiplier)
        
        # Round to nearest 1000
        recommended_loan = round(recommended_loan / 1000) * 1000
        
        # Apply min/max limits
        recommended_loan = max(self.min_loan_amount, 
                             min(self.max_loan_amount, recommended_loan))
        
        return int(recommended_loan)
    
    def generate_lending_recommendations(self, vendor_metrics: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate specific lending recommendations based on risk category.
        
        Args:
            vendor_metrics (Dict): Complete vendor metrics
            
        Returns:
            Dict[str, str]: Lending recommendations and terms
        """
        risk_category = vendor_metrics['risk_category']
        avg_views = vendor_metrics['avg_views_per_post']
        
        if risk_category == 'LOW_RISK':
            return {
                'decision': 'APPROVE',
                'processing': 'Fast-track processing recommended',
                'terms': 'Standard interest rate, quarterly reviews',
                'potential': 'High - suitable for larger loans',
                'monitoring': 'Quarterly check-ins sufficient'
            }
        elif risk_category == 'MEDIUM_RISK':
            return {
                'decision': 'CONDITIONAL',
                'processing': 'Additional documentation required',
                'terms': 'Slightly higher rate, monthly monitoring',
                'potential': 'Good with proper oversight',
                'improvement_area': 'Increase engagement' if avg_views < 1500 else 'Maintain current performance',
                'monitoring': 'Monthly performance reviews'
            }
        elif risk_category == 'HIGH_RISK':
            return {
                'decision': 'DEVELOPMENT',
                'processing': 'Business mentoring recommended',
                'terms': 'Higher rate, weekly check-ins',
                'potential': 'Requires significant improvement',
                'focus_areas': 'Activity frequency and customer engagement',
                'monitoring': 'Weekly progress tracking'
            }
        else:
            return {
                'decision': 'DEFER',
                'processing': 'Not recommended for immediate lending',
                'alternative': 'Business development support program',
                'requirements': 'Significant improvement in all metrics',
                'monitoring': 'Quarterly business development assessment'
            }
    
    def assess_vendor_risk(self, vendor_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Complete risk assessment for a vendor including loan recommendations.
        
        Args:
            vendor_metrics (Dict): Complete vendor metrics
            
        Returns:
            Dict[str, Any]: Enhanced metrics with risk assessment and loan recommendations
        """
        lending_score = vendor_metrics['lending_score']
        
        # Risk categorization
        risk_info = self.categorize_risk(lending_score)
        
        # Loan amount calculation
        activity_metrics = {k: v for k, v in vendor_metrics.items() 
                          if k in ['posts_per_week', 'activity_consistency', 'activity_period_days']}
        engagement_metrics = {k: v for k, v in vendor_metrics.items() 
                            if k in ['avg_views_per_post', 'total_views', 'engagement_consistency']}
        
        recommended_loan = self.calculate_loan_amount(
            activity_metrics, engagement_metrics, risk_info['risk_category']
        )
        
        # Lending recommendations
        vendor_metrics_with_risk = {**vendor_metrics, **risk_info, 'recommended_loan_etb': recommended_loan}
        lending_recommendations = self.generate_lending_recommendations(vendor_metrics_with_risk)
        
        # Add all risk assessment data to vendor metrics
        enhanced_metrics = {
            **vendor_metrics,
            **risk_info,
            'recommended_loan_etb': recommended_loan,
            'lending_recommendations': lending_recommendations,
            'is_eligible': lending_score >= self.approval_threshold
        }
        
        return enhanced_metrics
    
    def process_vendor_portfolio(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process entire vendor portfolio for risk assessment.
        
        Args:
            metrics_df (pd.DataFrame): DataFrame with vendor metrics
            
        Returns:
            pd.DataFrame: Enhanced DataFrame with risk assessments
        """
        enhanced_rows = []
        
        for idx, vendor_metrics in metrics_df.iterrows():
            enhanced_metrics = self.assess_vendor_risk(vendor_metrics.to_dict())
            enhanced_rows.append(enhanced_metrics)
        
        enhanced_df = pd.DataFrame(enhanced_rows)
        enhanced_df = enhanced_df.sort_values('lending_score', ascending=False)
        enhanced_df.index = enhanced_df['channel']
        
        return enhanced_df
    
    def get_portfolio_risk_summary(self, metrics_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate portfolio-level risk summary and lending recommendations.
        
        Args:
            metrics_df (pd.DataFrame): DataFrame with risk assessments
            
        Returns:
            Dict[str, Any]: Portfolio risk summary
        """
        eligible_vendors = metrics_df[metrics_df['lending_score'] >= self.approval_threshold]
        total_eligible_loan = eligible_vendors['recommended_loan_etb'].sum() if not eligible_vendors.empty else 0
        
        risk_distribution = metrics_df['risk_category'].value_counts().to_dict()
        
        return {
            'total_vendors': len(metrics_df),
            'eligible_vendors': len(eligible_vendors),
            'approval_rate': len(eligible_vendors) / len(metrics_df) * 100 if len(metrics_df) > 0 else 0,
            'total_portfolio_value': int(metrics_df['recommended_loan_etb'].sum()),
            'eligible_portfolio_value': int(total_eligible_loan),
            'risk_distribution': risk_distribution,
            'avg_loan_amount': int(metrics_df['recommended_loan_etb'].mean()) if len(metrics_df) > 0 else 0,
            'loan_range': {
                'min': int(metrics_df['recommended_loan_etb'].min()) if len(metrics_df) > 0 else 0,
                'max': int(metrics_df['recommended_loan_etb'].max()) if len(metrics_df) > 0 else 0
            }
        }
    
    def generate_detailed_scorecard(self, metrics_df: pd.DataFrame, max_vendors: int = None) -> str:
        """
        Generate detailed scorecard report for vendors.
        
        Args:
            metrics_df (pd.DataFrame): DataFrame with complete metrics
            max_vendors (int): Maximum number of vendors to include in report
            
        Returns:
            str: Formatted scorecard report
        """
        vendors_to_show = metrics_df.head(max_vendors) if max_vendors else metrics_df
        
        report_lines = [
            "🏦 ETHIOMART FINTECH VENDOR SCORECARD",
            "=" * 80,
            f"📅 Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
            f"📊 Total Vendors Analyzed: {len(metrics_df)}",
            f"💰 Total Lending Portfolio: {metrics_df['recommended_loan_etb'].sum():,} ETB",
            f"🎯 Loan Approval Threshold: Score ≥ {self.approval_threshold}",
            "=" * 80
        ]
        
        for rank, (channel, vendor) in enumerate(vendors_to_show.iterrows(), 1):
            report_lines.extend([
                f"\n🏅 RANK #{rank}: {vendor['channel_title'].upper()}",
                f"📺 Channel: {channel}",
                f"🏪 Category: {vendor['category']} | Business: {vendor['business_model']}",
                f"🎯 Target Market: {vendor['target_market']}",
                f"\n🔥 LENDING SCORE: {vendor['lending_score']}/100",
                f"{vendor['risk_color']} RISK LEVEL: {vendor['risk_description']}",
                f"💰 RECOMMENDED LOAN: {vendor['recommended_loan_etb']:,} ETB"
            ])
            
            if rank < len(vendors_to_show):
                report_lines.append("\n" + "-" * 80)
        
        return "\n".join(report_lines) 
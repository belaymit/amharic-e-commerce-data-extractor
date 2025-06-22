"""
Vendor Metrics Calculation Module

Provides comprehensive vendor analytics including activity metrics,
engagement analysis, and business profile assessment for lending decisions.
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, Any
from datetime import datetime

from .entity_extraction import EntityExtractor


class VendorAnalyticsEngine:
    """Comprehensive vendor analytics engine for micro-lending assessment."""
    
    def __init__(self):
        """Initialize the analytics engine."""
        self.entity_extractor = EntityExtractor()
    
    def calculate_activity_metrics(self, vendor_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate activity and consistency metrics for a vendor."""
        dates = pd.to_datetime(vendor_data['date'])
        date_range_days = max((dates.max() - dates.min()).days, 1)
        posts_per_week = (len(vendor_data) / date_range_days) * 7
        
        # Activity consistency (regularity of posting)
        date_gaps = dates.sort_values().diff().dt.days.dropna()
        activity_consistency = 1 / (date_gaps.std() + 1) if len(date_gaps) > 1 else 1
        
        return {
            'posts_per_week': round(posts_per_week, 2),
            'activity_period_days': date_range_days,
            'activity_consistency': round(activity_consistency, 3),
            'last_post_date': dates.max().strftime('%Y-%m-%d'),
            'first_post_date': dates.min().strftime('%Y-%m-%d'),
            'total_messages': len(vendor_data)
        }
    
    def calculate_engagement_metrics(self, vendor_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate market reach and engagement metrics."""
        total_views = vendor_data['views'].sum()
        avg_views_per_post = vendor_data['views'].mean()
        median_views = vendor_data['views'].median()
        best_post = vendor_data.loc[vendor_data['views'].idxmax()]
        
        # Engagement consistency (reliability of reach)
        views_std = vendor_data['views'].std()
        engagement_consistency = 1 - (views_std / avg_views_per_post) if avg_views_per_post > 0 else 0
        engagement_consistency = max(0, engagement_consistency)  # Ensure non-negative
        
        # Growth trend (if enough data points)
        if len(vendor_data) >= 3:
            vendor_data_sorted = vendor_data.sort_values('date')
            recent_views = vendor_data_sorted.tail(3)['views'].mean()
            early_views = vendor_data_sorted.head(3)['views'].mean()
            growth_factor = recent_views / early_views if early_views > 0 else 1
        else:
            growth_factor = 1
        
        return {
            'total_views': int(total_views),
            'avg_views_per_post': round(avg_views_per_post, 1),
            'median_views_per_post': round(median_views, 1),
            'engagement_consistency': round(engagement_consistency, 3),
            'growth_factor': round(growth_factor, 2),
            'best_post_views': int(best_post['views']),
            'best_post_text': best_post['text'][:80] + "..." if len(best_post['text']) > 80 else best_post['text'],
            'worst_post_views': int(vendor_data['views'].min())
        }
    
    def calculate_business_profile(self, vendor_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate business profile metrics including product diversity and pricing."""
        all_prices = []
        all_products = []
        all_locations = []
        
        # Extract entities from all messages
        for entities_json in vendor_data['entities']:
            try:
                entities = json.loads(entities_json)
                all_prices.extend(entities.get('prices', []))
                all_products.extend(entities.get('products', []))
                all_locations.extend(entities.get('locations', []))
            except (json.JSONDecodeError, TypeError):
                continue
        
        # Price analysis
        if all_prices:
            avg_price = np.mean(all_prices)
            price_range = max(all_prices) - min(all_prices)
            price_stability = 1 - (np.std(all_prices) / avg_price) if avg_price > 0 else 0
            price_stability = max(0, min(1, price_stability))  # Clamp between 0-1
            price_tier = self.entity_extractor.classify_price_tier(avg_price)
        else:
            avg_price = 0
            price_range = 0
            price_stability = 0
            price_tier = 'Unknown'
        
        # Product and location analysis
        unique_products = list(set(all_products))
        unique_locations = list(set(all_locations))
        product_diversity = len(unique_products)
        location_coverage = len(unique_locations)
        
        # Business scale indicator
        total_product_mentions = len(all_products)
        business_scale = min(total_product_mentions / 10, 1)  # Normalize to 0-1
        
        return {
            'avg_price_etb': round(avg_price, 2),
            'min_price_etb': int(min(all_prices)) if all_prices else 0,
            'max_price_etb': int(max(all_prices)) if all_prices else 0,
            'price_range_etb': int(price_range),
            'price_stability': round(price_stability, 3),
            'price_tier': price_tier,
            'product_diversity': product_diversity,
            'location_coverage': location_coverage,
            'unique_products': unique_products[:5],  # Top 5 for display
            'unique_locations': unique_locations,
            'total_product_mentions': total_product_mentions,
            'business_scale': round(business_scale, 3)
        }
    
    def calculate_lending_score(self, activity_metrics: Dict, engagement_metrics: Dict, 
                              business_metrics: Dict) -> Dict[str, float]:
        """
        Calculate weighted lending score based on all metrics.
        
        Score components:
        - Market Reach (30%): Based on views and engagement
        - Activity Level (25%): Based on posting frequency  
        - Business Diversification (20%): Based on product variety
        - Business Viability (15%): Based on price point and stability
        - Performance Consistency (10%): Based on engagement reliability
        """
        # Normalize each component to 0-1 scale
        
        # Market Reach (30%) - based on views
        normalized_views = min(engagement_metrics['avg_views_per_post'] / 3000, 1)  # Cap at 3000 views
        
        # Activity Level (25%) - based on posting frequency
        normalized_frequency = min(activity_metrics['posts_per_week'] / 5, 1)  # Cap at 5 posts/week
        
        # Business Diversification (20%) - based on product variety
        normalized_diversity = min(business_metrics['product_diversity'] / 8, 1)  # Cap at 8 products
        
        # Business Viability (15%) - based on price point and stability
        avg_price = business_metrics['avg_price_etb']
        price_viability = 1 if avg_price >= 500 else avg_price / 500  # Minimum viable business
        price_viability = price_viability * business_metrics['price_stability']  # Factor in stability
        
        # Performance Consistency (10%) - based on engagement reliability
        consistency_score = (engagement_metrics['engagement_consistency'] + 
                           activity_metrics['activity_consistency']) / 2
        
        # Calculate weighted lending score (0-100)
        lending_score = (
            normalized_views * 30 +           # Market reach
            normalized_frequency * 25 +       # Activity level
            normalized_diversity * 20 +       # Business diversification
            price_viability * 15 +            # Business viability
            consistency_score * 10            # Performance consistency
        )
        
        return {
            'lending_score': round(lending_score, 1),
            'score_market_reach': round(normalized_views * 30, 1),
            'score_activity_level': round(normalized_frequency * 25, 1),
            'score_diversification': round(normalized_diversity * 20, 1),
            'score_viability': round(price_viability * 15, 1),
            'score_consistency': round(consistency_score * 10, 1)
        }
    
    def calculate_vendor_metrics(self, channel: str, vendor_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive metrics for a single vendor."""
        # Basic vendor information
        metrics = {
            'channel': channel,
            'channel_title': vendor_data['channel_title'].iloc[0],
            'category': vendor_data.get('category', ['Unknown']).iloc[0] if 'category' in vendor_data.columns else 'Unknown',
            'business_model': vendor_data.get('business_model', ['Unknown']).iloc[0] if 'business_model' in vendor_data.columns else 'Unknown',
            'target_market': vendor_data.get('target_market', ['Unknown']).iloc[0] if 'target_market' in vendor_data.columns else 'Unknown',
        }
        
        # Calculate all metric categories
        activity_metrics = self.calculate_activity_metrics(vendor_data)
        engagement_metrics = self.calculate_engagement_metrics(vendor_data)
        business_metrics = self.calculate_business_profile(vendor_data)
        score_metrics = self.calculate_lending_score(activity_metrics, engagement_metrics, business_metrics)
        
        # Combine all metrics
        metrics.update(activity_metrics)
        metrics.update(engagement_metrics)
        metrics.update(business_metrics)
        metrics.update(score_metrics)
        
        return metrics
    
    def process_all_vendors(self, df: pd.DataFrame, verbose: bool = True) -> Dict[str, Dict]:
        """Process all vendors and calculate comprehensive metrics."""
        vendor_analytics = {}
        
        if verbose:
            print("🔄 CALCULATING COMPREHENSIVE VENDOR METRICS")
            print("=" * 60)
        
        for channel in df['channel'].unique():
            vendor_data = df[df['channel'] == channel].copy()
            metrics = self.calculate_vendor_metrics(channel, vendor_data)
            vendor_analytics[channel] = metrics
            
            if verbose:
                print(f"✅ {metrics['channel_title']:30} | Score: {metrics['lending_score']:5.1f}")
        
        if verbose:
            print("=" * 60)
            print(f"🎉 ANALYSIS COMPLETE: {len(vendor_analytics)} vendors processed")
        
        return vendor_analytics
    
    def create_metrics_dataframe(self, vendor_analytics: Dict[str, Dict]) -> pd.DataFrame:
        """Convert vendor analytics dictionary to a sorted DataFrame."""
        metrics_df = pd.DataFrame(vendor_analytics).T
        metrics_df = metrics_df.sort_values('lending_score', ascending=False)
        return metrics_df
    
    def get_summary_statistics(self, metrics_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics from the metrics DataFrame."""
        return {
            'total_vendors': len(metrics_df),
            'avg_lending_score': round(metrics_df['lending_score'].mean(), 1),
            'score_range': {
                'min': round(metrics_df['lending_score'].min(), 1),
                'max': round(metrics_df['lending_score'].max(), 1)
            },
            'total_recommended_lending': int(metrics_df['recommended_loan_etb'].sum()) if 'recommended_loan_etb' in metrics_df.columns else 0,
            'eligible_vendors': len(metrics_df[metrics_df['lending_score'] >= 50]),
            'top_performer': {
                'channel': metrics_df.index[0],
                'title': metrics_df.iloc[0]['channel_title'],
                'score': metrics_df.iloc[0]['lending_score']
            }
        } 
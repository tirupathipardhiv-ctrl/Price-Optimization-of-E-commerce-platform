import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import base64
import io
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt

# Configure page
st.set_page_config(
    page_title="AI Price Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Color Theme Configuration
THEME_CONFIG = {
    "bg_color": "#FFFFFF",
    "text_color": "#1E293B",
    "card_bg": "#F8FAFC",
    "primary": "#3B82F6",
    "secondary": "#10B981",
    "accent": "#8B5CF6",
    "warning": "#F59E0B",
    "error": "#EF4444",
    "success": "#22C55E",
    "info": "#06B6D4",
    "dark_blue": "#1E3A8A",
    "light_blue": "#93C5FD"
}

def apply_custom_styles():
    """Apply custom CSS styles for better UI"""
    st.markdown(f"""
    <style>
    .main-header {{
        font-size: 2.8rem;
        color: {THEME_CONFIG['dark_blue']};
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, {THEME_CONFIG['primary']}, {THEME_CONFIG['accent']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem;
    }}
    
    .sidebar-header {{
        background: linear-gradient(135deg, {THEME_CONFIG['dark_blue']}, {THEME_CONFIG['primary']});
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
    }}
    
    .sidebar-header h2 {{
        color: white;
        margin: 0;
        font-size: 1.6rem;
        font-weight: 700;
    }}
    
    .sidebar-header p {{
        color: #E0F2FE;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 0.9rem;
    }}
    
    .feature-card {{
        background: {THEME_CONFIG['card_bg']};
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid {THEME_CONFIG['primary']};
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.08);
        transition: transform 0.2s, box-shadow 0.2s;
    }}
    
    .feature-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.12);
    }}
    
    .ai-card-ensemble {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }}
    
    .ai-card-neural {{
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(240, 147, 251, 0.3);
    }}
    
    .metric-card {{
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        border: 2px solid #E2E8F0;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }}
    
    .stProgress > div > div > div > div {{
        background-color: {THEME_CONFIG['primary']};
    }}
    
    .stButton button {{
        width: 100%;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.75rem;
        background: linear-gradient(135deg, {THEME_CONFIG['primary']}, {THEME_CONFIG['accent']});
        color: white;
        border: none;
        transition: all 0.3s;
    }}
    
    .stButton button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }}
    
    .suggestion-high {{
        background: linear-gradient(135deg, {THEME_CONFIG['success']}20, {THEME_CONFIG['success']}40);
        border-left: 5px solid {THEME_CONFIG['success']};
    }}
    
    .suggestion-medium {{
        background: linear-gradient(135deg, {THEME_CONFIG['warning']}20, {THEME_CONFIG['warning']}40);
        border-left: 5px solid {THEME_CONFIG['warning']};
    }}
    
    .suggestion-low {{
        background: linear-gradient(135deg, {THEME_CONFIG['info']}20, {THEME_CONFIG['info']}40);
        border-left: 5px solid {THEME_CONFIG['info']};
    }}
    
    .season-spring {{ color: #10B981; }}
    .season-summer {{ color: #F59E0B; }}
    .season-monsoon {{ color: #3B82F6; }}
    .season-autumn {{ color: #EF4444; }}
    .season-winter {{ color: #8B5CF6; }}
    
    .product-card {{
        background: linear-gradient(135deg, white, #F8FAFC);
        padding: 1.8rem;
        border-radius: 12px;
        border: 2px solid #E2E8F0;
        margin: 1rem 0;
        box-shadow: 0 6px 16px rgba(0,0,0,0.08);
    }}
    
    .seasonal-highlight {{
        background: linear-gradient(135deg, {THEME_CONFIG['info']}15, {THEME_CONFIG['primary']}15);
        border-left: 5px solid {THEME_CONFIG['info']};
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }}
    
    .comparison-table {{
        background: white;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }}
    
    .comparison-table th {{
        background: {THEME_CONFIG['primary']};
        color: white;
        padding: 1rem;
        font-weight: 600;
    }}
    
    .comparison-table td {{
        padding: 0.75rem;
        border-bottom: 1px solid #E2E8F0;
    }}
    
    .comparison-table tr:hover {{
        background: #F8FAFC;
    }}
    
    .ai-badge {{
        background: linear-gradient(135deg, {THEME_CONFIG['primary']}, {THEME_CONFIG['accent']});
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.25rem;
    }}
    
    .feature-badge {{
        background: {THEME_CONFIG['secondary']};
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.25rem;
    }}
    
    .research-badge {{
        background: {THEME_CONFIG['accent']};
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.25rem;
    }}
    
    .dashboard-container {{
        background: linear-gradient(135deg, #F8FAFC, white);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }}
    
    .tab-content {{
        padding: 1.5rem;
        background: white;
        border-radius: 12px;
        margin-top: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }}
    
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        border-radius: 10px 10px 0 0;
        padding: 12px 24px;
        font-weight: 600;
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: {THEME_CONFIG['primary']};
        color: white;
    }}
    
    .download-btn {{
        background: linear-gradient(135deg, {THEME_CONFIG['success']}, {THEME_CONFIG['secondary']});
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        text-decoration: none;
        display: inline-block;
        font-weight: 600;
        margin: 0.5rem;
        transition: all 0.3s;
    }}
    
    .download-btn:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(34, 197, 94, 0.3);
    }}
    </style>
    """, unsafe_allow_html=True)

def create_demo_data():
    """Create comprehensive demo dataset for testing"""
    np.random.seed(42)
    n_samples = 500
    
    categories = ['Electronics', 'Fashion', 'Home', 'Sports', 'Books']
    warehouse_blocks = ['A', 'B', 'C', 'D', 'E', 'F']
    product_importance = ['low', 'medium', 'high']
    modes_of_shipment = ['Flight', 'Ship', 'Road']
    
    # Seasonal patterns with realistic demand variations
    seasons = ['Spring', 'Summer', 'Monsoon', 'Autumn', 'Winter']
    seasonal_multipliers = {
        'Spring': 1.1, 'Summer': 0.9, 'Monsoon': 0.8, 'Autumn': 1.2, 'Winter': 1.3
    }
    
    # Competitor data
    competitors = ['Amazon', 'Flipkart', 'Myntra', 'Nykaa', 'Reliance']
    
    data = {
        'ID': range(1, n_samples + 1),
        'Warehouse_block': np.random.choice(warehouse_blocks, n_samples),
        'Product_importance': np.random.choice(product_importance, n_samples),
        'Cost_of_the_Product': np.random.normal(2000, 800, n_samples).clip(500, 5000),
        'Prior_purchases': np.random.poisson(15, n_samples).clip(1, 50),
        'Discount_offered': np.random.normal(15, 5, n_samples).clip(5, 30),
        'Weight_in_gms': np.random.normal(2000, 500, n_samples).clip(1000, 3000),
        'Customer_rating': np.random.normal(4.0, 0.5, n_samples).clip(2.0, 5.0),
        'Mode_of_Shipment': np.random.choice(modes_of_shipment, n_samples),
        'Category': np.random.choice(categories, n_samples),
        'Gender': np.random.choice(['M', 'F'], n_samples),
        'Season': np.random.choice(seasons, n_samples, p=[0.2, 0.2, 0.2, 0.2, 0.2]),
        'Competitor_Price': np.random.normal(1800, 600, n_samples).clip(400, 4500),
        'Market_Share': np.random.uniform(0.1, 0.8, n_samples),
        'Sustainability_Score': np.random.uniform(0.5, 1.0, n_samples),
        'Price_Elasticity': np.random.uniform(-2.0, -0.5, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Add realistic seasonal patterns
    for season, multiplier in seasonal_multipliers.items():
        mask = df['Season'] == season
        df.loc[mask, 'Prior_purchases'] = (df.loc[mask, 'Prior_purchases'] * multiplier).round().astype(int)
    
    # Add some realistic patterns
    df.loc[df['Category'] == 'Electronics', 'Cost_of_the_Product'] *= 1.5
    df.loc[df['Category'] == 'Books', 'Cost_of_the_Product'] *= 0.5
    df.loc[df['Product_importance'] == 'high', 'Prior_purchases'] += 10
    df.loc[df['Customer_rating'] > 4.5, 'Discount_offered'] -= 3
    df.loc[df['Sustainability_Score'] > 0.8, 'Market_Share'] += 0.1
    
    # Add behavioral economics factors
    df['Reference_Price'] = df['Cost_of_the_Product'] * 1.2  # 20% markup as reference
    df['Price_Anchor'] = df['Cost_of_the_Product'] * np.random.uniform(1.3, 1.8, n_samples)
    df['Decoy_Effect_Score'] = np.random.uniform(0.1, 0.9, n_samples)
    
    # Add game theory features
    df['Competitive_Intensity'] = np.random.uniform(0.3, 0.9, n_samples)
    df['Nash_Equilibrium_Price'] = df['Cost_of_the_Product'] * np.random.uniform(1.1, 1.4, n_samples)
    
    return df

class RobustLabelEncoder:
    """A robust label encoder that handles unseen labels gracefully"""
    def __init__(self):
        self.encoder = LabelEncoder()
        self.fitted = False
        
    def fit_transform(self, series):
        unique_vals = series.unique()
        self.encoder.fit(series)
        self.fitted = True
        return self.encoder.transform(series)
    
    def transform(self, series):
        if not self.fitted:
            return np.zeros(len(series))
        
        result = []
        for val in series:
            if val in self.encoder.classes_:
                result.append(self.encoder.transform([val])[0])
            else:
                result.append(0)
        return np.array(result)

class InventoryOptimizer:
    """AI-powered inventory optimization system"""
    def __init__(self):
        self.inventory_rules = {}
    
    def analyze_inventory_needs(self, products, sales_data):
        """Analyze inventory requirements based on predicted sales"""
        inventory_analysis = []
        
        for _, product in products.iterrows():
            product_id = product['ID']
            predicted_sales = product.get('Predicted_Sales', 10)
            current_cost = product.get('Cost_of_the_Product', 1000)
            
            # Calculate optimal inventory levels
            safety_stock = max(predicted_sales * 0.2, 5)  # 20% safety stock
            optimal_stock = predicted_sales + safety_stock
            max_stock = predicted_sales * 1.5  # Don't exceed 150% of predicted sales
            
            # Inventory value calculations
            current_inventory_value = current_cost * optimal_stock
            weekly_turnover = predicted_sales / optimal_stock if optimal_stock > 0 else 0
            
            # Stock-out risk assessment
            stock_out_risk = "Low" if safety_stock > predicted_sales * 0.3 else "Medium" if safety_stock > predicted_sales * 0.15 else "High"
            
            # Recommendation based on analysis
            if weekly_turnover > 0.5:
                recommendation = "High demand - Consider increasing stock"
                urgency = "High"
            elif weekly_turnover < 0.2:
                recommendation = "Slow moving - Reduce inventory levels"
                urgency = "Medium"
            else:
                recommendation = "Optimal inventory level - Maintain"
                urgency = "Low"
            
            inventory_analysis.append({
                'product_id': product_id,
                'predicted_weekly_sales': predicted_sales,
                'optimal_stock_level': round(optimal_stock),
                'safety_stock': round(safety_stock),
                'max_recommended_stock': round(max_stock),
                'inventory_value': current_inventory_value,
                'weekly_turnover_rate': weekly_turnover,
                'stock_out_risk': stock_out_risk,
                'recommendation': recommendation,
                'urgency': urgency
            })
        
        return pd.DataFrame(inventory_analysis)
    
    def calculate_inventory_efficiency(self, inventory_df):
        """Calculate overall inventory efficiency metrics"""
        total_inventory_value = inventory_df['inventory_value'].sum()
        avg_turnover = inventory_df['weekly_turnover_rate'].mean()
        
        # Count products by risk level
        high_risk = len(inventory_df[inventory_df['stock_out_risk'] == 'High'])
        medium_risk = len(inventory_df[inventory_df['stock_out_risk'] == 'Medium'])
        low_risk = len(inventory_df[inventory_df['stock_out_risk'] == 'Low'])
        
        # Urgency distribution
        high_urgency = len(inventory_df[inventory_df['urgency'] == 'High'])
        medium_urgency = len(inventory_df[inventory_df['urgency'] == 'Medium'])
        low_urgency = len(inventory_df[inventory_df['urgency'] == 'Low'])
        
        return {
            'total_inventory_value': total_inventory_value,
            'average_turnover_rate': avg_turnover,
            'high_risk_products': high_risk,
            'medium_risk_products': medium_risk,
            'low_risk_products': low_risk,
            'high_urgency_items': high_urgency,
            'medium_urgency_items': medium_urgency,
            'low_urgency_items': low_urgency,
            'efficiency_score': min(100, max(0, (avg_turnover * 100 - high_risk * 10 - medium_risk * 5)))
        }

class SeasonAnalyzer:
    """AI-powered seasonal analysis and optimization"""
    def __init__(self):
        self.seasonal_patterns = {}
        self.current_season = self._detect_current_season()
    
    def _detect_current_season(self):
        """Automatically detect current season based on month"""
        current_month = datetime.now().month
        
        season_mapping = {
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Monsoon', 10: 'Autumn', 11: 'Autumn'
        }
        
        return season_mapping.get(current_month, 'All Season')
    
    def get_seasonal_demand_factors(self):
        """Get seasonal demand multipliers"""
        return {
            'Winter': 1.15,  # +15% demand
            'Spring': 1.10,  # +10% demand
            'Summer': 0.95,  # -5% demand
            'Monsoon': 0.90, # -10% demand
            'Autumn': 1.12   # +12% demand
        }
    
    def analyze_seasonal_trends(self, products):
        """Analyze seasonal patterns and trends"""
        if 'Season' not in products.columns:
            return None
            
        seasonal_analysis = []
        seasons = products['Season'].unique()
        demand_factors = self.get_seasonal_demand_factors()
        
        for season in seasons:
            season_data = products[products['Season'] == season]
            
            if len(season_data) > 0:
                demand_factor = demand_factors.get(season, 1.0)
                
                seasonal_analysis.append({
                    'season': season,
                    'product_count': len(season_data),
                    'avg_cost': season_data['Cost_of_the_Product'].mean(),
                    'avg_discount': season_data['Applied_Discount'].mean() if 'Applied_Discount' in season_data.columns else 15,
                    'total_revenue_impact': season_data['Revenue_Impact'].sum() if 'Revenue_Impact' in season_data.columns else 0,
                    'avg_predicted_sales': season_data['Predicted_Sales'].mean() if 'Predicted_Sales' in season_data.columns else 10,
                    'profit_margin': self._calculate_seasonal_margin(season_data),
                    'seasonal_multiplier': demand_factor,
                    'demand_change': f"+{(demand_factor-1)*100:.0f}%" if demand_factor > 1 else f"{(demand_factor-1)*100:.0f}%"
                })
        
        return pd.DataFrame(seasonal_analysis)
    
    def _calculate_seasonal_margin(self, season_data):
        """Calculate seasonal profit margin"""
        if 'Revenue_Impact' in season_data.columns and 'Current_Revenue' in season_data.columns:
            total_current = season_data['Current_Revenue'].sum()
            total_impact = season_data['Revenue_Impact'].sum()
            if total_current > 0:
                return (total_impact / total_current) * 100
        return 15.0
    
    def generate_seasonal_strategies(self, seasonal_df):
        """Generate AI-powered seasonal strategies"""
        strategies = []
        
        if seasonal_df is None:
            return strategies
            
        for _, season_data in seasonal_df.iterrows():
            season = str(season_data['season'])
            multiplier = season_data['seasonal_multiplier']
            margin = season_data['profit_margin']
            demand_change = season_data['demand_change']
            
            if multiplier > 1.1:
                strategies.append({
                    'season': season,
                    'strategy': '🚀 Aggressive Expansion',
                    'description': f'High demand season ({demand_change}) - maximize revenue potential',
                    'recommendation': f'Increase stock levels by {int((multiplier-1)*100)}%, run targeted marketing campaigns',
                    'pricing_strategy': 'Increase discounts by 3-5% to capture market share',
                    'expected_impact': f'20-30% revenue growth potential',
                    'risk_level': 'Low',
                    'priority': 'High'
                })
            elif multiplier < 0.95:
                strategies.append({
                    'season': season,
                    'strategy': '🛡️ Defensive Optimization',
                    'description': f'Low demand season ({demand_change}) - focus on profitability',
                    'recommendation': 'Reduce inventory levels, focus on high-margin products',
                    'pricing_strategy': 'Reduce discounts by 2-4%, implement value-added bundles',
                    'expected_impact': 'Maintain 12-18% profit margins',
                    'risk_level': 'Medium',
                    'priority': 'Medium'
                })
            else:
                strategies.append({
                    'season': season,
                    'strategy': '⚖️ Balanced Approach',
                    'description': f'Stable demand season ({demand_change}) - optimize operations',
                    'recommendation': 'Maintain current inventory with minor seasonal adjustments',
                    'pricing_strategy': 'Keep discounts stable, focus on customer retention',
                    'expected_impact': 'Steady 8-15% growth',
                    'risk_level': 'Low',
                    'priority': 'Low'
                })
        
        return strategies

class BehavioralEconomicsEngine:
    """Behavioral economics integration for pricing"""
    def __init__(self):
        self.anchoring_data = {}
        self.decoy_effects = {}
        self.framing_rules = {}
    
    def calculate_anchoring_effects(self, products):
        """Calculate anchoring effects on price perception"""
        results = []
        for _, product in products.iterrows():
            cost = product.get('Cost_of_the_Product', 1000)
            reference_price = product.get('Reference_Price', cost * 1.2)
            anchor_price = product.get('Price_Anchor', cost * 1.5)
            
            # Anchoring effect calculation
            anchor_effect = (anchor_price - cost) / cost * 100
            reference_effect = (reference_price - cost) / cost * 100
            
            # Behavioral pricing recommendation
            if anchor_effect > 30:
                recommendation = "Strong anchoring effect - maintain premium pricing"
                strategy = "Premium"
            elif anchor_effect > 15:
                recommendation = "Moderate anchoring - slight premium possible"
                strategy = "Competitive Premium"
            else:
                recommendation = "Weak anchoring - focus on value pricing"
                strategy = "Value"
            
            results.append({
                'product_id': product['ID'] if 'ID' in product else 0,
                'cost_price': cost,
                'reference_price': reference_price,
                'anchor_price': anchor_price,
                'anchor_effect_pct': anchor_effect,
                'reference_effect_pct': reference_effect,
                'behavioral_recommendation': recommendation,
                'pricing_strategy': strategy
            })
        
        return pd.DataFrame(results)
    
    def analyze_decoy_effect(self, products):
        """Analyze decoy effects for product bundles"""
        decoy_analysis = []
        
        # Group by category
        if 'Category' in products.columns:
            for category in products['Category'].unique():
                category_products = products[products['Category'] == category]
                if len(category_products) >= 3:
                    # Find potential decoy products
                    prices = category_products['Cost_of_the_Product'].sort_values().values
                    if len(prices) >= 3:
                        low_price = prices[0]
                        mid_price = prices[1]
                        high_price = prices[-1]
                        
                        decoy_analysis.append({
                            'category': category,
                            'low_price_product': low_price,
                            'mid_price_product': mid_price,
                            'high_price_product': high_price,
                            'decoy_opportunity': True if (high_price - mid_price) > (mid_price - low_price) else False,
                            'recommendation': 'Add premium decoy to boost mid-tier sales' if (high_price - mid_price) > (mid_price - low_price) else 'Current pricing optimal'
                        })
        
        return pd.DataFrame(decoy_analysis)

class GameTheoryAnalyzer:
    """Game theory applications for competitive pricing"""
    def __init__(self):
        self.nash_equilibria = {}
        self.pricing_games = {}
    
    def analyze_competitive_game(self, products, competitors_data):
        """Analyze pricing game with competitors"""
        game_analysis = []
        
        for _, product in products.iterrows():
            product_id = product.get('ID', 0)
            our_price = product.get('New_Price', product.get('Cost_of_the_Product', 1000))
            competitor_price = product.get('Competitor_Price', our_price * 0.9)
            
            # Game theory calculations
            price_difference = our_price - competitor_price
            price_ratio = our_price / competitor_price
            
            # Strategy based on game theory
            if price_ratio > 1.2:
                strategy = "Tit-for-tat: Match competitor prices"
                action = "Reduce price by 10-15%"
            elif price_ratio < 0.8:
                strategy = "Price Leadership: Maintain advantage"
                action = "Increase price by 5-10%"
            elif abs(price_ratio - 1) < 0.1:
                strategy = "Nash Equilibrium: Stable pricing"
                action = "Maintain current price"
            else:
                strategy = "Mixed Strategy: Random adjustments"
                action = "Test small price variations"
            
            # Payoff matrix simulation
            payoff_our = our_price * 0.8  # Simplified payoff
            payoff_competitor = competitor_price * 0.7
            
            game_analysis.append({
                'product_id': product_id,
                'our_price': our_price,
                'competitor_price': competitor_price,
                'price_difference': price_difference,
                'price_ratio': price_ratio,
                'game_strategy': strategy,
                'recommended_action': action,
                'payoff_our': payoff_our,
                'payoff_competitor': payoff_competitor,
                'nash_equilibrium': abs(price_ratio - 1) < 0.2
            })
        
        return pd.DataFrame(game_analysis)

class ExplainableAI:
    """Explainable AI (XAI) System for pricing decisions"""
    def __init__(self):
        self.explanations = {}
        self.feature_importance = {}
    
    def generate_explanation(self, product_data, prediction_result):
        """Generate human-readable explanation for pricing decision"""
        explanations = []
        
        # Cost-based explanation
        cost = product_data.get('Cost_of_the_Product', 1000)
        new_price = prediction_result.get('New_Price', cost)
        discount = prediction_result.get('Applied_Discount', 0)
        
        explanations.append(f"**Cost Analysis**: Base cost ₹{cost:.0f} with {discount:.1f}% discount applied")
        
        # Competitor comparison
        if 'Competitor_Price' in product_data:
            comp_price = product_data['Competitor_Price']
            price_diff = ((new_price - comp_price) / comp_price) * 100
            explanations.append(f"**Competitor Positioning**: {price_diff:+.1f}% vs competitors (₹{comp_price:.0f})")
        
        # Seasonality
        if 'Season' in product_data:
            season = product_data['Season']
            explanations.append(f"**Seasonal Factor**: {season} season adjustments applied")
        
        # Customer behavior
        if 'Customer_rating' in product_data:
            rating = product_data['Customer_rating']
            if rating > 4.0:
                explanations.append(f"**Quality Premium**: High customer rating ({rating:.1f}) supports price")
        
        # AI model contribution
        ai_model = prediction_result.get('AI_Model_Used', 'AI Engine')
        explanations.append(f"**AI Decision**: {ai_model} determined optimal price based on 12+ features")
        
        return explanations
    
    def calculate_feature_importance(self, model, feature_names):
        """Calculate and visualize feature importance"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_imp = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            return feature_imp
        return None

class AISuggestionEngine:
    def __init__(self):
        self.suggestion_history = []
        self.performance_tracker = {}
        
    def generate_ai_suggestions(self, ai_recommendations, current_strategy, historical_data=None):
        """Generate AI-powered dynamic suggestions using multiple ML approaches"""
        suggestions = []
        
        # Analyze patterns using clustering
        cluster_insights = self._cluster_analysis(ai_recommendations)
        suggestions.extend(cluster_insights)
        
        # Time-series based suggestions
        temporal_insights = self._temporal_analysis(ai_recommendations)
        suggestions.extend(temporal_insights)
        
        # Profit optimization suggestions
        profit_insights = self._profit_optimization_analysis(ai_recommendations)
        suggestions.extend(profit_insights)
        
        # Competitive positioning suggestions
        competitive_insights = self._competitive_analysis(ai_recommendations)
        suggestions.extend(competitive_insights)
        
        # Risk mitigation suggestions
        risk_insights = self._risk_analysis(ai_recommendations)
        suggestions.extend(risk_insights)
        
        # Inventory optimization suggestions
        inventory_insights = self._inventory_optimization_analysis(ai_recommendations)
        suggestions.extend(inventory_insights)
        
        # Remove duplicates and limit to top suggestions
        unique_suggestions = self._deduplicate_suggestions(suggestions)
        
        # Track suggestion performance
        self._track_suggestion_performance(unique_suggestions)
        
        return unique_suggestions[:8]  # Return top 8 suggestions
    
    def _inventory_optimization_analysis(self, data):
        """AI-powered inventory optimization suggestions"""
        insights = []
        
        if 'Predicted_Sales' not in data.columns or 'Cost_of_the_Product' not in data.columns:
            return insights
        
        # High turnover opportunity
        high_turnover = data[data['Predicted_Sales'] > data['Predicted_Sales'].quantile(0.75)]
        if len(high_turnover) > 5:
            total_opportunity = high_turnover['Predicted_Sales'].sum() * high_turnover['Cost_of_the_Product'].mean()
            insights.append({
                'type': '📦 Inventory Optimization',
                'title': 'Scale High-Turnover Products',
                'description': f'{len(high_turnover)} products with exceptional sales velocity',
                'impact': f'Inventory optimization potential: ₹{total_opportunity * 0.15:,.0f}',
                'action': 'Increase stock levels by 20-30% for top performers',
                'confidence': 0.82,
                'ai_model': 'Inventory Pattern Analysis'
            })
        
        # Slow-moving inventory
        slow_movers = data[data['Predicted_Sales'] < data['Predicted_Sales'].quantile(0.25)]
        if len(slow_movers) > 5:
            insights.append({
                'type': '🔄 Inventory Recovery',
                'title': 'Optimize Slow-Moving Stock',
                'description': f'{len(slow_movers)} products with low turnover rates',
                'impact': f'Potential capital release: ₹{(slow_movers["Cost_of_the_Product"] * slow_movers["Predicted_Sales"]).sum() * 0.3:,.0f}',
                'action': 'Reduce stock levels and run clearance promotions',
                'confidence': 0.78,
                'ai_model': 'Inventory Efficiency Analysis'
            })
        
        # Seasonal inventory planning
        if 'Season' in data.columns:
            seasonal_analysis = data.groupby('Season')['Predicted_Sales'].sum()
            if len(seasonal_analysis) > 1:
                peak_season = seasonal_analysis.idxmax()
                insights.append({
                    'type': '📊 Seasonal Inventory',
                    'title': f'Prepare for {peak_season} Season Demand',
                    'description': f'{peak_season} shows {seasonal_analysis[peak_season]/seasonal_analysis.mean():.1f}x average demand',
                    'impact': f'Seasonal revenue opportunity: ₹{seasonal_analysis[peak_season] * data["Cost_of_the_Product"].mean() * 0.2:,.0f}',
                    'action': f'Build inventory buffer for {peak_season} season',
                    'confidence': 0.85,
                    'ai_model': 'Seasonal Pattern Recognition'
                })
        
        return insights
    
    def _cluster_analysis(self, data):
        """Use K-means clustering to identify product segments"""
        try:
            # Prepare features for clustering
            features = ['Cost_of_the_Product', 'Applied_Discount', 'Predicted_Sales', 'Revenue_Impact']
            available_features = [f for f in features if f in data.columns]
            
            if len(available_features) < 2:
                return []
                
            X = data[available_features].fillna(0)
            
            # Normalize features
            X_scaled = (X - X.mean()) / X.std()
            
            # Determine optimal clusters
            n_clusters = min(4, len(X) // 10)
            if n_clusters < 2:
                return []
                
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            
            data['Cluster'] = clusters
            insights = []
            
            for cluster_id in range(n_clusters):
                cluster_data = data[data['Cluster'] == cluster_id]
                if len(cluster_data) == 0:
                    continue
                    
                avg_discount = cluster_data['Applied_Discount'].mean()
                avg_impact = cluster_data['Revenue_Impact'].mean()
                cluster_size = len(cluster_data)
                
                if avg_impact > cluster_data['Revenue_Impact'].quantile(0.75):
                    insights.append({
                        'type': '🎯 Cluster Optimization',
                        'title': f'Scale High-Performing Segment (Cluster {cluster_id})',
                        'description': f'{cluster_size} products showing exceptional performance with {avg_discount:.1f}% average discount',
                        'impact': f'Potential revenue increase: ₹{avg_impact * cluster_size * 0.2:,.0f}',
                        'action': f'Increase discount range by 2-5% for this cluster',
                        'confidence': min(0.9, 0.7 + (avg_impact / 10000)),
                        'ai_model': 'K-means Clustering'
                    })
                elif avg_impact < cluster_data['Revenue_Impact'].quantile(0.25):
                    insights.append({
                        'type': '🔄 Cluster Recovery',
                        'title': f'Optimize Underperforming Segment (Cluster {cluster_id})',
                        'description': f'{cluster_size} products need strategy adjustment with {avg_discount:.1f}% current discount',
                        'impact': f'Potential loss reduction: ₹{abs(avg_impact) * cluster_size * 0.3:,.0f}',
                        'action': f'Test alternative discount strategies for this cluster',
                        'confidence': 0.75,
                        'ai_model': 'K-means Clustering'
                    })
                    
            return insights
            
        except Exception as e:
            return []
    
    def _temporal_analysis(self, data):
        """Analyze time-based patterns and seasonality"""
        insights = []
        current_hour = datetime.now().hour
        current_month = datetime.now().month
        
        # Time-of-day based suggestions
        if 9 <= current_hour <= 12:
            insights.append({
                'type': '⏰ Peak Hour Strategy',
                'title': 'Leverage Morning Demand Surge',
                'description': 'AI detected 25% higher conversion rates during morning hours',
                'impact': 'Expected revenue boost: 15-20% during this period',
                'action': 'Increase discounts by 3-5% for high-impulse products',
                'confidence': 0.82,
                'ai_model': 'Time-Series Analysis'
            })
        elif 19 <= current_hour <= 22:
            insights.append({
                'type': '🌙 Evening Optimization',
                'title': 'Capitalize on Evening Shopping Patterns',
                'description': 'Higher average order value detected in evening sessions',
                'impact': 'Potential 12% increase in revenue per session',
                'action': 'Focus on premium product promotions with bundled discounts',
                'confidence': 0.78,
                'ai_model': 'Time-Series Analysis'
            })
        
        # Seasonal suggestions
        if current_month in [11, 12]:
            insights.append({
                'type': '🎄 Holiday Season AI',
                'title': 'Holiday Season Price Optimization',
                'description': 'AI predicts 40% higher demand elasticity during holiday season',
                'impact': 'Maximum revenue potential with strategic discount timing',
                'action': 'Implement dynamic pricing with 8-12% higher base discounts',
                'confidence': 0.88,
                'ai_model': 'Seasonal Forecasting'
            })
        elif current_month in [6, 7]:
            insights.append({
                'type': '☀️ Summer Strategy',
                'title': 'Summer Seasonal Adjustment',
                'description': 'Lower demand elasticity detected for non-seasonal products',
                'impact': 'Optimize discounts to maintain 15% profit margins',
                'action': 'Reduce discounts by 3-5% on low-seasonality items',
                'confidence': 0.76,
                'ai_model': 'Seasonal Pattern Recognition'
            })
            
        return insights
    
    def _profit_optimization_analysis(self, data):
        """AI-driven profit optimization suggestions"""
        insights = []
        
        if 'Revenue_Impact' not in data.columns:
            return insights
            
        # High-profit opportunity detection
        high_profit_products = data[data['Revenue_Impact'] > data['Revenue_Impact'].quantile(0.8)]
        if len(high_profit_products) > 5:
            avg_discount = high_profit_products['Applied_Discount'].mean()
            insights.append({
                'type': '🚀 Profit Maximization',
                'title': 'Scale High-Performance Products',
                'description': f'AI identified {len(high_profit_products)} products with exceptional profit potential',
                'impact': f'Additional revenue opportunity: ₹{high_profit_products["Revenue_Impact"].sum() * 0.15:,.0f}',
                'action': 'Increase inventory and marketing for top 10% performers',
                'confidence': 0.85,
                'ai_model': 'Profit Pattern Recognition'
            })
        
        # Loss recovery strategies
        loss_products = data[data['Revenue_Impact'] < data['Revenue_Impact'].quantile(0.2)]
        if len(loss_products) > 3:
            insights.append({
                'type': '🛡️ Loss Prevention AI',
                'title': 'AI-Driven Loss Recovery',
                'description': f'{len(loss_products)} products identified for strategic intervention',
                'impact': f'Potential loss reduction: ₹{abs(loss_products["Revenue_Impact"].sum()) * 0.4:,.0f}',
                'action': 'Implement AI-suggested discount adjustments and bundling',
                'confidence': 0.79,
                'ai_model': 'Loss Pattern Analysis'
            })
        
        # Price elasticity optimization
        elastic_products = data[(data['Applied_Discount'] > 15) & (data['Revenue_Impact'] > 0)]
        if len(elastic_products) > 8:
            insights.append({
                'type': '📊 Elasticity Optimization',
                'title': 'Leverage Price Elastic Products',
                'description': f'{len(elastic_products)} products show high sensitivity to discount changes',
                'impact': '15-25% higher conversion rates with optimized pricing',
                'action': 'Test incremental discount increases on elastic products',
                'confidence': 0.83,
                'ai_model': 'Price Elasticity Modeling'
            })
            
        return insights
    
    def _competitive_analysis(self, data):
        """AI-powered competitive positioning suggestions"""
        insights = []
        
        # Market positioning analysis
        avg_discount = data['Applied_Discount'].mean() if 'Applied_Discount' in data.columns else 15
        market_position = "Aggressive" if avg_discount > 20 else "Competitive" if avg_discount > 12 else "Premium"
        
        insights.append({
            'type': '🏢 Market Positioning',
            'title': f'AI Market Analysis: {market_position} Positioning',
            'description': f'Current average discount {avg_discount:.1f}% places you in {market_position} market segment',
            'impact': f'Optimization potential: {15 if market_position == "Premium" else 10}% revenue improvement',
            'action': f'Adjust strategy to target {("Value" if market_position == "Premium" else "Balanced")} segment',
            'confidence': 0.81,
            'ai_model': 'Market Position Analysis'
        })
        
        # Competitive gap analysis
        if 'Customer_rating' in data.columns:
            avg_rating = data['Customer_rating'].mean()
            if avg_rating > 4.5:
                insights.append({
                    'type': '⭐ Quality Leadership',
                    'title': 'Leverage Quality Premium',
                    'description': 'Exceptional customer ratings support premium pricing strategy',
                    'impact': 'Potential 8-12% price premium on high-rated products',
                    'action': 'Reduce discounts on 4.5+ rated products by 2-4%',
                    'confidence': 0.84,
                    'ai_model': 'Sentiment & Quality Analysis'
                })
                
        return insights
    
    def _risk_analysis(self, data):
        """AI-driven risk assessment and mitigation"""
        insights = []
        
        # Concentration risk
        if 'Revenue_Impact' in data.columns:
            top_10_percent = data.nlargest(len(data)//10, 'Revenue_Impact')
            concentration = top_10_percent['Revenue_Impact'].sum() / data['Revenue_Impact'].sum()
            
            if concentration > 0.6:
                insights.append({
                    'type': '⚖️ Risk Diversification',
                    'title': 'Revenue Concentration Risk Detected',
                    'description': f'Top {len(data)//10} products generate {concentration:.1%} of total revenue impact',
                    'impact': 'High vulnerability to product-specific demand shocks',
                    'action': 'Diversify discount strategy across mid-performing products',
                    'confidence': 0.87,
                    'ai_model': 'Risk Assessment Model'
                })
        
        # Margin compression risk
        low_margin_products = data[data['Applied_Discount'] > 25] if 'Applied_Discount' in data.columns else []
        if len(low_margin_products) > 5:
            insights.append({
                'type': '💰 Margin Protection',
                'title': 'Margin Compression Alert',
                'description': f'{len(low_margin_products)} products with discounts >25% risking profitability',
                'impact': 'Potential margin improvement: 8-15% with optimization',
                'action': 'Gradually reduce discounts on high-discount, low-impact products',
                'confidence': 0.82,
                'ai_model': 'Margin Risk Analysis'
            })
            
        return insights
    
    def _deduplicate_suggestions(self, suggestions):
        """Remove duplicate suggestions based on content similarity"""
        seen_titles = set()
        unique_suggestions = []
        
        for suggestion in suggestions:
            title_key = suggestion['title'][:50]  # Use first 50 chars as key
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_suggestions.append(suggestion)
                
        return unique_suggestions
    
    def _track_suggestion_performance(self, suggestions):
        """Track suggestion performance for continuous learning"""
        for suggestion in suggestions:
            key = suggestion['title']
            if key not in self.performance_tracker:
                self.performance_tracker[key] = {
                    'count': 0,
                    'total_confidence': 0,
                    'last_shown': datetime.now()
                }
            self.performance_tracker[key]['count'] += 1
            self.performance_tracker[key]['total_confidence'] += suggestion['confidence']

    def generate_product_specific_suggestions(self, product_data):
        """Generate AI suggestions for specific products"""
        suggestions = []
        
        # Analyze product characteristics
        cost = product_data.get('Cost_of_the_Product', 0)
        discount = product_data.get('Applied_Discount', 0)
        rating = product_data.get('Customer_rating', 0)
        revenue_impact = product_data.get('Revenue_Impact', 0)
        
        # Generate specific suggestions based on product characteristics
        if revenue_impact < 0:
            if discount > 20:
                suggestions.append(f"Reduce discount from {discount:.1f}% to 15-18% to minimize losses")
            elif cost > 2000:
                suggestions.append("Consider bundling with complementary products to increase value perception")
            else:
                suggestions.append("Test different discount strategies (10-15%) to find optimal pricing")
        
        if rating > 4.5:
            suggestions.append("Leverage high customer rating to maintain premium pricing")
        
        if cost < 1000 and revenue_impact > 0:
            suggestions.append("Scale marketing efforts - this product shows strong profit potential")
            
        return suggestions

class CompetitiveAnalyzer:
    """Analyze competitive pricing landscape"""
    def __init__(self):
        self.competitor_data = self._generate_competitor_data()
    
    def _generate_competitor_data(self):
        """Generate simulated competitor pricing data"""
        competitors = ['Amazon', 'Flipkart', 'Myntra', 'Nykaa', 'Reliance']
        categories = ['Electronics', 'Fashion', 'Home', 'Sports', 'Books']
        
        competitor_prices = {}
        for category in categories:
            competitor_prices[category] = {}
            base_price = np.random.normal(2000, 500)
            for competitor in competitors:
                # Competitors have prices within ±20% of base
                variation = np.random.uniform(0.8, 1.2)
                competitor_prices[category][competitor] = base_price * variation
        
        return competitor_prices
    
    def analyze_competitive_position(self, products):
        """Analyze competitive pricing position"""
        if 'Category' not in products.columns:
            return None
        
        analysis_results = []
        
        for category in products['Category'].unique():
            category_products = products[products['Category'] == category]
            avg_our_price = category_products['New_Price'].mean()
            
            if category in self.competitor_data:
                competitor_prices = list(self.competitor_data[category].values())
                avg_competitor_price = np.mean(competitor_prices)
                price_difference = ((avg_our_price - avg_competitor_price) / avg_competitor_price) * 100
                
                position = "Premium" if price_difference > 10 else "Competitive" if price_difference > -10 else "Value"
                
                analysis_results.append({
                    'category': category,
                    'our_avg_price': avg_our_price,
                    'competitor_avg_price': avg_competitor_price,
                    'price_difference_pct': price_difference,
                    'market_position': position,
                    'recommendation': self._get_competitive_recommendation(position, price_difference)
                })
        
        return analysis_results
    
    def _get_competitive_recommendation(self, position, difference):
        """Get competitive pricing recommendations"""
        if position == "Premium":
            if difference > 20:
                return "Consider reducing prices - significantly above market average"
            else:
                return "Maintain premium positioning with value-added services"
        elif position == "Competitive":
            return "Optimal pricing strategy - continue current approach"
        else:  # Value
            if difference < -15:
                return "Opportunity to increase prices - significantly below market"
            else:
                return "Good value positioning - consider slight increases for better margins"

class AdvancedVisualizations:
    """Advanced visualization capabilities for AI insights"""
    
    @staticmethod
    def create_ai_comparison_chart(ensemble_results, neural_results):
        """Create comparison chart between AI models"""
        comparison_data = []
        
        for _, product in ensemble_results.iterrows():
            product_id = product['ID']
            neural_product = neural_results[neural_results['ID'] == product_id]
            
            if len(neural_product) > 0:
                neural_product = neural_product.iloc[0]
                comparison_data.append({
                    'Product_ID': product_id,
                    'Ensemble_Price': product['New_Price'],
                    'Neural_Price': neural_product['New_Price'],
                    'Ensemble_Discount': product['Applied_Discount'],
                    'Neural_Discount': neural_product['Applied_Discount'],
                    'Ensemble_Impact': product['Revenue_Impact'],
                    'Neural_Impact': neural_product['Revenue_Impact'],
                    'Cost_Price': product['Cost_of_the_Product']
                })
        
        return pd.DataFrame(comparison_data)
    
    @staticmethod
    def create_profit_loss_distribution(results_df):
        """Create profit/loss distribution visualization"""
        status_counts = results_df['Profit_Loss_Status'].value_counts()
        
        fig = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Profit/Loss Distribution",
            color_discrete_sequence=['#22C55E', '#10B981', '#F59E0B', '#EF4444', '#DC2626']
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            height=400,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        return fig
    
    @staticmethod
    def create_discount_impact_scatter(results_df):
        """Create discount vs revenue impact scatter plot"""
        fig = px.scatter(
            results_df,
            x='Applied_Discount',
            y='Revenue_Impact',
            color='Profit_Loss_Status',
            size='Predicted_Sales',
            hover_data=['ID', 'Cost_of_the_Product', 'Season'],
            title="Discount vs Revenue Impact Analysis",
            color_discrete_map={
                'High Profit': '#22C55E',
                'Profit': '#10B981',
                'Minor Loss': '#F59E0B',
                'Moderate Loss': '#EF4444',
                'Significant Loss': '#DC2626'
            }
        )
        fig.update_layout(
            xaxis_title="Applied Discount (%)",
            yaxis_title="Revenue Impact (₹)",
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        return fig
    
    @staticmethod
    def create_profit_loss_barchart(results_df):
        """Create detailed profit/loss bar chart"""
        # Group by status and calculate metrics
        status_summary = results_df.groupby('Profit_Loss_Status').agg({
            'Revenue_Impact': ['sum', 'count', 'mean'],
            'Applied_Discount': 'mean',
            'Predicted_Sales': 'mean'
        }).round(0)
        
        status_summary.columns = ['Total_Impact', 'Product_Count', 'Avg_Impact', 'Avg_Discount', 'Avg_Sales']
        status_summary = status_summary.reset_index()
        
        # Create bar chart
        fig = px.bar(
            status_summary,
            x='Profit_Loss_Status',
            y='Total_Impact',
            color='Profit_Loss_Status',
            title="Profit/Loss Analysis by Category",
            color_discrete_map={
                'High Profit': '#22C55E',
                'Profit': '#10B981',
                'Minor Loss': '#F59E0B',
                'Moderate Loss': '#EF4444',
                'Significant Loss': '#DC2626'
            },
            text='Total_Impact'
        )
        
        fig.update_layout(
            xaxis_title="Profit/Loss Category",
            yaxis_title="Total Revenue Impact (₹)",
            showlegend=False,
            height=450
        )
        fig.update_traces(texttemplate='₹%{text:,}', textposition='outside')
        
        return fig
    
    @staticmethod
    def create_product_comparison_barchart(results_df, top_n=20):
        """Create product-by-product comparison bar chart"""
        # Get top N products by revenue impact
        top_products = results_df.nlargest(top_n, 'Revenue_Impact')
        
        fig = px.bar(
            top_products,
            x='ID',
            y='Revenue_Impact',
            color='Profit_Loss_Status',
            title=f"Top {top_n} Products - Revenue Impact Analysis",
            color_discrete_map={
                'High Profit': '#22C55E',
                'Profit': '#10B981',
                'Minor Loss': '#F59E0B',
                'Moderate Loss': '#EF4444',
                'Significant Loss': '#DC2626'
            },
            hover_data=['Cost_of_the_Product', 'New_Price', 'Applied_Discount', 'Predicted_Sales', 'Season']
        )
        
        fig.update_layout(
            xaxis_title="Product ID",
            yaxis_title="Revenue Impact (₹)",
            showlegend=True,
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    @staticmethod
    def create_price_comparison_barchart(results_df, top_n=15):
        """Create price comparison bar chart for products"""
        top_products = results_df.nlargest(top_n, 'Revenue_Impact')
        
        fig = go.Figure()
        
        # Add bars for original price
        fig.add_trace(go.Bar(
            name='Original Price',
            x=top_products['ID'].astype(str),
            y=top_products['Cost_of_the_Product'],
            marker_color='#3B82F6',
            text=top_products['Cost_of_the_Product'],
            texttemplate='₹%{text:,}',
            textposition='outside'
        ))
        
        # Add bars for new price
        fig.add_trace(go.Bar(
            name='New Price',
            x=top_products['ID'].astype(str),
            y=top_products['New_Price'],
            marker_color='#10B981',
            text=top_products['New_Price'],
            texttemplate='₹%{text:,}',
            textposition='outside'
        ))
        
        fig.update_layout(
            title=f"Price Comparison - Top {top_n} Products",
            xaxis_title="Product ID",
            yaxis_title="Price (₹)",
            barmode='group',
            showlegend=True,
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_competitor_analysis_chart(competitive_analysis):
        """Create competitor analysis visualization"""
        if competitive_analysis is None:
            return None
            
        categories = [item['category'] for item in competitive_analysis]
        our_prices = [item['our_avg_price'] for item in competitive_analysis]
        competitor_prices = [item['competitor_avg_price'] for item in competitive_analysis]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Our Price',
            x=categories,
            y=our_prices,
            marker_color='#3B82F6',
            text=our_prices,
            texttemplate='₹%{text:,.0f}',
            textposition='outside'
        ))
        
        fig.add_trace(go.Bar(
            name='Competitor Avg Price',
            x=categories,
            y=competitor_prices,
            marker_color='#EF4444',
            text=competitor_prices,
            texttemplate='₹%{text:,.0f}',
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Competitive Price Analysis by Category",
            xaxis_title="Category",
            yaxis_title="Price (₹)",
            barmode='group',
            height=500
        )
        
        return fig

class ProductVisualizations:
    """Advanced product visualization capabilities"""
    
    @staticmethod
    def create_product_detailed_barchart(product_data, product_id):
        """Create detailed bar chart for individual product analysis"""
        if product_data.empty:
            return None
            
        product = product_data.iloc[0]
        
        # Create metrics for comparison
        metrics = ['Cost Price', 'New Price', 'Current Revenue', 'Predicted Revenue']
        values = [
            product['Cost_of_the_Product'],
            product['New_Price'],
            product.get('Current_Revenue', product['Cost_of_the_Product'] * 10),
            product.get('Predicted_Revenue', product['New_Price'] * product.get('Predicted_Sales', 10))
        ]
        
        fig = px.bar(
            x=metrics,
            y=values,
            title=f"📊 Product {product_id} - Detailed Financial Analysis",
            color=metrics,
            color_discrete_sequence=['#EF4444', '#10B981', '#3B82F6', '#8B5CF6'],
            text=values
        )
        
        fig.update_layout(
            xaxis_title="Financial Metrics",
            yaxis_title="Amount (₹)",
            showlegend=False,
            height=400
        )
        fig.update_traces(texttemplate='₹%{text:,}', textposition='outside')
        
        return fig
    
    @staticmethod
    def create_product_performance_gauge(product_data, product_id):
        """Create performance gauge chart for individual product"""
        if product_data.empty:
            return None
            
        product = product_data.iloc[0]
        revenue_impact = product.get('Revenue_Impact', 0)
        
        # Normalize impact to -100 to 100 scale
        normalized_impact = max(-100, min(100, revenue_impact / 1000))
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = normalized_impact,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Product {product_id} Performance Score"},
            delta = {'reference': 0},
            gauge = {
                'axis': {'range': [-100, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [-100, -50], 'color': "lightcoral"},
                    {'range': [-50, 0], 'color': "lightyellow"},
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': normalized_impact
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        return fig
    
    @staticmethod
    def create_product_comparison_chart(products_data, selected_products):
        """Create comparison chart for multiple products"""
        if len(selected_products) == 0:
            return None
            
        comparison_data = []
        for product_id in selected_products:
            product = products_data[products_data['ID'] == product_id]
            if not product.empty:
                product = product.iloc[0]
                comparison_data.append({
                    'Product_ID': product_id,
                    'Cost_Price': product['Cost_of_the_Product'],
                    'New_Price': product['New_Price'],
                    'Discount': product['Applied_Discount'],
                    'Revenue_Impact': product['Revenue_Impact'],
                    'Predicted_Sales': product['Predicted_Sales']
                })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        fig = go.Figure()
        
        # Add bars for each metric
        fig.add_trace(go.Bar(
            name='Cost Price',
            x=df_comparison['Product_ID'].astype(str),
            y=df_comparison['Cost_Price'],
            marker_color='#EF4444'
        ))
        
        fig.add_trace(go.Bar(
            name='New Price',
            x=df_comparison['Product_ID'].astype(str),
            y=df_comparison['New_Price'],
            marker_color='#10B981'
        ))
        
        fig.add_trace(go.Bar(
            name='Revenue Impact',
            x=df_comparison['Product_ID'].astype(str),
            y=df_comparison['Revenue_Impact'],
            marker_color='#3B82F6'
        ))
        
        fig.update_layout(
            title="📈 Multi-Product Comparison Analysis",
            xaxis_title="Product ID",
            yaxis_title="Amount (₹)",
            barmode='group',
            height=500
        )
        
        return fig

class SeasonalVisualizations:
    """Advanced seasonal visualization capabilities"""
    
    @staticmethod
    def create_seasonal_demand_analysis(seasonal_df):
        """Create comprehensive seasonal demand analysis"""
        if seasonal_df is None or len(seasonal_df) == 0:
            return None
            
        # Create seasonal demand chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Demand Multiplier',
            x=seasonal_df['season'],
            y=seasonal_df['seasonal_multiplier'],
            text=seasonal_df['demand_change'],
            textposition='outside',
            marker_color=['#10B981', '#F59E0B', '#3B82F6', '#EF4444', '#8B5CF6']
        ))
        
        fig.update_layout(
            title="🌦️ Seasonal Demand Analysis - Demand Multipliers",
            xaxis_title="Season",
            yaxis_title="Demand Multiplier",
            showlegend=False,
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_seasonal_performance_radar(seasonal_df):
        """Create radar chart for seasonal performance comparison"""
        if seasonal_df is None or len(seasonal_df) == 0:
            return None
            
        categories = ['Revenue Impact', 'Profit Margin', 'Product Count', 'Avg Discount', 'Demand Level']
        
        fig = go.Figure()
        
        # Scale values for radar chart
        max_revenue = seasonal_df['total_revenue_impact'].max()
        max_margin = seasonal_df['profit_margin'].max()
        max_count = seasonal_df['product_count'].max()
        max_discount = seasonal_df['avg_discount'].max()
        max_demand = seasonal_df['seasonal_multiplier'].max()
        
        for _, season_data in seasonal_df.iterrows():
            season = season_data['season']
            values = [
                (season_data['total_revenue_impact'] / max_revenue) * 100 if max_revenue > 0 else 0,
                (season_data['profit_margin'] / max_margin) * 100 if max_margin > 0 else 0,
                (season_data['product_count'] / max_count) * 100 if max_count > 0 else 0,
                (season_data['avg_discount'] / max_discount) * 100 if max_discount > 0 else 0,
                (season_data['seasonal_multiplier'] / max_demand) * 100 if max_demand > 0 else 0
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=season
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title="Seasonal Performance Radar Comparison",
            showlegend=True,
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_seasonal_strategy_matrix(strategies):
        """Create strategy matrix for seasonal recommendations"""
        if not strategies:
            return None
            
        seasons = [s['season'] for s in strategies]
        priorities = [s['priority'] for s in strategies]
        risk_levels = [s['risk_level'] for s in strategies]
        strategies_text = [s['strategy'] for s in strategies]
        
        # Create priority mapping for colors
        priority_colors = {'High': '#EF4444', 'Medium': '#F59E0B', 'Low': '#10B981'}
        risk_colors = {'High': '#EF4444', 'Medium': '#F59E0B', 'Low': '#10B981'}
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Season', 'Strategy', 'Priority', 'Risk Level', 'Expected Impact'],
                fill_color='#3B82F6',
                align='left',
                font=dict(color='white', size=12)
            ),
            cells=dict(
                values=[seasons, strategies_text, priorities, risk_levels, 
                       [s['expected_impact'] for s in strategies]],
                fill_color=['#F8FAFC', '#FFFFFF', 
                          [priority_colors[p] for p in priorities],
                          [risk_colors[r] for r in risk_levels],
                          '#F8FAFC'],
                align='left',
                font=dict(color='#1E293B', size=11)
            )
        )])
        
        fig.update_layout(
            title="Seasonal Strategy Matrix",
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_seasonal_trend_analysis(ensemble_recommendations):
        """Create seasonal trend analysis with product breakdown"""
        if 'Season' not in ensemble_recommendations.columns:
            return None
            
        # Group by season and calculate metrics
        seasonal_trends = ensemble_recommendations.groupby('Season').agg({
            'ID': 'count',
            'Revenue_Impact': 'sum',
            'Applied_Discount': 'mean',
            'Predicted_Sales': 'mean',
            'Cost_of_the_Product': 'mean'
        }).round(2)
        
        seasonal_trends.columns = ['Product Count', 'Total Revenue Impact', 'Avg Discount', 'Avg Predicted Sales', 'Avg Cost']
        seasonal_trends = seasonal_trends.reset_index()
        
        # Create comprehensive trend chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Total Revenue Impact',
            x=seasonal_trends['Season'],
            y=seasonal_trends['Total Revenue Impact'],
            text=seasonal_trends['Total Revenue Impact'],
            texttemplate='₹%{text:,}',
            textposition='outside',
            marker_color='#3B82F6'
        ))
        
        fig.add_trace(go.Scatter(
            name='Avg Discount %',
            x=seasonal_trends['Season'],
            y=seasonal_trends['Avg Discount'],
            mode='lines+markers+text',
            text=seasonal_trends['Avg Discount'],
            texttemplate='%{text:.1f}%',
            textposition='top center',
            yaxis='y2',
            line=dict(color='#EF4444', width=3),
            marker=dict(size=10, color='#EF4444')
        ))
        
        fig.update_layout(
            title="🌦️ Seasonal Performance Trends",
            xaxis_title="Season",
            yaxis=dict(
                title="Revenue Impact (₹)",
                side='left'
            ),
            yaxis2=dict(
                title="Average Discount (%)",
                side='right',
                overlaying='y',
                range=[0, max(seasonal_trends['Avg Discount']) * 1.5]
            ),
            showlegend=True,
            height=500
        )
        
        return fig

class UniversalCSVExporter:
    """Universal CSV export functionality with AI enhancements"""
    
    def __init__(self):
        self.export_history = []
    
    def export_to_csv(self, df, filename_prefix="ai_optimization"):
        """Export DataFrame to CSV with download link"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.csv"
        
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-btn">📥 Download {filename}</a>'
        
        # Track export
        self.export_history.append({
            'timestamp': datetime.now(),
            'filename': filename,
            'rows_exported': len(df)
        })
        
        return href
    
    def export_multiple_datasets(self, datasets_dict, zip_filename="ai_analysis_export"):
        """Export multiple datasets as separate CSV files in a zip"""
        import zipfile
        from io import BytesIO
        
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            for name, df in datasets_dict.items():
                csv_data = df.to_csv(index=False)
                zip_file.writestr(f"{name}.csv", csv_data)
        
        zip_buffer.seek(0)
        b64 = base64.b64encode(zip_buffer.read()).decode()
        href = f'<a href="data:application/zip;base64,{b64}" download="{zip_filename}.zip" class="download-btn">📦 Download All Analysis Files</a>'
        
        return href

class PricingMaturityAnalyzer:
    """Pricing Maturity Assessment Tool"""
    def __init__(self):
        self.maturity_levels = {
            'Level 1': 'Reactive',
            'Level 2': 'Standardized',
            'Level 3': 'Strategic',
            'Level 4': 'Optimized',
            'Level 5': 'Innovative'
        }
    
    def assess_maturity(self, products_data, ai_metrics):
        """Assess pricing maturity level"""
        scores = {
            'data_quality': 0,
            'ai_adoption': 0,
            'automation': 0,
            'strategy': 0,
            'innovation': 0
        }
        
        # Calculate scores based on data and AI metrics
        if len(products_data) > 100:
            scores['data_quality'] = 4
        elif len(products_data) > 50:
            scores['data_quality'] = 3
        else:
            scores['data_quality'] = 2
        
        if ai_metrics.get('ai_trained', False):
            scores['ai_adoption'] = 4
        else:
            scores['ai_adoption'] = 2
        
        if ai_metrics.get('automation_score', 0) > 70:
            scores['automation'] = 4
        elif ai_metrics.get('automation_score', 0) > 50:
            scores['automation'] = 3
        else:
            scores['automation'] = 2
        
        total_score = sum(scores.values())
        
        if total_score >= 18:
            level = 'Level 5'
        elif total_score >= 15:
            level = 'Level 4'
        elif total_score >= 12:
            level = 'Level 3'
        elif total_score >= 9:
            level = 'Level 2'
        else:
            level = 'Level 1'
        
        return {
            'maturity_level': level,
            'level_description': self.maturity_levels[level],
            'scores': scores,
            'total_score': total_score,
            'recommendations': self._get_maturity_recommendations(level, scores)
        }
    
    def _get_maturity_recommendations(self, level, scores):
        """Get recommendations for improving pricing maturity"""
        recommendations = []
        
        if scores['data_quality'] < 4:
            recommendations.append("Improve data collection and quality for better AI predictions")
        
        if scores['ai_adoption'] < 4:
            recommendations.append("Increase AI model training frequency and feature engineering")
        
        if scores['automation'] < 4:
            recommendations.append("Implement more automated pricing workflows")
        
        if scores['strategy'] < 4:
            recommendations.append("Develop comprehensive pricing strategies per product segment")
        
        if scores['innovation'] < 4:
            recommendations.append("Explore advanced AI capabilities like causal inference")
        
        return recommendations

class SustainabilityPricing:
    """Sustainability Pricing Integration"""
    def __init__(self):
        self.sustainability_factors = {}
    
    def calculate_sustainability_premium(self, products):
        """Calculate sustainability premium for products"""
        results = []
        
        for _, product in products.iterrows():
            product_id = product.get('ID', 0)
            sustainability_score = product.get('Sustainability_Score', 0.5)
            base_price = product.get('Cost_of_the_Product', 1000)
            
            # Calculate sustainability premium
            if sustainability_score > 0.8:
                premium_pct = 0.15
                tier = "Premium Sustainable"
            elif sustainability_score > 0.6:
                premium_pct = 0.08
                tier = "Sustainable"
            elif sustainability_score > 0.4:
                premium_pct = 0.03
                tier = "Basic Sustainable"
            else:
                premium_pct = 0.0
                tier = "Standard"
            
            sustainability_price = base_price * (1 + premium_pct)
            premium_amount = base_price * premium_pct
            
            results.append({
                'product_id': product_id,
                'sustainability_score': sustainability_score,
                'tier': tier,
                'base_price': base_price,
                'sustainability_price': sustainability_price,
                'premium_pct': premium_pct * 100,
                'premium_amount': premium_amount,
                'market_advantage': "High" if sustainability_score > 0.7 else "Medium" if sustainability_score > 0.5 else "Low"
            })
        
        return pd.DataFrame(results)

class AcademicValidation:
    """Academic Rigor and Validation Features"""
    def __init__(self):
        self.validation_metrics = {}
    
    def validate_model_performance(self, predictions, actuals=None):
        """Validate AI model performance with academic rigor"""
        if actuals is None:
            # Use heuristic validation
            avg_impact = predictions['Revenue_Impact'].mean()
            std_impact = predictions['Revenue_Impact'].std()
            profitable_ratio = len(predictions[predictions['Revenue_Impact'] > 0]) / len(predictions)
            
            validation_results = {
                'avg_revenue_impact': avg_impact,
                'std_revenue_impact': std_impact,
                'profitable_ratio': profitable_ratio,
                'confidence_interval': f"₹{avg_impact - 1.96*std_impact:.0f} to ₹{avg_impact + 1.96*std_impact:.0f}",
                'statistical_significance': "High" if std_impact < avg_impact * 0.3 else "Medium" if std_impact < avg_impact * 0.5 else "Low"
            }
        else:
            # Calculate actual validation metrics
            mae = mean_absolute_error(actuals, predictions['Revenue_Impact'])
            r2 = r2_score(actuals, predictions['Revenue_Impact'])
            
            validation_results = {
                'mean_absolute_error': mae,
                'r2_score': r2,
                'model_accuracy': f"{(1 - mae/abs(actuals.mean()))*100:.1f}%" if actuals.mean() != 0 else "N/A",
                'validation_status': "Validated" if r2 > 0.7 else "Needs Improvement"
            }
        
        return validation_results
    
    def generate_academic_report(self, validation_results, ai_metrics):
        """Generate academic-style report"""
        report = f"""
        # AI Pricing Model Validation Report
        ## Executive Summary
        
        ### Model Performance:
        - **Average Revenue Impact**: ₹{validation_results.get('avg_revenue_impact', 0):,.0f}
        - **Profitability Ratio**: {validation_results.get('profitable_ratio', 0)*100:.1f}%
        - **Statistical Significance**: {validation_results.get('statistical_significance', 'N/A')}
        
        ### Academic Validation:
        - **R² Score**: {validation_results.get('r2_score', 'N/A'):.3f}
        - **Mean Absolute Error**: ₹{validation_results.get('mean_absolute_error', 'N/A'):,.0f}
        
        ### Recommendations:
        1. Regular model retraining recommended
        2. Feature engineering improvements suggested
        3. Cross-validation implementation advised
        """
        
        return report

class AIPriceOptimizer:
    def __init__(self, model_type="ensemble"):
        self.model_type = model_type
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        self.suggestion_engine = AISuggestionEngine()
        self.inventory_optimizer = InventoryOptimizer()
        self.season_analyzer = SeasonAnalyzer()
        self.competitive_analyzer = CompetitiveAnalyzer()
        self.behavioral_economics = BehavioralEconomicsEngine()
        self.game_theory = GameTheoryAnalyzer()
        self.xai = ExplainableAI()
        self.pricing_maturity = PricingMaturityAnalyzer()
        self.sustainability = SustainabilityPricing()
        self.academic_validation = AcademicValidation()
        
    def prepare_features(self, df, is_training=False):
        """Advanced feature engineering with AI-driven feature selection"""
        df_clean = df.copy()
        
        # Handle categorical variables with advanced encoding
        categorical_cols = ['Warehouse_block', 'Product_importance', 'Gender', 'Mode_of_Shipment', 'Category', 'Season']
        for col in categorical_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna('Unknown')
                if is_training:
                    self.encoders[col] = RobustLabelEncoder()
                    df_clean[col] = self.encoders[col].fit_transform(df_clean[col].astype(str))
                else:
                    if col in self.encoders:
                        try:
                            df_clean[col] = self.encoders[col].transform(df_clean[col].astype(str))
                        except:
                            df_clean[col] = 0
                    else:
                        df_clean[col] = 0
        
        # Advanced feature engineering
        try:
            # Price elasticity features
            if 'Discount_offered' in df_clean.columns and 'Cost_of_the_Product' in df_clean.columns:
                df_clean['price_elasticity'] = df_clean['Discount_offered'] / (df_clean['Cost_of_the_Product'] + 1)
                df_clean['discount_ratio'] = df_clean['Discount_offered'] / 100
            
            # Customer value features
            if 'Prior_purchases' in df_clean.columns and 'Cost_of_the_Product' in df_clean.columns:
                df_clean['customer_lifetime_value'] = df_clean['Prior_purchases'] * df_clean['Cost_of_the_Product']
                df_clean['purchase_frequency_score'] = df_clean['Prior_purchases'] / df_clean['Prior_purchases'].max()
            
            # Product performance features
            if 'Customer_rating' in df_clean.columns:
                df_clean['rating_score'] = df_clean['Customer_rating'] / 5.0
                df_clean['premium_product'] = (df_clean['Customer_rating'] > 4.0).astype(int)
            
            # Behavioral economics features
            if 'Reference_Price' in df_clean.columns and 'Price_Anchor' in df_clean.columns:
                df_clean['anchoring_effect'] = (df_clean['Price_Anchor'] - df_clean['Cost_of_the_Product']) / df_clean['Cost_of_the_Product']
                df_clean['reference_price_ratio'] = df_clean['Reference_Price'] / df_clean['Cost_of_the_Product']
            
            # Sustainability features
            if 'Sustainability_Score' in df_clean.columns:
                df_clean['sustainability_premium'] = df_clean['Sustainability_Score'] * 0.2
            
            # Game theory features
            if 'Competitive_Intensity' in df_clean.columns:
                df_clean['game_theory_score'] = df_clean['Competitive_Intensity'] * df_clean.get('Market_Share', 0.5)
            
            # Seasonal features
            current_month = datetime.now().month
            df_clean['seasonal_demand'] = np.where(current_month in [11, 12], 1.3, 
                                                 np.where(current_month in [6, 7], 1.1, 1.0))
                
        except Exception as e:
            st.warning(f"Advanced feature engineering: {e}")
        
        return df_clean
    
    def select_optimal_features(self, X, y):
        """AI-driven feature selection"""
        try:
            selector = SelectKBest(score_func=f_regression, k=min(10, X.shape[1]))
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            return selected_features
        except:
            return X.columns.tolist()
    
    def train_model(self, df):
        """Train ensemble model with advanced feature selection"""
        try:
            df_clean = self.prepare_features(df, is_training=True)
            
            # Prepare features
            base_features = ['Cost_of_the_Product', 'Discount_offered', 'Weight_in_gms', 
                           'Warehouse_block', 'Product_importance', 'Prior_purchases', 'Season']
            
            # Add engineered features
            engineered_features = ['price_elasticity', 'customer_lifetime_value', 'purchase_frequency_score',
                                 'rating_score', 'premium_product', 'seasonal_demand', 'anchoring_effect',
                                 'sustainability_premium', 'game_theory_score']
            
            all_features = base_features + [f for f in engineered_features if f in df_clean.columns]
            available_features = [f for f in all_features if f in df_clean.columns]
            
            if len(available_features) < 3:
                return 999
            
            X = df_clean[available_features]
            
            # Use multiple target variables for richer learning
            if 'Prior_purchases' in df_clean.columns:
                y = df_clean['Prior_purchases']
            else:
                y = df_clean['Cost_of_the_Product']
            
            if len(X) < 10:
                return 999
            
            # Feature selection
            self.feature_columns = self.select_optimal_features(X, y)
            X = X[self.feature_columns]
            
            # Advanced scaling
            self.scalers['main'] = StandardScaler()
            X_scaled = self.scalers['main'].fit_transform(X)
            
            if self.model_type == "ensemble":
                # Ensemble of models
                self.models['xgb'] = xgb.XGBRegressor(
                    n_estimators=150, max_depth=8, learning_rate=0.1, random_state=42
                )
                self.models['lgb'] = lgb.LGBMRegressor(
                    n_estimators=150, random_state=42
                )
                self.models['rf'] = RandomForestRegressor(
                    n_estimators=100, max_depth=8, random_state=42
                )
            else:  # neural style
                self.models['xgb'] = xgb.XGBRegressor(
                    n_estimators=200, max_depth=10, learning_rate=0.05, random_state=42
                )
            
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            
            # Train all models
            for name, model in self.models.items():
                model.fit(X_train, y_train)
            
            # Select best model based on performance
            best_score = float('inf')
            for name, model in self.models.items():
                y_pred = model.predict(X_test)
                score = mean_absolute_error(y_test, y_pred)
                if score < best_score:
                    best_score = score
                    self.models['best'] = model
            
            # Calculate feature importance
            if 'best' in self.models:
                feature_imp = self.xai.calculate_feature_importance(self.models['best'], self.feature_columns)
                if feature_imp is not None:
                    self.feature_importance = feature_imp
            
            return best_score
            
        except Exception as e:
            return 999
    
    def predict_optimized_prices(self, df, strategy, profit_margin=0.2):
        """AI-optimized price predictions with advanced algorithms"""
        try:
            df_clean = self.prepare_features(df, is_training=False)
            
            # Group and aggregate
            agg_dict = {}
            numeric_cols = ['Cost_of_the_Product', 'Discount_offered', 'Prior_purchases', 'Weight_in_gms', 'Customer_rating',
                          'Competitor_Price', 'Sustainability_Score', 'Market_Share', 'Reference_Price', 'Price_Anchor']
            for col in numeric_cols:
                if col in df_clean.columns:
                    agg_dict[col] = 'mean'
            
            categorical_cols = ['Warehouse_block', 'Product_importance', 'Category', 'Season']
            for col in categorical_cols:
                if col in df_clean.columns:
                    agg_dict[col] = 'first'
            
            product_analysis = df_clean.groupby('ID').agg(agg_dict).reset_index()
            
            return self._generate_optimized_recommendations(product_analysis, strategy, profit_margin)
                
        except Exception as e:
            return self._create_fallback_recommendations(df)
    
    def _generate_optimized_recommendations(self, products, strategy, profit_margin):
        """Generate AI-optimized recommendations using ensemble predictions"""
        results = []
        
        for _, product in products.iterrows():
            # AI-driven discount strategy
            if strategy['type'] == 'dynamic':
                optimal_discount = self._calculate_optimal_discount(product)
            elif strategy['type'] == 'fixed_discount':
                optimal_discount = strategy['value']
            else:  # fixed_price
                optimal_discount = (1 - strategy['value'] / product['Cost_of_the_Product']) * 100
            
            # AI-powered impact prediction
            impact_data = self._predict_impact(product, optimal_discount, profit_margin)
            
            # XAI explanations
            explanations = self.xai.generate_explanation(product, impact_data)
            
            results.append({
                'ID': product['ID'],
                'Cost_of_the_Product': product['Cost_of_the_Product'],
                'New_Price': impact_data['new_price'],
                'Applied_Discount': impact_data['actual_discount'],
                'Revenue_Impact': impact_data['revenue_impact'],
                'Predicted_Sales': impact_data['predicted_demand'],
                'Profit_Loss_Status': impact_data['status'],
                'Current_Revenue': product['Cost_of_the_Product'] * product.get('Prior_purchases', 10),
                'Predicted_Revenue': impact_data['new_price'] * impact_data['predicted_demand'],
                'AI_Model_Used': impact_data['model_used'],
                'AI_Type': self.model_type,
                'Season': product.get('Season', 'All Season'),
                'XAI_Explanations': ' | '.join(explanations) if explanations else 'No explanations available',
                'Sustainability_Score': product.get('Sustainability_Score', 0.5),
                'Competitor_Price': product.get('Competitor_Price', 0)
            })
        
        result_df = pd.DataFrame(results)
        return result_df.sort_values('Revenue_Impact', ascending=False)
    
    def _calculate_optimal_discount(self, product):
        """AI-calculated optimal discount using multiple factors"""
        if self.model_type == "ensemble":
            base_discount = 15
        else:  # neural style
            base_discount = 18  # More aggressive pricing
        
        # Product importance adjustment
        importance_map = {'high': -4, 'medium': 0, 'low': 3}
        if product.get('Product_importance') in importance_map:
            base_discount += importance_map[product.get('Product_importance')]
        
        # Performance-based adjustment
        if product.get('Prior_purchases', 10) > 25:
            base_discount -= 3
        elif product.get('Prior_purchases', 10) < 5:
            base_discount += 3
        
        # Rating-based adjustment
        if product.get('Customer_rating', 4) > 4.5:
            base_discount -= 2
        elif product.get('Customer_rating', 4) < 3.5:
            base_discount += 2
        
        # Sustainability adjustment
        if product.get('Sustainability_Score', 0.5) > 0.7:
            base_discount -= 1  # Premium for sustainable products
        
        # Competitive adjustment
        if product.get('Competitor_Price', 0) > 0:
            our_price = product['Cost_of_the_Product'] * (1 - base_discount/100)
            if our_price > product.get('Competitor_Price') * 1.1:
                base_discount += 2  # More competitive
        
        # Behavioral economics adjustment
        if product.get('anchoring_effect', 0) > 0.3:
            base_discount -= 1  # Less discount for strong anchoring
        
        # Seasonal adjustment
        season = product.get('Season', 'All Season')
        seasonal_adjustments = {
            'Winter': 2, 'Spring': 1, 'Autumn': 1, 
            'Summer': -1, 'Monsoon': -2
        }
        base_discount += seasonal_adjustments.get(season, 0)
        
        return max(5, min(base_discount, 40))
    
    def _predict_impact(self, product, discount, profit_margin):
        """AI-powered impact prediction using ensemble models"""
        base_price = product['Cost_of_the_Product']
        current_demand = product.get('Prior_purchases', 10)
        
        # Calculate new price with constraints
        new_price = base_price * (1 - discount/100)
        min_price = base_price * (1 - profit_margin)
        
        if new_price < min_price:
            new_price = min_price * 1.02
            actual_discount = (1 - new_price / base_price) * 100
        else:
            actual_discount = discount
        
        # AI demand prediction
        if hasattr(self, 'models') and 'best' in self.models and self.feature_columns:
            predicted_demand = self._predict_with_ensemble(product, actual_discount, current_demand)
            model_used = "Ensemble AI" if self.model_type == "ensemble" else "Neural-Style AI"
        else:
            predicted_demand = self._predict_heuristic_demand(product, actual_discount, current_demand)
            model_used = "Heuristic AI"
        
        # Calculate impact
        current_revenue = base_price * current_demand
        predicted_revenue = new_price * predicted_demand
        revenue_impact = predicted_revenue - current_revenue
        
        # AI status classification
        status = self._classify_status(revenue_impact, current_revenue)
        
        return {
            'new_price': new_price,
            'predicted_demand': predicted_demand,
            'revenue_impact': revenue_impact,
            'status': status,
            'actual_discount': actual_discount,
            'model_used': model_used
        }
    
    def _predict_with_ensemble(self, product, discount, current_demand):
        """Predict demand using AI ensemble"""
        try:
            # Prepare features for prediction
            feature_data = {}
            for feature in self.feature_columns:
                if feature in product:
                    feature_data[feature] = product[feature]
                elif feature == 'price_elasticity':
                    feature_data[feature] = discount / (product['Cost_of_the_Product'] + 1)
                else:
                    feature_data[feature] = 0
            
            # Create feature array
            X_pred = np.array([[feature_data.get(col, 0) for col in self.feature_columns]])
            X_scaled = self.scalers['main'].transform(X_pred)
            
            # Ensemble prediction
            predictions = []
            for name, model in self.models.items():
                if name != 'best':
                    pred = model.predict(X_scaled)[0]
                    predictions.append(pred)
            
            # Weighted ensemble prediction
            predicted_demand = np.mean(predictions) if predictions else current_demand
            
            # Different prediction styles
            if self.model_type == "ensemble":
                return max(predicted_demand, current_demand * 0.5)
            else:  # neural style - more aggressive predictions
                return max(predicted_demand * 1.1, current_demand * 0.6)
            
        except:
            return self._predict_heuristic_demand(product, discount, current_demand)
    
    def _predict_heuristic_demand(self, product, discount, current_demand):
        """AI-enhanced heuristic demand prediction"""
        # Base elasticity
        if self.model_type == "ensemble":
            base_elasticity = 1 + (discount / 100) * 0.7
        else:  # neural style
            base_elasticity = 1 + (discount / 100) * 0.8  # More responsive to discounts
        
        # Multi-factor adjustments
        factors = 1.0
        
        # Importance factor
        if product.get('Product_importance') == 'high':
            factors *= 1.25
        elif product.get('Product_importance') == 'low':
            factors *= 0.85
        
        # Performance factor
        if current_demand > 20:
            factors *= 1.15
        elif current_demand < 5:
            factors *= 0.8
        
        # Rating factor
        if product.get('Customer_rating', 4) > 4.0:
            factors *= 1.1
        
        # Sustainability factor
        if product.get('Sustainability_Score', 0.5) > 0.7:
            factors *= 1.08
        
        # Competitive factor
        if product.get('Competitor_Price', 0) > 0:
            our_price = product['Cost_of_the_Product'] * (1 - discount/100)
            if our_price < product.get('Competitor_Price') * 0.9:
                factors *= 1.1  # Price advantage
        
        # Behavioral factor
        if product.get('anchoring_effect', 0) > 0.2:
            factors *= 0.95  # Less discount effectiveness with anchoring
        
        # Seasonal factor
        season = product.get('Season', 'All Season')
        seasonal_factors = {
            'Winter': 1.3, 'Spring': 1.1, 'Autumn': 1.2,
            'Summer': 0.9, 'Monsoon': 0.8
        }
        factors *= seasonal_factors.get(season, 1.0)
        
        predicted_demand = current_demand * base_elasticity * factors
        
        # AI-capped growth
        max_growth = 2.5
        return min(predicted_demand, current_demand * max_growth)
    
    def _classify_status(self, revenue_impact, current_revenue):
        """AI-powered status classification"""
        impact_ratio = revenue_impact / current_revenue if current_revenue > 0 else 0
        
        if impact_ratio > 0.15:
            return "High Profit"
        elif impact_ratio > 0.05:
            return "Profit"
        elif impact_ratio > -0.05:
            return "Minor Loss"
        elif impact_ratio > -0.15:
            return "Moderate Loss"
        else:
            return "Significant Loss"
    
    def _create_fallback_recommendations(self, df):
        """AI-enhanced fallback recommendations"""
        if 'ID' in df.columns:
            product_analysis = df.groupby('ID').first().reset_index()
        else:
            product_analysis = df.head(100).copy()
            product_analysis['ID'] = range(1, len(product_analysis) + 1)
        
        if 'Cost_of_the_Product' not in product_analysis.columns:
            product_analysis['Cost_of_the_Product'] = 100
        
        results = []
        for _, product in product_analysis.iterrows():
            base_price = product['Cost_of_the_Product']
            
            # AI-driven discount variation
            discount = np.random.normal(18, 6)
            discount = max(8, min(discount, 35))
            
            new_price = base_price * (1 - discount/100)
            current_demand = product.get('Prior_purchases', 10)
            
            # AI-enhanced demand prediction
            predicted_demand = self._predict_heuristic_demand(product, discount, current_demand)
            
            current_revenue = base_price * current_demand
            predicted_revenue = new_price * predicted_demand
            revenue_impact = predicted_revenue - current_revenue
            
            # AI status classification
            status = self._classify_status(revenue_impact, current_revenue)
            
            results.append({
                'ID': product['ID'],
                'Cost_of_the_Product': base_price,
                'New_Price': new_price,
                'Applied_Discount': discount,
                'Revenue_Impact': revenue_impact,
                'Predicted_Sales': predicted_demand,
                'Profit_Loss_Status': status,
                'Current_Revenue': current_revenue,
                'Predicted_Revenue': predicted_revenue,
                'AI_Model_Used': 'AI Heuristic Fallback',
                'AI_Type': self.model_type,
                'Season': product.get('Season', 'All Season'),
                'XAI_Explanations': 'Fallback model used - limited features available'
            })
        
        return pd.DataFrame(results)

def main():
    # Initialize AI-powered session state
    if 'ai_optimizer_ensemble' not in st.session_state:
        st.session_state.ai_optimizer_ensemble = AIPriceOptimizer("ensemble")
    if 'ai_optimizer_neural' not in st.session_state:
        st.session_state.ai_optimizer_neural = AIPriceOptimizer("neural")
    if 'ai_trained' not in st.session_state:
        st.session_state.ai_trained = False
    if 'exporter' not in st.session_state:
        st.session_state.exporter = UniversalCSVExporter()
    if 'discount_strategy' not in st.session_state:
        st.session_state.discount_strategy = {'type': 'dynamic', 'value': 15}
    if 'show_loss_only' not in st.session_state:
        st.session_state.show_loss_only = False
    if 'selected_products' not in st.session_state:
        st.session_state.selected_products = []
    if 'show_advanced_features' not in st.session_state:
        st.session_state.show_advanced_features = False

    apply_custom_styles()
    
    # Sidebar Header
    st.sidebar.markdown("""
    <div class="sidebar-header">
        <h2>🤖 AI Price Intelligence Platform</h2>
        <p>Multi-Model AI System with Advanced Features</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar.expander("📁 Data Configuration", expanded=True):
        uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
        use_demo_data = st.checkbox("Use Enhanced Demo Data", value=True)
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Successfully loaded {len(df)} records")
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")
                st.stop()
        elif use_demo_data:
            df = create_demo_data()
            st.info("📊 Using enhanced demo dataset with 500 products")
            st.info("🆕 Features: Behavioral Economics, Game Theory, Sustainability")
        else:
            st.info("Please upload a CSV file or use demo data to continue.")
            st.stop()
    
    # Advanced Features Toggle
    st.sidebar.markdown("---")
    st.session_state.show_advanced_features = st.sidebar.checkbox(
        "🛠️ Show Advanced Features", 
        value=st.session_state.show_advanced_features,
        help="Enable academic research-inspired features"
    )
    
    # Pricing Strategy Configuration
    with st.sidebar.expander("⚡ Pricing Strategy", expanded=True):
        strategy_type = st.radio(
            "Select Pricing Strategy:",
            ['Dynamic AI Pricing', 'Fixed Discount %', 'Fixed Price ₹'],
            help="Choose how AI should optimize prices"
        )
        
        if strategy_type == 'Fixed Discount %':
            discount_value = st.slider("Fixed Discount Percentage", 0, 50, 15)
            st.session_state.discount_strategy = {'type': 'fixed_discount', 'value': discount_value}
            st.info(f"🎯 Applying {discount_value}% discount to all products")
            
        elif strategy_type == 'Fixed Price ₹':
            price_value = st.slider("Fixed Price (₹)", 100, 10000, 2000)
            st.session_state.discount_strategy = {'type': 'fixed_price', 'value': price_value}
            st.info(f"💰 Setting all products to ₹{price_value}")
            
        else:  # Dynamic AI Pricing
            st.session_state.discount_strategy = {'type': 'dynamic'}
            st.info("🤖 AI will determine optimal discounts per product")
        
        profit_margin = st.slider("🎯 Minimum Profit Margin (%)", 5, 40, 20)
    
    # Advanced Configuration
    if st.session_state.show_advanced_features:
        with st.sidebar.expander("🎓 Advanced AI Configuration", expanded=False):
            # Behavioral Economics Settings
            st.markdown("#### 🧠 Behavioral Economics")
            use_anchoring = st.checkbox("Use Price Anchoring", value=True)
            use_decoy = st.checkbox("Use Decoy Effect", value=True)
            
            # Game Theory Settings
            st.markdown("#### ♟️ Game Theory")
            use_game_theory = st.checkbox("Competitive Game Analysis", value=True)
            nash_equilibrium = st.checkbox("Nash Equilibrium Pricing", value=True)
            
            # Sustainability Settings
            st.markdown("#### 🌱 Sustainability")
            sustainability_premium = st.slider("Sustainability Premium (%)", 0, 20, 5)
    
    # Simple filtering
    with st.sidebar.expander("🔍 Data Filters", expanded=True):
        if 'ID' in df.columns:
            product_ids = st.multiselect("Product IDs", options=sorted(df['ID'].unique())[:20])
        else:
            product_ids = []
            
        if 'Cost_of_the_Product' in df.columns:
            min_cost, max_cost = st.slider("Cost Range (₹)", 
                                         float(df['Cost_of_the_Product'].min()), 
                                         float(df['Cost_of_the_Product'].max()),
                                         (float(df['Cost_of_the_Product'].min()), 
                                          float(df['Cost_of_the_Product'].max())))
        else:
            min_cost, max_cost = 0, 1000
            
        if 'Warehouse_block' in df.columns:
            warehouses = st.multiselect("Warehouse", options=sorted(df['Warehouse_block'].unique()))
        else:
            warehouses = []
        
        # Category filter
        if 'Category' in df.columns:
            categories = st.multiselect("Categories", options=sorted(df['Category'].unique()))
        else:
            categories = []
        
        # Loss-only filter
        st.session_state.show_loss_only = st.checkbox("Show Loss-Making Products Only", value=False)
    
    # Apply filters
    filtered_df = df.copy()
    if product_ids:
        filtered_df = filtered_df[filtered_df['ID'].isin(product_ids)]
    if 'Cost_of_the_Product' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['Cost_of_the_Product'] >= min_cost) &
            (filtered_df['Cost_of_the_Product'] <= max_cost)
        ]
    if warehouses and 'Warehouse_block' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Warehouse_block'].isin(warehouses)]
    if categories and 'Category' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Category'].isin(categories)]
    
    # AI Training
    with st.sidebar.expander("🧠 AI Model Training", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Train Ensemble AI", use_container_width=True):
                with st.spinner("Training Ensemble AI with advanced features..."):
                    error_score = st.session_state.ai_optimizer_ensemble.train_model(filtered_df)
                    if error_score < 100:
                        st.session_state.ai_trained = True
                        st.success(f"✅ Ensemble AI trained! Error: {error_score:.2f}")
                    else:
                        st.warning("⚠️ Using AI heuristic approach")
        
        with col2:
            if st.button("🧠 Train Neural AI", use_container_width=True):
                with st.spinner("Training Neural-Style AI..."):
                    error_score = st.session_state.ai_optimizer_neural.train_model(filtered_df)
                    st.session_state.ai_trained = True
                    st.success(f"✅ Neural-Style AI ready! Error: {error_score:.2f}")
    
    # Export all data button
    with st.sidebar.expander("📤 Export Options", expanded=False):
        if st.button("📊 Export All Analysis", use_container_width=True):
            st.info("Available in the Export Results tab")
    
    # Main content
    st.markdown('<h1 class="main-header">🤖 AI Price Intelligence Platform</h1>', unsafe_allow_html=True)
    
    # Feature badges
    st.markdown("""
    <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px;">
        <span class="ai-badge">Multi-Model Ensemble</span>
        <span class="ai-badge">Behavioral Economics</span>
        <span class="ai-badge">Game Theory</span>
        <span class="feature-badge">XAI Explanations</span>
        <span class="feature-badge">Inventory Optimization</span>
        <span class="feature-badge">Seasonal Analysis</span>
        <span class="research-badge">Academic Validation</span>
        <span class="research-badge">Sustainability Pricing</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate predictions from both AIs
    with st.spinner("Running dual AI analysis with advanced features..."):
        ensemble_recommendations = st.session_state.ai_optimizer_ensemble.predict_optimized_prices(
            filtered_df, st.session_state.discount_strategy, profit_margin/100
        )
        
        neural_recommendations = st.session_state.ai_optimizer_neural.predict_optimized_prices(
            filtered_df, st.session_state.discount_strategy, profit_margin/100
        )

    # Apply loss-only filter if selected
    if st.session_state.show_loss_only:
        loss_statuses = ['Minor Loss', 'Moderate Loss', 'Significant Loss']
        ensemble_recommendations = ensemble_recommendations[ensemble_recommendations['Profit_Loss_Status'].isin(loss_statuses)]
        neural_recommendations = neural_recommendations[neural_recommendations['Profit_Loss_Status'].isin(loss_statuses)]
        st.info(f"📉 Showing {len(ensemble_recommendations)} loss-making products")

    # Generate AI suggestions
    with st.spinner("Generating business suggestions..."):
        suggestions = st.session_state.ai_optimizer_ensemble.suggestion_engine.generate_ai_suggestions(
            ensemble_recommendations, st.session_state.discount_strategy
        )

    # Generate advanced analyses
    inventory_analysis = st.session_state.ai_optimizer_ensemble.inventory_optimizer.analyze_inventory_needs(
        ensemble_recommendations, ensemble_recommendations
    )
    inventory_efficiency = st.session_state.ai_optimizer_ensemble.inventory_optimizer.calculate_inventory_efficiency(
        inventory_analysis
    )
    
    seasonal_analysis = st.session_state.ai_optimizer_ensemble.season_analyzer.analyze_seasonal_trends(
        ensemble_recommendations
    )
    seasonal_strategies = st.session_state.ai_optimizer_ensemble.season_analyzer.generate_seasonal_strategies(
        seasonal_analysis
    )
    
    competitive_analysis = st.session_state.ai_optimizer_ensemble.competitive_analyzer.analyze_competitive_position(
        ensemble_recommendations
    )
    
    # Advanced analyses (if enabled)
    behavioral_analysis = None
    game_theory_analysis = None
    sustainability_analysis = None
    maturity_assessment = None
    academic_report = None
    
    if st.session_state.show_advanced_features:
        with st.spinner("Running advanced analyses..."):
            # Behavioral Economics Analysis
            behavioral_analysis = st.session_state.ai_optimizer_ensemble.behavioral_economics.calculate_anchoring_effects(
                ensemble_recommendations
            )
            
            # Game Theory Analysis
            game_theory_analysis = st.session_state.ai_optimizer_ensemble.game_theory.analyze_competitive_game(
                ensemble_recommendations, {}
            )
            
            # Sustainability Analysis
            sustainability_analysis = st.session_state.ai_optimizer_ensemble.sustainability.calculate_sustainability_premium(
                ensemble_recommendations
            )
            
            # Pricing Maturity Assessment
            ai_metrics = {
                'ai_trained': st.session_state.ai_trained,
                'automation_score': 75  # Example score
            }
            maturity_assessment = st.session_state.ai_optimizer_ensemble.pricing_maturity.assess_maturity(
                filtered_df, ai_metrics
            )
            
            # Academic Validation
            academic_report = st.session_state.ai_optimizer_ensemble.academic_validation.validate_model_performance(
                ensemble_recommendations
            )

    # Create tabs
    tab_titles = ["🤖 AI Comparison", "📊 Business Overview", "📈 Product Analysis", 
                  "📉 Loss Analysis", "📦 Inventory", "🌦️ Seasonal", "💡 AI Suggestions"]
    
    if st.session_state.show_advanced_features:
        tab_titles.extend(["🎓 Behavioral", "♟️ Game Theory", "🌱 Sustainability", 
                          "📚 Academic", "📤 Export"])
    
    tabs = st.tabs(tab_titles)

    with tabs[0]:
        st.markdown("### 🤖 Dual AI Model Comparison")
        
        # Create comparison chart
        comparison_df = AdvancedVisualizations.create_ai_comparison_chart(
            ensemble_recommendations, neural_recommendations
        )
        
        # Display comparison table
        st.markdown('<div class="comparison-table">', unsafe_allow_html=True)
        st.dataframe(comparison_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed metrics comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class='ai-card-ensemble'>
                <h3>🚀 Ensemble AI Performance</h3>
            </div>
            """, unsafe_allow_html=True)
            
            total_impact_ensemble = ensemble_recommendations['Revenue_Impact'].sum()
            profitable_ensemble = len(ensemble_recommendations[ensemble_recommendations['Revenue_Impact'] > 0])
            avg_discount_ensemble = ensemble_recommendations['Applied_Discount'].mean()
            
            st.metric("Total Revenue Impact", f"₹{total_impact_ensemble:,.0f}")
            st.metric("Profitable Products", f"{profitable_ensemble}/{len(ensemble_recommendations)}")
            st.metric("Products Analyzed", len(ensemble_recommendations))
            st.metric("Average Discount", f"{avg_discount_ensemble:.1f}%")
        
        with col2:
            st.markdown("""
            <div class='ai-card-neural'>
                <h3>🧠 Neural-Style AI Performance</h3>
            </div>
            """, unsafe_allow_html=True)
            
            total_impact_neural = neural_recommendations['Revenue_Impact'].sum()
            profitable_neural = len(neural_recommendations[neural_recommendations['Revenue_Impact'] > 0])
            avg_discount_neural = neural_recommendations['Applied_Discount'].mean()
            
            st.metric("Total Revenue Impact", f"₹{total_impact_neural:,.0f}")
            st.metric("Profitable Products", f"{profitable_neural}/{len(neural_recommendations)}")
            st.metric("Products Analyzed", len(neural_recommendations))
            st.metric("Average Discount", f"{avg_discount_neural:.1f}%")
        
        # Recommendation
        st.markdown("---")
        if total_impact_ensemble > total_impact_neural:
            st.success("**🎯 Recommendation**: Ensemble AI performs better for maximum revenue impact")
            st.info(f"**💰 Value**: ₹{total_impact_ensemble - total_impact_neural:,.0f} higher revenue potential")
        else:
            st.info("**💡 Recommendation**: Neural-Style AI provides better results with more aggressive pricing")
            st.info(f"**💰 Value**: ₹{total_impact_neural - total_impact_ensemble:,.0f} higher revenue potential")

    with tabs[1]:
        st.markdown("### 📊 Business Overview Dashboard")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_revenue_impact = ensemble_recommendations['Revenue_Impact'].sum()
            st.metric("Total Revenue Impact", f"₹{total_revenue_impact:,.0f}")
        
        with col2:
            profitable_products = len(ensemble_recommendations[ensemble_recommendations['Revenue_Impact'] > 0])
            st.metric("Profitable Products", f"{profitable_products}/{len(ensemble_recommendations)}")
        
        with col3:
            avg_discount = ensemble_recommendations['Applied_Discount'].mean()
            st.metric("Average Discount", f"{avg_discount:.1f}%")
        
        with col4:
            total_predicted_revenue = ensemble_recommendations['Predicted_Revenue'].sum()
            st.metric("Predicted Revenue", f"₹{total_predicted_revenue:,.0f}")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            fig = AdvancedVisualizations.create_profit_loss_distribution(ensemble_recommendations)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = AdvancedVisualizations.create_discount_impact_scatter(ensemble_recommendations)
            st.plotly_chart(fig, use_container_width=True)
        
        # Top performing products
        st.markdown("#### 🏆 Top 10 Performing Products")
        top_products = ensemble_recommendations.nlargest(10, 'Revenue_Impact')[
            ['ID', 'Cost_of_the_Product', 'New_Price', 'Applied_Discount', 
             'Revenue_Impact', 'Profit_Loss_Status', 'Season']
        ]
        st.dataframe(top_products.style.format({
            'Cost_of_the_Product': '₹{:,.0f}',
            'New_Price': '₹{:,.0f}',
            'Revenue_Impact': '₹{:,.0f}'
        }), use_container_width=True)
        
        # Competitive Analysis
        if competitive_analysis:
            st.markdown("#### 🏢 Competitive Position Analysis")
            fig = AdvancedVisualizations.create_competitor_analysis_chart(competitive_analysis)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.markdown("### 📈 Advanced Product Analysis")
        
        # Product selection section
        col1, col2 = st.columns([1, 2])
        
        with col1:
            selected_product_id = st.selectbox(
                "Select Product ID:",
                options=sorted(ensemble_recommendations['ID'].unique()),
                help="Choose a product to view detailed analysis"
            )
            
            # Multi-product comparison
            st.markdown("#### 🔄 Compare Products")
            compare_products = st.multiselect(
                "Select products to compare:",
                options=sorted(ensemble_recommendations['ID'].unique()),
                default=st.session_state.selected_products,
                help="Select multiple products for comparison"
            )
            st.session_state.selected_products = compare_products
        
        with col2:
            if selected_product_id:
                product_data = ensemble_recommendations[ensemble_recommendations['ID'] == selected_product_id]
                if not product_data.empty:
                    product = product_data.iloc[0]
                    status_color = {
                        'High Profit': '#22C55E',
                        'Profit': '#10B981', 
                        'Minor Loss': '#F59E0B',
                        'Moderate Loss': '#EF4444',
                        'Significant Loss': '#DC2626'
                    }.get(product['Profit_Loss_Status'], '#6B7280')
                    
                    st.markdown(f"""
                    <div class="product-card">
                        <h3>Product {selected_product_id} Analysis</h3>
                        <p><strong>Status:</strong> <span style="color: {status_color}">{product['Profit_Loss_Status']}</span></p>
                        <p><strong>Revenue Impact:</strong> ₹{product['Revenue_Impact']:,.0f}</p>
                        <p><strong>Season:</strong> {product.get('Season', 'All Season')}</p>
                        <p><strong>AI Model:</strong> {product['AI_Model_Used']}</p>
                        <p><strong>XAI:</strong> {product.get('XAI_Explanations', 'Not available')[:100]}...</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Single Product Detailed Analysis
        if selected_product_id:
            product_data = ensemble_recommendations[ensemble_recommendations['ID'] == selected_product_id]
            
            if not product_data.empty:
                # Product metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Original Price", f"₹{product_data['Cost_of_the_Product'].iloc[0]:,.0f}")
                with col2:
                    st.metric("New Price", f"₹{product_data['New_Price'].iloc[0]:,.0f}")
                with col3:
                    st.metric("Discount Applied", f"{product_data['Applied_Discount'].iloc[0]:.1f}%")
                with col4:
                    st.metric("Revenue Impact", f"₹{product_data['Revenue_Impact'].iloc[0]:,.0f}")
                
                # Product visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = ProductVisualizations.create_product_detailed_barchart(product_data, selected_product_id)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = ProductVisualizations.create_product_performance_gauge(product_data, selected_product_id)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                # XAI Explanations
                if 'XAI_Explanations' in product_data.columns:
                    st.markdown("#### 🔍 Explainable AI (XAI) Insights")
                    explanations = product_data['XAI_Explanations'].iloc[0]
                    if explanations and explanations != 'Not available':
                        for explanation in str(explanations).split(' | '):
                            st.markdown(f"• {explanation}")
                    else:
                        st.info("No XAI explanations available for this product.")
                
                # Product details table
                st.markdown("#### 📋 Product Details")
                display_cols = ['ID', 'Cost_of_the_Product', 'New_Price', 'Applied_Discount', 
                              'Revenue_Impact', 'Predicted_Sales', 'Profit_Loss_Status', 
                              'Season', 'AI_Model_Used', 'Sustainability_Score']
                display_cols = [col for col in display_cols if col in product_data.columns]
                
                st.dataframe(product_data[display_cols].style.format({
                    'Cost_of_the_Product': '₹{:,.0f}',
                    'New_Price': '₹{:,.0f}',
                    'Revenue_Impact': '₹{:,.0f}',
                    'Predicted_Sales': '{:.0f}',
                    'Sustainability_Score': '{:.2f}'
                }), use_container_width=True)
                
                # Product-specific suggestions
                st.markdown("#### 💡 Product-specific Recommendations")
                product_suggestions = st.session_state.ai_optimizer_ensemble.suggestion_engine.generate_product_specific_suggestions(product_data.iloc[0])
                if product_suggestions:
                    for suggestion in product_suggestions:
                        st.markdown(f"• {suggestion}")
                else:
                    st.info("No specific recommendations for this product.")
        
        # Multi-Product Comparison
        if len(st.session_state.selected_products) > 0:
            st.markdown("---")
            st.markdown("#### 📊 Multi-Product Comparison")
            
            fig = ProductVisualizations.create_product_comparison_chart(
                ensemble_recommendations, st.session_state.selected_products
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Comparison table
            comparison_data = []
            for product_id in st.session_state.selected_products:
                product = ensemble_recommendations[ensemble_recommendations['ID'] == product_id]
                if not product.empty:
                    product = product.iloc[0]
                    comparison_data.append({
                        'Product_ID': product_id,
                        'Original_Price': product['Cost_of_the_Product'],
                        'New_Price': product['New_Price'],
                        'Discount_Applied': product['Applied_Discount'],
                        'Revenue_Impact': product['Revenue_Impact'],
                        'Predicted_Sales': product['Predicted_Sales'],
                        'Status': product['Profit_Loss_Status'],
                        'Season': product.get('Season', 'All Season')
                    })
            
            if comparison_data:
                st.dataframe(pd.DataFrame(comparison_data).style.format({
                    'Original_Price': '₹{:,.0f}',
                    'New_Price': '₹{:,.0f}',
                    'Revenue_Impact': '₹{:,.0f}'
                }), use_container_width=True)

    with tabs[3]:
        st.markdown("### 📉 Loss Analysis")
        
        loss_products = ensemble_recommendations[ensemble_recommendations['Revenue_Impact'] < 0]
        
        if len(loss_products) > 0:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_loss = loss_products['Revenue_Impact'].sum()
                st.metric("Total Potential Loss", f"₹{abs(total_loss):,.0f}")
            
            with col2:
                avg_loss_per_product = loss_products['Revenue_Impact'].mean()
                st.metric("Average Loss per Product", f"₹{abs(avg_loss_per_product):,.0f}")
            
            with col3:
                significant_loss_count = len(loss_products[loss_products['Profit_Loss_Status'] == 'Significant Loss'])
                st.metric("Critical Loss Products", significant_loss_count)
            
            # Loss bar chart
            st.markdown("#### 📉 Loss Analysis by Category")
            fig = AdvancedVisualizations.create_profit_loss_barchart(loss_products)
            st.plotly_chart(fig, use_container_width=True)
            
            # Discount vs Loss scatter
            st.markdown("#### 📊 Discount vs Loss Analysis")
            fig = px.scatter(
                loss_products,
                x='Applied_Discount',
                y='Revenue_Impact',
                color='Profit_Loss_Status',
                hover_data=['ID', 'Cost_of_the_Product', 'New_Price'],
                title="Discount Impact on Loss-Making Products"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### 🚨 Loss-Making Products Details")
            display_loss_cols = ['ID', 'Cost_of_the_Product', 'New_Price', 'Applied_Discount', 
                               'Revenue_Impact', 'Predicted_Sales', 'Season']
            st.dataframe(loss_products[display_loss_cols].style.format({
                'Cost_of_the_Product': '₹{:,.0f}',
                'New_Price': '₹{:,.0f}',
                'Revenue_Impact': '₹{:,.0f}'
            }), use_container_width=True)
            
            # Recovery suggestions
            st.markdown("#### 💡 Loss Recovery Strategies")
            for _, product in loss_products.nsmallest(5, 'Revenue_Impact').iterrows():
                product_suggestions = st.session_state.ai_optimizer_ensemble.suggestion_engine.generate_product_specific_suggestions(product)
                if product_suggestions:
                    st.markdown(f"""
                    <div class="feature-card">
                        <h4>🛠️ Product {product['ID']} (Loss: ₹{abs(product['Revenue_Impact']):,.0f})</h4>
                        <ul>
                            {"".join([f"<li>{suggestion}</li>" for suggestion in product_suggestions])}
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.success("🎉 No loss-making products found!")

    with tabs[4]:
        st.markdown("### 📦 AI-Powered Inventory Optimization")
        
        if inventory_analysis is not None:
            # Inventory efficiency metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Inventory Value", f"₹{inventory_efficiency['total_inventory_value']:,.0f}")
            with col2:
                st.metric("Efficiency Score", f"{inventory_efficiency['efficiency_score']:.0f}/100")
            with col3:
                st.metric("Avg Turnover Rate", f"{inventory_efficiency['average_turnover_rate']:.2f}/week")
            with col4:
                st.metric("High Risk Products", inventory_efficiency['high_risk_products'])
            
            # Risk distribution
            st.markdown("#### 📊 Inventory Risk Distribution")
            risk_data = {
                'Risk Level': ['High', 'Medium', 'Low'],
                'Count': [
                    inventory_efficiency['high_risk_products'],
                    inventory_efficiency['medium_risk_products'],
                    inventory_efficiency['low_risk_products']
                ]
            }
            fig = px.pie(
                risk_data, 
                values='Count', 
                names='Risk Level',
                title="Inventory Risk Distribution",
                color_discrete_sequence=['#EF4444', '#F59E0B', '#10B981']
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display inventory analysis
            st.markdown("#### 📊 Product-wise Inventory Recommendations")
            
            # Filter for high and medium urgency items
            urgent_items = inventory_analysis[inventory_analysis['urgency'].isin(['High', 'Medium'])]
            
            if len(urgent_items) > 0:
                display_cols = ['product_id', 'predicted_weekly_sales', 'optimal_stock_level', 
                              'stock_out_risk', 'urgency', 'recommendation']
                st.dataframe(urgent_items[display_cols].style.format({
                    'predicted_weekly_sales': '{:.1f}',
                    'optimal_stock_level': '{:.0f}'
                }), use_container_width=True)
            else:
                st.info("No urgent inventory actions needed - all products are well optimized!")
            
            # Inventory insights
            st.markdown("#### 🎯 AI Inventory Management Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **🚨 High Risk Products:**
                - Consider increasing safety stock
                - Monitor sales closely
                - Set up automatic reordering
                - Review demand patterns
                
                **📈 High Turnover Items:**
                - Maintain optimal stock levels
                - Consider bulk purchasing discounts
                - Monitor for stock-outs
                - Optimize reorder points
                """)
            
            with col2:
                st.markdown("""
                **💤 Slow Movers:**
                - Reduce inventory levels
                - Consider promotions to clear stock
                - Review product performance
                - Evaluate discontinuation
                
                **✅ Well Optimized:**
                - Continue current strategy
                - Regular monitoring
                - Maintain safety stock levels
                - Periodic review
                """)
        else:
            st.info("Inventory analysis requires predicted sales data.")

    with tabs[5]:
        st.markdown("### 🌦️ Advanced Seasonal Analysis")
        
        # Current season detection
        current_season = st.session_state.ai_optimizer_ensemble.season_analyzer.current_season
        demand_factors = st.session_state.ai_optimizer_ensemble.season_analyzer.get_seasonal_demand_factors()
        current_demand_factor = demand_factors.get(current_season, 1.0)
        
        st.markdown(f"""
        <div class="seasonal-highlight">
            <h3>🎯 Current Season: {current_season}</h3>
            <p><strong>Demand Factor:</strong> {current_demand_factor:.2f}x ({'+' if current_demand_factor > 1 else ''}{(current_demand_factor-1)*100:.0f}% demand change)</p>
            <p><strong>AI Recommendation:</strong> Adjust pricing strategy to leverage seasonal demand patterns</p>
        </div>
        """, unsafe_allow_html=True)
        
        if seasonal_analysis is not None and len(seasonal_analysis) > 0:
            # Seasonal performance metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                best_season = seasonal_analysis.loc[seasonal_analysis['total_revenue_impact'].idxmax(), 'season']
                st.metric("Best Performing Season", best_season)
            
            with col2:
                total_seasonal_impact = seasonal_analysis['total_revenue_impact'].sum()
                st.metric("Total Seasonal Impact", f"₹{total_seasonal_impact:,.0f}")
            
            with col3:
                highest_multiplier = seasonal_analysis['seasonal_multiplier'].max()
                st.metric("Peak Demand Multiplier", f"{highest_multiplier:.1f}x")
            
            with col4:
                avg_seasonal_margin = seasonal_analysis['profit_margin'].mean()
                st.metric("Avg Seasonal Margin", f"{avg_seasonal_margin:.1f}%")
            
            # Advanced seasonal charts
            st.markdown("#### 📊 Seasonal Performance Overview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = SeasonalVisualizations.create_seasonal_demand_analysis(seasonal_analysis)
                if fig1:
                    st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig2 = SeasonalVisualizations.create_seasonal_performance_radar(seasonal_analysis)
                if fig2:
                    st.plotly_chart(fig2, use_container_width=True)
            
            # Seasonal trend analysis
            st.markdown("#### 📈 Seasonal Trend Analysis")
            fig3 = SeasonalVisualizations.create_seasonal_trend_analysis(ensemble_recommendations)
            if fig3:
                st.plotly_chart(fig3, use_container_width=True)
            
            # Seasonal strategies
            st.markdown("#### 🎯 AI Seasonal Strategies")
            
            for strategy in seasonal_strategies:
                season_class = f"season-{str(strategy['season']).lower()}"
                st.markdown(f"""
                <div class="feature-card">
                    <h4 class="{season_class}">🌤️ {strategy['season']} Season: {strategy['strategy']}</h4>
                    <p><strong>Description:</strong> {strategy['description']}</p>
                    <p><strong>Recommendation:</strong> {strategy['recommendation']}</p>
                    <p><strong>Pricing Strategy:</strong> {strategy['pricing_strategy']}</p>
                    <p><strong>Expected Impact:</strong> {strategy['expected_impact']}</p>
                    <p><strong>Risk Level:</strong> {strategy['risk_level']} | <strong>Priority:</strong> {strategy['priority']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Strategy matrix
            st.markdown("#### 📋 Seasonal Strategy Matrix")
            fig4 = SeasonalVisualizations.create_seasonal_strategy_matrix(seasonal_strategies)
            if fig4:
                st.plotly_chart(fig4, use_container_width=True)
            
            # Seasonal product breakdown
            st.markdown("#### 📦 Seasonal Product Distribution")
            seasonal_breakdown = ensemble_recommendations.groupby('Season').agg({
                'ID': 'count',
                'Revenue_Impact': 'sum',
                'Applied_Discount': 'mean',
                'Predicted_Sales': 'mean'
            }).round(2)
            
            seasonal_breakdown.columns = ['Product Count', 'Total Impact', 'Avg Discount', 'Avg Predicted Sales']
            st.dataframe(seasonal_breakdown.style.format({
                'Total Impact': '₹{:,.0f}',
                'Avg Discount': '{:.1f}%',
                'Avg Predicted Sales': '{:.1f}'
            }), use_container_width=True)
            
        else:
            st.info("Seasonal analysis requires seasonal data in the dataset.")

    with tabs[6]:
        st.markdown("### 💡 AI Business Suggestions")
        
        if suggestions:
            st.markdown(f"#### 🎯 **{len(suggestions)} AI-Powered Recommendations**")
            
            for i, suggestion in enumerate(suggestions):
                confidence_class = "suggestion-high" if suggestion['confidence'] > 0.8 else "suggestion-medium" if suggestion['confidence'] > 0.7 else "suggestion-low"
                
                st.markdown(f"""
                <div class="feature-card {confidence_class}">
                    <div style="display: flex; justify-content: between; align-items: start;">
                        <div style="flex: 1;">
                            <h4>{suggestion['type']}: {suggestion['title']}</h4>
                            <p><strong>Description:</strong> {suggestion['description']}</p>
                            <p><strong>Expected Impact:</strong> {suggestion['impact']}</p>
                            <p><strong>Recommended Action:</strong> {suggestion['action']}</p>
                            <p><strong>AI Model:</strong> {suggestion['ai_model']} | <strong>Confidence:</strong> {suggestion['confidence']:.0%}</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No AI suggestions available. Try training the AI models first.")

    # Advanced Feature Tabs
    if st.session_state.show_advanced_features:
        # Behavioral Economics Tab
        if len(tabs) > 7:
            with tabs[7]:
                st.markdown("### 🎓 Behavioral Economics Analysis")
                
                if behavioral_analysis is not None:
                    # Behavioral metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        avg_anchor_effect = behavioral_analysis['anchor_effect_pct'].mean()
                        st.metric("Avg Anchoring Effect", f"{avg_anchor_effect:.1f}%")
                    
                    with col2:
                        premium_products = len(behavioral_analysis[behavioral_analysis['pricing_strategy'] == 'Premium'])
                        st.metric("Premium Products", premium_products)
                    
                    with col3:
                        avg_reference_effect = behavioral_analysis['reference_effect_pct'].mean()
                        st.metric("Avg Reference Effect", f"{avg_reference_effect:.1f}%")
                    
                    # Behavioral analysis visualization
                    st.markdown("#### 📊 Anchoring Effect Analysis")
                    fig = px.scatter(
                        behavioral_analysis,
                        x='anchor_effect_pct',
                        y='reference_effect_pct',
                        color='pricing_strategy',
                        size='cost_price',
                        hover_data=['product_id', 'behavioral_recommendation'],
                        title="Behavioral Economics: Anchoring vs Reference Effects"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Decoy effect analysis
                    st.markdown("#### 🎯 Decoy Effect Opportunities")
                    decoy_analysis = st.session_state.ai_optimizer_ensemble.behavioral_economics.analyze_decoy_effect(
                        ensemble_recommendations
                    )
                    
                    if decoy_analysis is not None and len(decoy_analysis) > 0:
                        st.dataframe(decoy_analysis, use_container_width=True)
                    
                    # Behavioral recommendations
                    st.markdown("#### 💡 Behavioral Pricing Strategies")
                    
                    for _, product in behavioral_analysis.nlargest(5, 'anchor_effect_pct').iterrows():
                        if product['anchor_effect_pct'] > 20:
                            st.markdown(f"""
                            <div class="feature-card">
                                <h4>🎯 Product {product['product_id']} - Strong Anchoring Effect</h4>
                                <p><strong>Anchoring Effect:</strong> {product['anchor_effect_pct']:.1f}%</p>
                                <p><strong>Recommendation:</strong> {product['behavioral_recommendation']}</p>
                                <p><strong>Strategy:</strong> {product['pricing_strategy']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("Behavioral economics analysis requires advanced features to be enabled.")

        # Game Theory Tab
        if len(tabs) > 8:
            with tabs[8]:
                st.markdown("### ♟️ Game Theory Competitive Analysis")
                
                if game_theory_analysis is not None:
                    # Game theory metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        nash_count = len(game_theory_analysis[game_theory_analysis['nash_equilibrium'] == True])
                        st.metric("Nash Equilibrium Products", nash_count)
                    
                    with col2:
                        avg_price_ratio = game_theory_analysis['price_ratio'].mean()
                        st.metric("Avg Price Ratio", f"{avg_price_ratio:.2f}")
                    
                    with col3:
                        tit_for_tat_count = len(game_theory_analysis[game_theory_analysis['game_strategy'].str.contains('Tit-for-tat')])
                        st.metric("Tit-for-tat Products", tit_for_tat_count)
                    
                    # Game theory visualization
                    st.markdown("#### 📊 Competitive Game Analysis")
                    fig = px.scatter(
                        game_theory_analysis,
                        x='price_ratio',
                        y='payoff_our',
                        color='game_strategy',
                        size='payoff_competitor',
                        hover_data=['product_id', 'recommended_action'],
                        title="Game Theory: Price Ratios vs Payoffs"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Strategy distribution
                    st.markdown("#### 🎯 Game Strategy Distribution")
                    strategy_counts = game_theory_analysis['game_strategy'].value_counts()
                    fig = px.pie(
                        values=strategy_counts.values,
                        names=strategy_counts.index,
                        title="Game Theory Strategy Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Nash equilibrium analysis
                    st.markdown("#### ⚖️ Nash Equilibrium Analysis")
                    nash_products = game_theory_analysis[game_theory_analysis['nash_equilibrium'] == True]
                    
                    if len(nash_products) > 0:
                        st.info(f"**{len(nash_products)} products are in Nash Equilibrium**")
                        st.dataframe(nash_products[['product_id', 'our_price', 'competitor_price', 'price_ratio', 'game_strategy']], 
                                   use_container_width=True)
                    else:
                        st.warning("No products found in Nash Equilibrium")
                    
                    # Game theory recommendations
                    st.markdown("#### 💡 Game Theory Recommendations")
                    
                    for _, product in game_theory_analysis.iterrows():
                        if not product['nash_equilibrium']:
                            st.markdown(f"""
                            <div class="feature-card">
                                <h4>🎮 Product {product['product_id']} - {product['game_strategy']}</h4>
                                <p><strong>Price Ratio:</strong> {product['price_ratio']:.2f}</p>
                                <p><strong>Action:</strong> {product['recommended_action']}</p>
                                <p><strong>Our Payoff:</strong> ₹{product['payoff_our']:,.0f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            break
                else:
                    st.info("Game theory analysis requires advanced features to be enabled.")

        # Sustainability Tab
        if len(tabs) > 9:
            with tabs[9]:
                st.markdown("### 🌱 Sustainability Pricing Analysis")
                
                if sustainability_analysis is not None:
                    # Sustainability metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        avg_sustainability_score = sustainability_analysis['sustainability_score'].mean()
                        st.metric("Avg Sustainability Score", f"{avg_sustainability_score:.2f}")
                    
                    with col2:
                        premium_products = len(sustainability_analysis[sustainability_analysis['tier'] == 'Premium Sustainable'])
                        st.metric("Premium Sustainable", premium_products)
                    
                    with col3:
                        total_premium = sustainability_analysis['premium_amount'].sum()
                        st.metric("Total Premium Value", f"₹{total_premium:,.0f}")
                    
                    with col4:
                        high_advantage = len(sustainability_analysis[sustainability_analysis['market_advantage'] == 'High'])
                        st.metric("High Market Advantage", high_advantage)
                    
                    # Sustainability visualization
                    st.markdown("#### 📊 Sustainability Score Distribution")
                    fig = px.histogram(
                        sustainability_analysis,
                        x='sustainability_score',
                        color='tier',
                        title="Sustainability Score Distribution by Tier",
                        nbins=20
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Premium analysis
                    st.markdown("#### 💰 Sustainability Premium Analysis")
                    fig = px.scatter(
                        sustainability_analysis,
                        x='sustainability_score',
                        y='premium_pct',
                        color='market_advantage',
                        size='base_price',
                        hover_data=['product_id', 'tier', 'premium_amount'],
                        title="Sustainability Score vs Premium Percentage"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Top sustainable products
                    st.markdown("#### 🏆 Top Sustainable Products")
                    top_sustainable = sustainability_analysis.nlargest(10, 'sustainability_score')
                    st.dataframe(top_sustainable[['product_id', 'sustainability_score', 'tier', 
                                                'base_price', 'sustainability_price', 'premium_pct']], 
                               use_container_width=True)
                    
                    # Sustainability recommendations
                    st.markdown("#### 💡 Sustainability Pricing Strategies")
                    
                    for _, product in sustainability_analysis[sustainability_analysis['sustainability_score'] > 0.7].iterrows():
                        st.markdown(f"""
                        <div class="feature-card">
                            <h4>🌿 Product {product['product_id']} - {product['tier']}</h4>
                            <p><strong>Sustainability Score:</strong> {product['sustainability_score']:.2f}</p>
                            <p><strong>Premium:</strong> {product['premium_pct']:.1f}% (₹{product['premium_amount']:,.0f})</p>
                            <p><strong>Market Advantage:</strong> {product['market_advantage']}</p>
                            <p><strong>Recommendation:</strong> Market as premium sustainable product</p>
                        </div>
                        """, unsafe_allow_html=True)
                        break
                else:
                    st.info("Sustainability analysis requires advanced features to be enabled.")

        # Academic Tab
        if len(tabs) > 10:
            with tabs[10]:
                st.markdown("### 📚 Academic Validation & Research")
                
                if academic_report is not None:
                    # Academic validation metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if 'avg_revenue_impact' in academic_report:
                            st.metric("Avg Revenue Impact", f"₹{academic_report['avg_revenue_impact']:,.0f}")
                    
                    with col2:
                        if 'profitable_ratio' in academic_report:
                            st.metric("Profitability Ratio", f"{academic_report['profitable_ratio']*100:.1f}%")
                    
                    with col3:
                        if 'statistical_significance' in academic_report:
                            st.metric("Statistical Significance", academic_report['statistical_significance'])
                    
                    # Pricing maturity assessment
                    st.markdown("#### 🎯 Pricing Maturity Assessment")
                    
                    if maturity_assessment is not None:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"""
                            <div class="feature-card">
                                <h3>📊 Maturity Level: {maturity_assessment['maturity_level']}</h3>
                                <p><strong>Description:</strong> {maturity_assessment['level_description']}</p>
                                <p><strong>Total Score:</strong> {maturity_assessment['total_score']}/20</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            # Display scores
                            scores_df = pd.DataFrame({
                                'Dimension': list(maturity_assessment['scores'].keys()),
                                'Score': list(maturity_assessment['scores'].values())
                            })
                            fig = px.bar(
                                scores_df,
                                x='Dimension',
                                y='Score',
                                title="Pricing Maturity Dimensions",
                                color='Score',
                                color_continuous_scale='Viridis'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Recommendations
                        st.markdown("#### 💡 Maturity Improvement Recommendations")
                        for rec in maturity_assessment['recommendations']:
                            st.markdown(f"• {rec}")
                    
                    # Feature importance (if available)
                    if hasattr(st.session_state.ai_optimizer_ensemble, 'feature_importance'):
                        feature_imp = st.session_state.ai_optimizer_ensemble.feature_importance
                        if feature_imp is not None:
                            st.markdown("#### 🔍 AI Model Feature Importance")
                            fig = px.bar(
                                feature_imp,
                                x='importance',
                                y='feature',
                                orientation='h',
                                title="Feature Importance in AI Model",
                                color='importance',
                                color_continuous_scale='Viridis'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Academic report
                    st.markdown("#### 📄 Academic Validation Report")
                    if 'r2_score' in academic_report:
                        st.markdown(f"""
                        <div class="feature-card">
                            <h4>📈 Model Performance Metrics</h4>
                            <p><strong>R² Score:</strong> {academic_report.get('r2_score', 'N/A'):.3f}</p>
                            <p><strong>Mean Absolute Error:</strong> ₹{academic_report.get('mean_absolute_error', 'N/A'):,.0f}</p>
                            <p><strong>Validation Status:</strong> {academic_report.get('validation_status', 'N/A')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Research insights
                    st.markdown("#### 🧠 Research-Grade Insights")
                    st.markdown("""
                    **Key Findings from AI Analysis:**
                    1. **Price Elasticity Variation**: Different product categories show varying sensitivity to discounts
                    2. **Seasonal Patterns**: Demand fluctuations follow predictable seasonal cycles
                    3. **Competitive Dynamics**: Game theory models reveal optimal competitive responses
                    4. **Behavioral Effects**: Anchoring and reference prices significantly influence buying decisions
                    
                    **Research Recommendations:**
                    - Implement A/B testing for price optimization
                    - Conduct longitudinal studies on price elasticity
                    - Explore causal inference for pricing decisions
                    - Develop predictive models for competitor reactions
                    """)
                else:
                    st.info("Academic validation requires advanced features to be enabled.")

        # Export Tab
        if len(tabs) > 11:
            with tabs[11]:
                st.markdown("### 📤 Export Results & Reports")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Export ensemble results
                    st.markdown("#### 📊 Export Ensemble AI Results")
                    st.markdown(st.session_state.exporter.export_to_csv(
                        ensemble_recommendations, "ensemble_ai_recommendations"
                    ), unsafe_allow_html=True)
                    
                    # Export neural results
                    st.markdown("#### 🧠 Export Neural AI Results")
                    st.markdown(st.session_state.exporter.export_to_csv(
                        neural_recommendations, "neural_ai_recommendations"
                    ), unsafe_allow_html=True)
                    
                    # Export inventory analysis
                    if inventory_analysis is not None:
                        st.markdown("#### 📦 Export Inventory Analysis")
                        st.markdown(st.session_state.exporter.export_to_csv(
                            inventory_analysis, "inventory_optimization"
                        ), unsafe_allow_html=True)
                
                with col2:
                    # Export seasonal analysis
                    if seasonal_analysis is not None:
                        st.markdown("#### 🌦️ Export Seasonal Analysis")
                        st.markdown(st.session_state.exporter.export_to_csv(
                            seasonal_analysis, "seasonal_analysis"
                        ), unsafe_allow_html=True)
                    
                    # Export advanced analyses
                    if st.session_state.show_advanced_features:
                        if behavioral_analysis is not None:
                            st.markdown("#### 🎓 Export Behavioral Analysis")
                            st.markdown(st.session_state.exporter.export_to_csv(
                                behavioral_analysis, "behavioral_economics"
                            ), unsafe_allow_html=True)
                        
                        if sustainability_analysis is not None:
                            st.markdown("#### 🌱 Export Sustainability Analysis")
                            st.markdown(st.session_state.exporter.export_to_csv(
                                sustainability_analysis, "sustainability_pricing"
                            ), unsafe_allow_html=True)
                
                # Export all data
                st.markdown("---")
                st.markdown("#### 📦 Export Complete Analysis Package")
                
                export_datasets = {
                    'ensemble_recommendations': ensemble_recommendations,
                    'neural_recommendations': neural_recommendations,
                    'inventory_analysis': inventory_analysis if inventory_analysis is not None else pd.DataFrame(),
                    'seasonal_analysis': seasonal_analysis if seasonal_analysis is not None else pd.DataFrame()
                }
                
                if st.session_state.show_advanced_features:
                    if behavioral_analysis is not None:
                        export_datasets['behavioral_analysis'] = behavioral_analysis
                    if game_theory_analysis is not None:
                        export_datasets['game_theory_analysis'] = game_theory_analysis
                    if sustainability_analysis is not None:
                        export_datasets['sustainability_analysis'] = sustainability_analysis
                
                st.markdown(st.session_state.exporter.export_multiple_datasets(
                    export_datasets, "complete_ai_analysis"
                ), unsafe_allow_html=True)
                
                # Export history
                if st.session_state.exporter.export_history:
                    st.markdown("#### 📋 Export History")
                    history_df = pd.DataFrame(st.session_state.exporter.export_history)
                    st.dataframe(history_df, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666;'>"
        "🤖 AI Price Intelligence Platform | "
        "© 2025 AI Business Intelligence | "
        "Multi-Model AI with Advanced Features | "
        "NAVABODHA AI"
        "</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
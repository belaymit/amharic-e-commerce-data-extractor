# Task 6: FinTech Vendor Scorecard for Micro-Lending

## 🎯 Objective
Develop a comprehensive vendor analytics engine that combines NER-extracted entities with engagement metrics to create lending scores for micro-lending decisions.

## 🏦 Business Context

**EthioMart's Micro-Lending Initiative**

EthioMart wants to identify its most active and promising vendors to offer them small business loans (micro-lending). A lender can't assess a vendor based on text alone; they need to see evidence of business activity and customer engagement.

## 📊 Vendor Analytics Engine

### Key Performance Indicators (KPIs)

#### 1. Activity & Consistency
- **Posting Frequency**: Average number of posts per week
- **Activity Period**: How long the vendor has been active
- **Consistency**: Regularity of posting patterns

#### 2. Market Reach & Engagement
- **Average Views per Post**: Primary indicator of customer reach
- **Total Views**: Overall market exposure
- **Best Post Performance**: Peak engagement levels
- **Engagement Consistency**: Reliability of audience reach

#### 3. Business Profile (from NER Model)
- **Average Price Point**: Market segment positioning
- **Product Diversity**: Business breadth and specialization
- **Location Coverage**: Geographic market presence
- **Price Stability**: Business model consistency

## 🔢 Lending Score Calculation

### Weighted Scoring Model (0-100 scale)

```
Lending Score = (Market Reach × 30%) + 
                (Activity Level × 25%) + 
                (Business Diversification × 20%) + 
                (Business Viability × 15%) + 
                (Performance Consistency × 10%)
```

### Component Details

1. **Market Reach (30%)**
   - Based on average views per post
   - Normalized to 0-1 scale (cap at 2000 views)
   - Indicates customer base size

2. **Activity Level (25%)**
   - Posts per week frequency
   - Normalized to 0-1 scale (cap at 7 posts/week)
   - Shows business commitment

3. **Business Diversification (20%)**
   - Number of unique product types
   - Normalized to 0-1 scale (cap at 5 products)
   - Reduces concentration risk

4. **Business Viability (15%)**
   - Average price point factor
   - Minimum threshold: 500 ETB
   - Indicates sustainable business model

5. **Performance Consistency (10%)**
   - Inverse of engagement variability
   - Lower variability = higher score
   - Shows reliable performance

## 🚦 Risk Assessment Categories

### 🟢 LOW RISK (Score ≥ 75)
- **Description**: Excellent lending candidate
- **Characteristics**: High engagement, consistent activity, diversified business
- **Loan Amount**: Up to 50,000 ETB
- **Monitoring**: Standard quarterly reviews

### 🟡 MEDIUM RISK (Score 50-74)
- **Description**: Good candidate with monitoring
- **Characteristics**: Moderate engagement, regular activity, some diversification
- **Loan Amount**: 20,000-35,000 ETB
- **Monitoring**: Monthly performance reviews

### 🔴 HIGH RISK (Score 25-49)
- **Description**: Requires careful evaluation
- **Characteristics**: Limited engagement, irregular activity, narrow focus
- **Loan Amount**: 10,000-20,000 ETB
- **Monitoring**: Weekly check-ins

### ⚫ VERY HIGH RISK (Score < 25)
- **Description**: Not recommended for lending
- **Characteristics**: Poor engagement, minimal activity, unstable business
- **Loan Amount**: Not recommended
- **Action**: Business development support

## 📈 Key Metrics Tracked

### Vendor-Level Metrics
| Metric | Description | Weight | Source |
|--------|-------------|--------|--------|
| Avg Views/Post | Customer engagement indicator | 30% | Telegram metadata |
| Posts/Week | Business activity level | 25% | Message frequency |
| Product Diversity | Business breadth | 20% | NER extraction |
| Avg Price (ETB) | Market positioning | 15% | NER extraction |
| Consistency Score | Performance reliability | 10% | Statistical analysis |

### Business Intelligence Outputs
- **Market Segmentation**: High/medium/low price vendors
- **Category Performance**: Electronics vs Fashion vs General
- **Geographic Analysis**: Location-based business patterns
- **Seasonal Trends**: Activity patterns over time

## 🛠️ Implementation

### Main Script Usage
```bash
# Run Task 6 analysis
python main.py --task 6

# Interactive mode
python main.py --interactive
# Then select option 6
```

### Jupyter Notebook Analysis
```bash
cd notebooks/
jupyter notebook Vendor_Scorecard.ipynb
```

## 📊 Outputs Generated

### 1. Vendor Scorecard CSV
- `data/vendor_analytics/lending_scorecard.csv`
- Complete vendor metrics and scores
- Ready for loan approval workflows

### 2. Executive Summary
- `data/vendor_analytics/executive_summary.csv`
- Key metrics for management review
- Risk category distributions

### 3. Loan Portfolio Report
- `data/vendor_analytics/loan_portfolio.csv`
- Approved vendors with recommended amounts
- Risk tier classifications

### 4. JSON API Data
- `data/vendor_analytics/vendor_analysis.json`
- API-ready format for system integration
- Real-time scoring updates

## 📋 Sample Output Format

```
🏆 VENDOR SCORECARD RESULTS
================================================================================

📊 Ethiopian Fashion Store (@EthioFashion)
   • Avg. Views/Post: 1,200
   • Posts/Week: 5.2
   • Avg. Price (ETB): 2,100
   • Product Diversity: 4 types
   • Best Post: 1,850 views
   • 🏦 LENDING SCORE: 78.5/100
   • Risk Assessment: 🟢 LOW RISK - Excellent lending candidate
   • Recommended Loan: 45,000 ETB
```

## 💡 Business Recommendations

### Immediate Actions
1. **Approve Low-Risk Vendors** (Score ≥ 75)
   - Fast-track loan processing
   - Standard interest rates
   - Quarterly performance reviews

2. **Conditional Approval** (Score 50-74)
   - Additional documentation required
   - Slightly higher interest rates
   - Monthly monitoring

3. **Business Development** (Score < 50)
   - Training and mentorship programs
   - Digital marketing support
   - Gradual qualification pathway

### Risk Mitigation
- **Digital Monitoring**: Automated tracking of posting frequency and engagement
- **Performance Milestones**: 1, 3, and 6-month checkpoints
- **Portfolio Diversification**: Balance across categories and risk levels

## 🔍 Success Criteria
- [ ] Vendor analytics engine operational
- [ ] Lending scores calculated for all vendors
- [ ] Risk assessment categories assigned
- [ ] Business intelligence dashboard created
- [ ] Loan recommendations generated
- [ ] Export files created for loan processing

## 📚 Integration Points

### With Previous Tasks
- **Task 1**: Uses scraped vendor data
- **Task 2**: Leverages CoNLL-labeled training data
- **Task 3**: Incorporates NER model predictions
- **Task 4**: Benefits from best model selection
- **Task 5**: Includes interpretability insights

### For Business Operations
- **Loan Approval Workflow**: Automated scoring integration
- **Risk Management**: Real-time vendor monitoring
- **Business Intelligence**: Market trend analysis
- **Customer Relationship**: Vendor success programs

## 🎯 Expected Business Impact

### Financial Inclusion
- **Qualified Vendors**: 60-80% of active vendors eligible
- **Loan Pool**: 500,000-2,000,000 ETB total capacity
- **Default Reduction**: 15-25% improvement in loan performance
- **Processing Efficiency**: 70% faster loan decisions

### Market Development
- **Vendor Growth**: 20-30% increase in business activity
- **Platform Adoption**: Higher vendor engagement
- **Data-Driven Insights**: Evidence-based lending decisions
- **Competitive Advantage**: First-mover in AI-powered micro-lending

## 🚀 Future Enhancements

### Advanced Analytics
- **Machine Learning**: Predictive default modeling
- **Time Series**: Seasonal business pattern analysis
- **Network Analysis**: Vendor relationship mapping
- **Sentiment Analysis**: Customer feedback incorporation

### Real-Time Integration
- **Live Scoring**: Continuous vendor assessment
- **API Development**: Third-party integration capabilities
- **Mobile Dashboard**: On-the-go lending decisions
- **Automated Alerts**: Risk threshold notifications

---

**Task 6 provides the crucial business intelligence layer that transforms raw NER data into actionable financial decisions for EthioMart's micro-lending initiative.** 
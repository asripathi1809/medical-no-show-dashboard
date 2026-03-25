# medical-no-show-dashboard

Medical No-Show Analytics & Intervention Strategy
Overview

Missed medical appointments are a persistent operational challenge—impacting provider utilization, delaying care, and creating avoidable revenue loss.

Using a dataset of 110,000+ appointments (22,000 no-shows) from Brazil, I built an interactive Tableau dashboard and analytical workflow to identify high-risk patient segments and simulate targeted intervention strategies.

The goal was not just to analyze no-shows—but to answer:

Where should a hospital focus its limited outreach resources to reduce missed appointments most effectively?

Problem Framing
~20% of scheduled appointments result in no-shows (~1 in 5)
This creates:
Underutilized provider capacity
Delayed patient care
Inefficient allocation of scheduling resources

Objective:
Identify actionable drivers of no-shows and translate them into prioritized, cost-effective intervention strategies

Data & Methodology
Dataset: Medical No-Shows (Kaggle)
Tools: Python (cleaning, feature prep), Tableau (visualization), Excel (exploration)
Key steps:
Cleaned and validated inconsistent patient and appointment records
Engineered features such as:
Age groups
Chronic condition indicators
Neighborhood-level patterns
Segmented patients to identify high-risk cohorts, not just overall trends
Key Insights (Reframed for Action)
Lead time + patient history effects: Certain groups consistently exhibited higher no-show rates, indicating predictable behavioral patterns
Non-chronic minors emerged as a high-risk segment, suggesting lower perceived urgency for appointments
Geographic variation (e.g., specific neighborhoods) revealed concentrated pockets of no-show risk
Temporal patterns suggested weekday-based variation, enabling targeted scheduling interventions

Intervention Strategy (What a hospital could actually do)

Instead of broad outreach, this analysis supports targeted intervention allocation:

Prioritize reminder calls / notifications for high-risk appointment windows (e.g., midweek slots)
Focus outreach on high-risk neighborhoods, where marginal impact is highest
Design targeted communication strategies for low-engagement groups (e.g., minors without chronic conditions)
Business Impact (Modeled Scenario)

To evaluate practical value, I modeled a conservative intervention scenario:

Avg. cost per appointment: ~$100
If no-shows reduced by 10%:
~2,200 additional completed appointments annually
≈ $220K recovered value

Estimated intervention cost (targeted outreach): ~$5K/year

Important Tradeoffs & Considerations
Increasing recall (identifying more high-risk patients) may increase false positives, leading to unnecessary outreach
Intervention effectiveness depends on patient responsiveness, not just identification
Dataset limitations (e.g., missing behavioral/contextual factors) may affect generalizability
Dashboard

Interactive Tableau dashboard:
https://public.tableau.com/app/profile/ananya.sripathi/viz/Why1in5PeopleMissAppointmentsinBrazil/Dashboard2

What I’d Do Next
Incorporate predictive modeling for appointment-level risk scoring
Run A/B tests on intervention strategies (reminders vs overbooking)
Integrate with scheduling systems for real-time decision support
Refine ROI model using actual intervention outcomes
Author

Ananya Sripathi
Data Analytics | Healthcare & Predictive Modeling



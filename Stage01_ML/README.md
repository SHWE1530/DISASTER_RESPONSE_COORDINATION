# Stage 01: Machine Learning (Disaster Response Coordination)

This directory contains the machine learning pipeline for generating Zone Risk Scores.

## FIELD BRIEFING SHEET: ZONE RISK SCORE MODEL

### Overview
This reference guide helps Emergency Response Teams understand the **Zone Risk Score** output provided by the Disaster Response Coordination System (Stage 01).

Our model turns continuous sensor data (river levels, rainfall), weather patterns, and emergency-call volumes into an instant risk score (Low, Moderate, Severe) for every city zone, allowing you to prioritize deployments effectively.

---

### The Risk Categories

#### 🟢 LOW RISK
- **Definition:** Normal conditions or minor weather events. Infrastructure is handling the load. 
- **Typical Indicators:** Rainfall < 20mm, River levels well below danger thresholds, Emergency call volumes within normal daily averages.
- **Action:** Routine monitoring. No immediate field deployment required.

#### 🟡 MODERATE RISK
- **Definition:** Elevated stress on infrastructure. Potential for localized waterlogging, isolated road closures, or minor disruptions.
- **Typical Indicators:** Continuous rainfall, River levels approaching warning thresholds, noticeable uptick in emergency calls.
- **Action:** Alert field teams. Pre-position equipment in vulnerable zones. Monitor for escalation.

#### 🔴 SEVERE RISK (HIGH PRIORITY)
- **Definition:** Imminent or ongoing critical disaster event. High probability of widespread flooding, bridge/road closures, and significant population impact.
- **Numerical Definition of Severe:** The system classifies a zone as SEVERE when:
  - **River Level:** Exceeds the designated danger threshold for the district.
  - **Emergency Calls & Rainfall:** A sustained surge in calls (>= 35) combined with heavy rainfall (>= 40mm).
- **Action:** Immediate full-scale deployment. Execute evacuation protocols for affected populations. Coordinate with central command for bridge/road closures.

---

### Top Risk Factors (What the Model Looks At)

When assessing a zone, the intelligence system prioritizes these signals in order:

1. **River Level (m):** The absolute height of the river compared to historical baselines.
2. **Rainfall (mm):** The rolling accumulation of rain in the district.
3. **Bridge Closures:** Infrastructure failure is a leading indicator of severe impact.
4. **Population Affected:** Density of people in the impacted zone.
5. **Road Closures:** Disruptions to transport networks.

---

### How to use the Dashboard

1. **Live Assessment:** Enter the latest readings from the field (Rainfall, River Level, Call Volumes) into the "Live risk assessment" panel to get an instant classification.
2. **Confidence Score:** Pay attention to the Confidence Score percentage. A high confidence (e.g., >95%) means the current conditions strongly match past disasters.
3. **Monitor Trends:** Use the "Risk Trend" and "Risk by Region" panels to see which districts have the highest frequency of Severe events.

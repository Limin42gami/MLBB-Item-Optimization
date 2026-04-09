📌 MLBB Item Optimization

This project focuses on building a machine learning–based system to recommend optimal item builds and emblem configurations for Mobile Legends: Bang Bang based on in-game draft conditions.

🧩 Essential Data

The model is trained using the following core inputs:
1. Hero selection
2. Item builds
3. Emblem configurations

⚙️ Key Features
- Emblem and items Recommendation
- Suggests optimal emblem and items setups based on:
    1. Selected hero
    2. Enemy composition (countering)
    3. Allied composition (synergy)
- Item Priority & Build Path Optimization: -
    1. Core item priority (which items to complete first)
    2. Build path sequencing priority (which components to build before completing full items)
 
- Item Build Recommendation generates item builds by considering: -
       - Enemy and allied heroes
       - Game duration
       - Rank and skill level
       - Damage composition (physical vs magic)

🔄 System Flow
1. User pick a hero
2. User input enemy heroes
3. User input allied heroes
4. The model predicts:
   - Optimal item builds
   - Recommended build order

5. The system outputs:
    - Core items (Items that the hero need to buy no matter the situation)
    - Situational items (Items that the hero need to buy based on situation)
    - Build Order

🔍 Current Observations
1. Predictions may lack accuracy due to limited rule integration.
   - This system relies primarily on machine learning, which can lead to suboptimal recommendations without domain-specific constraints.
   - Example issue: Anti-heal items may be over-prioritized, even though their unique passives do not stack, resulting in inefficient builds.
2. Accuracy of predictions worsen due to game duration, rank and skill level consideration.
   - Including these features introduced noise and reduced model performance.
   - No matter the rank, skill level, and game duration, every hero has their core item and situational item 
     
3. The current interface is not user-friendly and is difficult to use during gameplay. A more accessible and responsive UI is required for real-time usability.

🚀 Planned Improvements
1. Introduce a rule-based layer to complement the machine learning model
   - Example: Prevent stacking of identical unique passives
   - Develop a structured system to standardize item effects and passives
   - This will improve rule implementation and model accuracy
2. Remove game duration, rank and skill level features.
3. Build a proper UI.
   
⚠️ Current Challenges
1. Inconsistent Passive Descriptions
   - Different items may have unique passives with varying descriptions despite providing similar effects.
   - This inconsistency makes it difficult to define and enforce clear rules within the system.

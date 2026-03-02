Abstract

The decoupling of stochastic renewable energy generation from critical water demand in off-grid
solar irrigation systems presents a fundamental control challenge. This is particularly acute in
hydrogeologically constrained environments, where aggressive extraction can lead to irreversible
aquifer damage [1]

This study proposes a Hierarchical Model Predictive Control (HMPC) framework that integrates
a high-fidelity hydraulic model based on Thiem’s equilibrium equations. The architecture utilizes
a two-layer strategy: a 6-day High-Level Planner for volumetric allocation and an hourly MPC
Regulator for real-time constraint enforcement. Utilizing real-world TMY data for Nairobi, the
simulation evaluates system resilience.

Under constrained aquifer conditions, characterized by low transmissivity (T = 8.7 × 10⁻5 m²/s), 
a 3000 Wp PV system achieved superior operational results compared to larger systems. While 
intuition might suggest that more power leads to more water, increasing the PV array size (e.g., 
to 5000 Wp or more) without a corresponding increase in aquifer yield leads to the Oversizing 
Paradox.

The larger power input from oversized systems dewaters the borehole too quickly, causing the 
dynamic drawdown to hit critical safety limits and triggering frequent, long pump lockout 
periods. Furthermore, larger systems are hindered by a higher start-up power threshold (modeled 
as 20% of rated capacity), meaning they remain inactive during low-irradiance hours that a 3000 
Wp system can successfully utilize.

Under relaxed hydraulic conditions, the "Oversizing Paradox" is mitigated because the aquifer 
can replenish the borehole nearly as fast as the pump extracts water. This environment is 
characterized by high transmissivity (T = 1.0 × 10⁻¹ m²/s), which prevents the rapid dewatering 
of the borehole that typically triggers safety lockouts in constrained geologies. Consequently, a 
"more is better" relationship between PV capacity and water yield is established, as the system is 
limited primarily by available solar energy rather than the aquifer's recharge rate.

The results demonstrate that "hydro-aware" control strategies are essential for the sustainable 
intensification of solar irrigation, confirming that aquifer properties, rather than solar peak
power, are the governing constraint in many African contexts.

References
[1] "Meunier, S., et al. (2023). Aquifer conditions, not irradiance determine the potential of 
photovoltaic energy.". 
[2] "Bwambale, E., et al. (2023). Data-driven model predictive control for precision irrigation.". 
[3] " Scattolini, R. and Colaneri, P. (2007). Hierarchical Model Predictive Control.".

Social Post (LinkedIN): https://www.linkedin.com/posts/samuel-ochor_over-the-last-five-months-i-have-had-the-activity-7424692500897968129-ulDf?utm_source=social_share_send&utm_medium=member_desktop_web&rcm=ACoAACV0gGwBloTf3EGWuVsJyWnLVfmk77aWweg

import numpy
import time
import os  # Needed to create the output folder
import matplotlib.pyplot as matplotlibPlotter
from scipy.optimize import minimize
import warnings
import pvlib

# Suppress runtime warnings from solver
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==========================================
# 1. SYSTEM DESIGN AND PARAMETERS
# ==========================================

# Borehole and Aquifer Parameters
# Borehole and Aquifer Parameters
boreholeParameters = {
    'staticWaterDepth': 15.0,
    'aquiferTransmissivity': 1.0e-1,
    'boreholeRadius': 0.5,  # 1m diameter well
    'boreholeLossCoefficient': 10.0,
    'coneOfDepressionRadius': 500,
    'saturatedThickness': 100.0,
}

boreholeParameters['motorPumpDepth'] = boreholeParameters['staticWaterDepth'] + 100

pipeParameters = {
    'linearPipeLossCoefficient': 1.0,
    'junctionPipeLossCoefficient': 1.0,
    'pipeLength': boreholeParameters['motorPumpDepth']
}

# --- HARDWARE SIZING ---
design_capacity_m3 = 70  # 70 m3 Tank

reservoirParameters = {
    'maxReservoirVolume': design_capacity_m3,  # 70.0 m3
    'minReservoirVolume': design_capacity_m3 * 0.1,  # 7.0 m3
    'initialReservoirVolume': design_capacity_m3 * 0.5,
    'reseivorTerminal': 1.0, # the reseivor should ideally be 100% full set to [0-1]

}

# PV and Pump System Parameters
staticHeadForSizing = 15.0
waterDensity = 1000.0
gravitationalAcceleration = 9.81
systemEfficiencyForSizing = 0.3
peakSunHours = 4.0
sizingSafetyFactor = 1.5
pvPeakPowerWatts = (
                           waterDensity * gravitationalAcceleration * staticHeadForSizing * sizingSafetyFactor * design_capacity_m3) / (
                           3600 * systemEfficiencyForSizing * peakSunHours)
pvPeakPowerWatts = numpy.ceil(pvPeakPowerWatts / 100) * 100

pvPumpParameters = {
    'pvPeakPower': pvPeakPowerWatts,
    'motorPumpEfficiency': 0.36,
    'pvLossCoefficient': 0.25,
    'referenceIrradiance': 1000.0,
    'waterDensity': waterDensity,
    'gravitationalAcceleration': gravitationalAcceleration,
    'start_up_ratio': 0.2, # <--- Pump won't start below 0.2 of PeakPower
    'minPumpPower': 1500.0,  # <--- PlaceHolder
}


# MPC and Simulation Parameters
simulationParameters = {
    'timeStepSeconds': 3600,
    'simulationDays': 18,  # Enough to show the full 18-day cycle + overlap, put 24
    'predictionHorizonHours': 72,
    'demandBlockDays': 6,  # The "Planner" runs every 6 days
}
simulationParameters['totalSimulationSteps'] = simulationParameters['simulationDays'] * 24

# --- WEIGHTS ---
mpcWeights = {
    'deficitWeight': 3.0,
    'drawdownWeight': 1,
    'pumpStopStartWeight': 1,
}


# ==========================================
# 2. FUNCTIONS
# ==========================================

def fetchRealWeatherProfile(numberOfSteps):
    """
    Fetches REAL TMY weather for Nairobi.
    Returns ONLY Irradiance.
    """
    print("1. Fetching Real Weather Data (Nairobi, Kenya)...")
    try:
        # Fetch TMY data
        result = pvlib.iotools.get_pvgis_tmy(
            latitude=-1.29, longitude=36.82, outputformat='json',
            startyear=2005, endyear=2016
        )
        data = result[0]
        print("   Data fetched successfully!")
    except Exception as e:
        print(f"   Error fetching data: {e}. Returning zeros.")
        return numpy.zeros(numberOfSteps)

    # Extract GHI
    try:
        ghi_full_year = data['ghi']
    except KeyError:
        ghi_full_year = data['G(h)']

    print("2. Processing 'Accelerated Year'...")
    # Take continuous block of data for smoothness
    full_irradiance = ghi_full_year.iloc[:numberOfSteps].values

    # Pad if data is too short
    if len(full_irradiance) < numberOfSteps:
        full_irradiance = numpy.pad(full_irradiance, (0, numberOfSteps - len(full_irradiance)))

    return full_irradiance


def solvePumpFlowRate(electricalPower):
    # ---  Minimum Power Threshold ---
    # If power is below the limit (or negative), pump is OFF.
    min_power = pvPumpParameters.get('minPumpPower', 0.0)
    if electricalPower < min_power:
        return 0, 0
    # 2. Coefficients for Cubic Equation (Eq 15 in Meunier et al. )
    # A * Q^3 + B * Q^2 + C * Q + D = 0

    # Coeff A: Friction Losses (Borehole + Pipe)
    # Represents (beta + nu*L + K)
    if electricalPower <= 0: return 0, 0
    cubicCoeffA = boreholeParameters['boreholeLossCoefficient'] + pipeParameters['linearPipeLossCoefficient'] * \
                  pipeParameters['pipeLength'] + pipeParameters['junctionPipeLossCoefficient']

    # Coeff B: Aquifer Drawdown Resistance (Thiem Equation)
    # Represents ln(rc/rb) / (2 * pi * T)
    cubicCoeffB = numpy.log(boreholeParameters['coneOfDepressionRadius'] / boreholeParameters['boreholeRadius']) / (
            2 * numpy.pi * boreholeParameters['aquiferTransmissivity'])

    # Coeff C: Static Lift
    cubicCoeffC = boreholeParameters['staticWaterDepth']

    # Coeff D: Hydraulic Power
    cubicCoeffD = - (electricalPower * pvPumpParameters['motorPumpEfficiency']) / (
            pvPumpParameters['waterDensity'] * pvPumpParameters['gravitationalAcceleration'])

    # 3. Solve for Flow (Q) in m3/s
    roots = numpy.roots([cubicCoeffA, cubicCoeffB, cubicCoeffC, cubicCoeffD])

    # Filter for real, positive roots (The "physically feasible solution"
    realPositiveRoots = [r.real for r in roots if numpy.isreal(r) and r.real > 0]

    # 4. Calculate Drawdown Components
    if not realPositiveRoots: return 0, 0
    flowRate = min(realPositiveRoots)

    aquiferDrawdown = (cubicCoeffB) * flowRate
    boreholeDrawdown = boreholeParameters['boreholeLossCoefficient'] * flowRate ** 2
    totalDrawdown = aquiferDrawdown + boreholeDrawdown

    if (boreholeParameters['staticWaterDepth'] + totalDrawdown) >= boreholeParameters['motorPumpDepth']:
        return 0, 0

    return flowRate, totalDrawdown


# ==========================================
# 3. LEVEL 1: HIGH-LEVEL PLANNER
# ==========================================

def optimizeBlockDemand(solar_forecast_chunk, total_block_target):
    """
    Distributes a FIXED total volume (e.g., 300m3) across 6 days
    proportionally to the available solar energy.
    """
    num_days = simulationParameters['demandBlockDays']  # 6

    # Calculate "Solar Score" per day
    daily_solar_scores = []
    for d in range(num_days):
        day_start = d * 24
        # Sum of irradiance for this day (Indices 0-24 inside the chunk)
        # Handle cases where chunk might be smaller at end of sim
        if day_start < len(solar_forecast_chunk):
            day_slice = solar_forecast_chunk[day_start: day_start + 24]
            day_score = numpy.sum(day_slice)
        else:
            day_score = 0
        daily_solar_scores.append(day_score)

    daily_solar_scores = numpy.array(daily_solar_scores)
    total_solar_score = numpy.sum(daily_solar_scores)

    # Avoid division by zero
    if total_solar_score == 0:
        return [total_block_target / num_days] * num_days

    # Proportional Allocation: More Sun = More Work
    allocations = (daily_solar_scores / total_solar_score) * total_block_target

    return allocations


# ==========================================
# 4. LEVEL 2: MPC COST FUNCTION
# ==========================================

def calculateMpcObjectiveCost(controlPlan, currentReservoirVolume, irradianceForecast, demandForecast,
                              previousControlAction):
    deficitCost = 0;
    drawdownCost = 0;
    pumpStopStartCost = 0
    simulatedReservoirVolume = numpy.zeros(simulationParameters['predictionHorizonHours'] + 1)
    simulatedReservoirVolume[0] = currentReservoirVolume

    # 1. RETRIEVE THE THRESHOLD (The MPC needs to know this!)
    min_power_threshold = pvPumpParameters.get('minPumpPower', 0.0)

    for hour in range(simulationParameters['predictionHorizonHours']):
        # Calculate Potential Power from Sun
        raw_avail_power = irradianceForecast[hour] / pvPumpParameters['referenceIrradiance'] * \
                          pvPumpParameters['pvPeakPower'] * \
                          (1 - pvPumpParameters['pvLossCoefficient'])

        # --- NEW: APPLY CUTOFF LOGIC TO FORECAST ---
        # If the predicted sun is too weak to start the pump, the MPC should assume 0 Power.
        if raw_avail_power < min_power_threshold:
            availPower = 0.0
        else:
            availPower = raw_avail_power
        # -------------------------------------------

        elecPower = availPower * controlPlan[hour]

        # Pass explicit threshold to solver if needed, or rely on dictionary
        flowRate, totalDrawdown = solvePumpFlowRate(elecPower)

        volumeIn = flowRate * simulationParameters['timeStepSeconds']
        volumeDemanded = demandForecast[hour] * simulationParameters['timeStepSeconds']

        volAvail = simulatedReservoirVolume[hour] + volumeIn
        volDrawn = min(volumeDemanded, max(0, volAvail - reservoirParameters['minReservoirVolume']))

        deficitCost += (volumeDemanded - volDrawn) ** 2

        simulatedReservoirVolume[hour + 1] = min(max(volAvail - volDrawn, 0), reservoirParameters['maxReservoirVolume'])
        drawdownCost += totalDrawdown ** 2

        prev = previousControlAction if hour == 0 else controlPlan[hour - 1]
        pumpStopStartCost += (controlPlan[hour] - prev) ** 2

        # --- TERMINAL COST ---
        final_storage = simulatedReservoirVolume[-1]
        safe_buffer = reservoirParameters['maxReservoirVolume'] * reservoirParameters['reseivorTerminal']
        if final_storage < safe_buffer:
            terminal_penalty = (safe_buffer - final_storage) ** 2 * mpcWeights['deficitWeight'] #Reseivor not full deficit
        else:
            terminal_penalty = 0

        return (mpcWeights['deficitWeight'] * deficitCost +
                mpcWeights['drawdownWeight'] * drawdownCost +
                mpcWeights['pumpStopStartWeight'] * pumpStopStartCost +
                terminal_penalty)

    return (mpcWeights['deficitWeight'] * deficitCost +
            mpcWeights['drawdownWeight'] * drawdownCost +
            mpcWeights['pumpStopStartWeight'] * pumpStopStartCost)


# ==========================================
# 5. MAIN SIMULATION LOOP
# ==========================================

def runSimulation(pv_power_watts):
    # ============================================================
    # 1. DYNAMIC SYSTEM SIZING UPDATE
    # ============================================================

    # A. Update the Peak Power (The Battery/Panel Size)
    pvPumpParameters['pvPeakPower'] = float(pv_power_watts)

    # B. Update the Minimum Power Threshold (The "Kick" required)
    #    Rule: We need 10% of rated power to overcome static head & friction
    new_min_threshold = float(pv_power_watts) * pvPumpParameters['start_up_ratio']

    #    Update the global dictionary so the Solver sees it!
    pvPumpParameters['minPumpPower'] = new_min_threshold

    print(f"   -> Starting simulation for {pv_power_watts} Watts...")
    print(f"      [System Check] Min Start Power scaled to: {new_min_threshold:.1f} W")

    printSystemSetup()
    # --- START TIMER ---
    # Timer is for checking time the code takes to run on your PC
    startTime = time.time()

    # 1. Setup Data
    # Calculate exactly how many hours we need
    total_sim_hours = simulationParameters['simulationDays'] * 24
    total_horizon_needed = total_sim_hours + simulationParameters['predictionHorizonHours']

    fullIrradianceProfile = fetchRealWeatherProfile(total_horizon_needed)
    fullDemandProfile = numpy.zeros(total_horizon_needed)

    # ==============================================================================
    # FIX: PRE-FLIGHT PLANNING (Run Planner for ALL blocks before simulation starts)
    # ==============================================================================
    print("\n[Pre-Flight] Generating full 18-day Demand Profile...")

    block_days = simulationParameters['demandBlockDays']  # 6
    block_hours = block_days * 24
    cycle_targets = [300.0, 100.0, 400.0]  # The targets for each block

    # Loop through all 3 blocks (Days 0-6, 6-12, 12-18)
    for block_idx in range(len(cycle_targets)):
        # 1. Determine timing
        start_hour = block_idx * block_hours
        end_hour = start_hour + block_hours

        # 2. Slice the weather for this specific block
        solar_chunk = fullIrradianceProfile[start_hour: end_hour]

        # 3. Optimize Demand for this block
        target_vol = cycle_targets[block_idx]
        daily_demands = optimizeBlockDemand(solar_chunk, target_vol)

        print(f"   -> Block {block_idx + 1} (Days {block_idx * 6}-{(block_idx + 1) * 6}): "
              f"Target {target_vol}m3 -> Schedule: {[round(x, 1) for x in daily_demands]}")

        # 4. Write to the master Demand Profile
        irrigation_duration = 9  # Hours per day (13:00 - 22:00)

        for day_i in range(block_days):
            # Calculate hourly flow rate for this specific day
            day_total_vol = daily_demands[day_i]
            hourly_rate = (day_total_vol / irrigation_duration) / 3600.0

            # Find the exact hours in the master array
            day_start_hour = start_hour + (day_i * 24)

            # Apply profile (1pm to 10pm)
            for h in range(24):
                global_idx = day_start_hour + h
                if global_idx < len(fullDemandProfile):
                    if 13 <= h < 22:
                        fullDemandProfile[global_idx] = hourly_rate

    print("[Pre-Flight] Demand Generation Complete.\n")
    print("=" * 50)
    # ==============================================================================

    # History Arrays
    reservoirVolumeHistory = numpy.zeros(simulationParameters['totalSimulationSteps'] + 1)
    reservoirVolumeHistory[0] = reservoirParameters['initialReservoirVolume']
    controlActionHistory = numpy.zeros(simulationParameters['totalSimulationSteps'])
    pumpFlowRateHistory = numpy.zeros(simulationParameters['totalSimulationSteps'])
    drawdownHistory = numpy.zeros(simulationParameters['totalSimulationSteps'])
    availablePowerHistory = numpy.zeros(simulationParameters['totalSimulationSteps'])
    deficitHistory = numpy.zeros(simulationParameters['totalSimulationSteps'])
    volumeDrawnHistory = numpy.zeros(simulationParameters['totalSimulationSteps'])
    usedPowerHistory = numpy.zeros(simulationParameters['totalSimulationSteps'])
    isPumpRunningHistory = numpy.zeros(simulationParameters['totalSimulationSteps'])

    initialControlPlanGuess = numpy.zeros(simulationParameters['predictionHorizonHours'])
    previousControlAction = 0.0

    # Stats Trackers
    daily_pumped = 0;
    daily_demand = 0;
    daily_deficit = 0;
    daily_solar = 0

    for currentHour in range(simulationParameters['totalSimulationSteps']):

        # --- LEVEL 2: MPC CONTROLLER ---
        currentReservoirVolume = reservoirVolumeHistory[currentHour]
        irradianceForecast = fullIrradianceProfile[
            currentHour: currentHour + simulationParameters['predictionHorizonHours']]
        demandForecast = fullDemandProfile[currentHour: currentHour + simulationParameters['predictionHorizonHours']]

        # 1. Calculate Available Power for the whole horizon first
        available_power_forecast = (irradianceForecast / pvPumpParameters['referenceIrradiance']) * \
                                   pvPumpParameters['pvPeakPower'] * \
                                   (1 - pvPumpParameters['pvLossCoefficient'])

        # ...

        # 1. Determine the Threshold for THIS specific moment
        #    Real pumps use Hysteresis to prevent chattering.
        #    If we are OFF, we need MORE power (700W) to overcome friction and start.
        #    If we are ON, we can keep running down to the minimum (500W).

        # 1. Determine the Threshold for THIS specific moment
        stop_limit_watts = pvPumpParameters['minPumpPower']  # This is now dynamic (e.g. 360W)

        # Scale the buffer too! (e.g. 5% of peak power extra to wake up)
        hysteresis_buffer = pvPumpParameters['pvPeakPower'] * 0.05
        start_limit_watts = stop_limit_watts + hysteresis_buffer

        # Check previous state (threshold depends on history)
        if previousControlAction > 0.0:
            current_threshold = stop_limit_watts
        else:
            current_threshold = start_limit_watts

        # 2. Create Dynamic Bounds for the horizon
        dynamic_bounds = []

        for i, power_W in enumerate(available_power_forecast):
            # MPC Lookahead Logic:
            # For the very first hour (now), we use the smart Hysteresis threshold.
            # For future hours (prediction), we assume the conservative 'Start Limit'
            # to ensure the plan is robust.

            limit_to_use = current_threshold if i == 0 else start_limit_watts

            if power_W < limit_to_use:
                # Sun too weak -> Force OFF
                dynamic_bounds.append((0.0, 0.0))
            else:
                # Sun strong enough -> Active
                # Use the physical minimum (500W) for the ratio calculation
                # to allow the solver to throttle down if needed.
                min_fraction = (stop_limit_watts / power_W) * 1.01
                if min_fraction > 1.0: min_fraction = 1.0

                dynamic_bounds.append((min_fraction, 1.0))

        # 3. Pass these bounds to the solver

        # --- CRITICAL FIX: CLAMP INITIAL GUESS ---
        # The solver might crash if our "Guess" (0.0 from last night)
        # is outside the new "Bounds" (e.g., 0.2 to 1.0).
        # We must force the guess to be inside the valid range first.

        for i, (low, high) in enumerate(dynamic_bounds):
            # i is the hour index (0 to 72)
            if initialControlPlanGuess[i] < low:
                initialControlPlanGuess[i] = low  # Bump up 0.0 -> 0.2
            elif initialControlPlanGuess[i] > high:
                initialControlPlanGuess[i] = high
                # -----------------------------------------
        result = minimize(
            calculateMpcObjectiveCost,
            initialControlPlanGuess,  #
            args=(currentReservoirVolume, irradianceForecast, demandForecast, previousControlAction),
            method='SLSQP',
            bounds=dynamic_bounds  # <--- USE NEW BOUNDS
        )

        action = result.x[0]
        initialControlPlanGuess = numpy.roll(result.x, -1);
        initialControlPlanGuess[-1] = result.x[-1]

        # Physics
        availPower = fullIrradianceProfile[currentHour] / pvPumpParameters['referenceIrradiance'] * pvPumpParameters[
            'pvPeakPower'] * (1 - pvPumpParameters['pvLossCoefficient'])
        elecPower = availPower * action
        flowRate, totalDrawdown = solvePumpFlowRate(elecPower)

        # ==========================================================
        # CLEANUP: FORCE STOP IF PUMP STALLS
        # ==========================================================


        if flowRate == 0:
            elecPower = 0.0
            # ==========================================================

        # Balance
        volIn = flowRate * simulationParameters['timeStepSeconds']
        volDemanded = fullDemandProfile[currentHour] * simulationParameters['timeStepSeconds']
        volAvail = reservoirVolumeHistory[currentHour] + volIn
        volDrawn = min(volDemanded, max(0, volAvail - reservoirParameters['minReservoirVolume']))

        # Updates
        reservoirVolumeHistory[currentHour + 1] = min(max(volAvail - volDrawn, 0),
                                                      reservoirParameters['maxReservoirVolume'])
        controlActionHistory[currentHour] = action
        pumpFlowRateHistory[currentHour] = flowRate * 3600
        drawdownHistory[currentHour] = totalDrawdown
        availablePowerHistory[currentHour] = availPower / 1000
        usedPowerHistory[currentHour] = elecPower / 1000
        deficitHistory[currentHour] = (volDemanded - volDrawn)
        volumeDrawnHistory[currentHour] = volDrawn
        isPumpRunningHistory[currentHour] = 1 if flowRate > 0 else 0
        previousControlAction = action

        # ==============================================================================
        # DISPLAY
        # ==============================================================================

        # 1. Re-calculate bounds for display
        debug_min_power = pvPumpParameters['minPumpPower']
        if availPower < debug_min_power:
            bound_str = "[0.00 - 0.00] (FORCED OFF)"
        else:
            lower_b = debug_min_power / availPower
            bound_str = f"[{lower_b:.2f} - 1.00] (ACTIVE)"

        # 2. Format Key Metrics
        d_day = currentHour // 24
        d_hour = currentHour % 24

        sun_val = fullIrradianceProfile[currentHour]
        cmd_pct = action * 100.0
        pwr_used = elecPower

        flow_m3h = flowRate * 3600.0
        dd_val = totalDrawdown  # <--- NEW: Get the drawdown value

        dem_m3h = fullDemandProfile[currentHour] * 3600.0
        sup_m3h = (volDrawn / simulationParameters['timeStepSeconds']) * 3600.0
        def_m3h = dem_m3h - sup_m3h

        res_current = reservoirVolumeHistory[currentHour]
        res_next = reservoirVolumeHistory[currentHour + 1]

        # 3. PRINT REPORT
        #    Added 'DD' column for Drawdown in meters
        print(f"D{d_day:02d} H{d_hour:02d} | "
              f"Sun:{sun_val:4.0f} | "
              f"Pwr:{availPower:5.0f}W (Lim:{debug_min_power}W) | "
              f"Bnds:{bound_str:22s} | "
              f"CMD:{cmd_pct:5.1f}% | "
              f"Flow:{flow_m3h:5.1f} m3/h | "
              f"DD:{dd_val:5.2f}m | "  # <--- NEW COLUMN
              f"Res:{res_current:5.1f}->{res_next:5.1f} | "
              f"Dem:{dem_m3h:4.1f} - Sup:{sup_m3h:4.1f} = Def:{def_m3h:4.1f}"
              )
        print("-" * 100)
        # ==============================================================================
        # ==============================================================================

        # Stats
        daily_pumped += volIn;
        daily_demand += volDemanded;
        daily_deficit += (volDemanded - volDrawn);
        daily_solar += fullIrradianceProfile[currentHour]

        # End of Day Report
        if (currentHour + 1) % 24 == 0:
            day_num = (currentHour + 1) // 24
            print(
                f"Day {day_num:02d}: Sun {int(daily_solar / 1000)}k | Demand {daily_demand:.1f} | Pumped {daily_pumped:.1f} | Deficit {daily_deficit:.1f}")
            daily_pumped = 0;
            daily_demand = 0;
            daily_deficit = 0;
            daily_solar = 0

    print("\nSimulation complete.")
    # --- END TIMER ---
    endTime = time.time()
    totalTime = endTime - startTime
    print(f"\nSimulation complete. Total Run Time: {totalTime:.2f} seconds")
    return (reservoirVolumeHistory, controlActionHistory, pumpFlowRateHistory,
            drawdownHistory, availablePowerHistory, fullIrradianceProfile,
            deficitHistory, volumeDrawnHistory, usedPowerHistory, isPumpRunningHistory, fullDemandProfile)


def printSystemSetup():
    print("=" * 50)
    print(" MPC IRRIGATION SIMULATOR")
    print("==================================================")
    print(f"Block Size: {simulationParameters['demandBlockDays']} Days")
    print(f"Targets: 300 -> 100 -> 400 m3 (Cycling)")


# ==========================================
# ==========================================
# VISUALIZATION SUITE
# ==========================================

def plotComparisonCurves(agg_results):
    """
    Plots:
    1. Deficit vs Power (RAW DATA ONLY, NO CURVE FIT)
    2. Total Yield vs Power
    3. Reliability vs Power
    """
    if not os.path.exists("simulation_results/comparative_analysis"):
        os.makedirs("simulation_results/comparative_analysis")

    powers = numpy.array([r['power'] for r in agg_results])
    total_water = numpy.array([r['total_water'] for r in agg_results])
    total_deficit = numpy.array([r['total_deficit'] for r in agg_results])
    total_demand = numpy.array([r['total_demand'] for r in agg_results])

    # --- 1. DEFICIT VS POWER (RAW ONLY) ---
    fig, ax = matplotlibPlotter.subplots(figsize=(12, 7))
    ax.set_title('OPTIMIZATION: Deficit vs PV Array Size', fontsize=18, fontweight='bold')

    # Plot raw data only
    ax.plot(powers, total_deficit, 'ro-', markersize=10, linewidth=2, label='Simulation Data')

    # --- NEW: Force ticks on X-Axis ---
    # --- UPDATED: High Visibility Ticks ---
    ax.set_xticks(powers)
    ax.set_xticklabels(powers, fontsize=7, fontweight='bold', rotation=90, ha='right')
    ax.tick_params(axis='x', which='major', pad=5)
    # --------------------------------------
    # ----------------------------------

    ax.set_xlabel('PV Array Size (Watts)', fontsize=14)
    ax.set_ylabel('Total Deficit ($m^3$)', fontsize=14, color='red')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=12)

    fig.savefig("simulation_results/comparative_analysis/1_Deficit_Analysis.png", dpi=300)
    matplotlibPlotter.close(fig)

    # --- 2. YIELD & RELIABILITY ---
    fig, ax1 = matplotlibPlotter.subplots(figsize=(12, 7))
    ax1.set_title('SYSTEM SIZING: Yield & Reliability', fontsize=18, fontweight='bold')

    color = 'tab:blue'
    ax1.set_xlabel('PV Array Size (Watts)', fontsize=14)
    ax1.set_ylabel('Total Water Pumped ($m^3$)', color=color, fontsize=14)
    ax1.plot(powers, total_water, color=color, marker='o', linewidth=3)
    ax1.tick_params(axis='y', labelcolor=color)

    # --- NEW: Force ticks on X-Axis ---
    # --- UPDATED: High Visibility Ticks ---
    ax1.set_xticks(powers)
    ax1.set_xticklabels(powers, fontsize=7, fontweight='bold', rotation=90, ha='right')
    ax1.tick_params(axis='x', which='major', pad=5)
    # --------------------------------------
    # ----------------------------------
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    color = 'tab:green'
    # Calculate Reliability %
    reliability = (1 - (total_deficit / total_demand)) * 100
    reliability[reliability < 0] = 0  # Clamp negative

    ax2.set_ylabel('Reliability (% Met)', color=color, fontsize=14)
    ax2.plot(powers, reliability, color=color, marker='s', linestyle='--', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    fig.savefig("simulation_results/comparative_analysis/2_System_Sizing_Curve.png", dpi=300)
    matplotlibPlotter.close(fig)
    print("   -> Saved Comparative Analysis curves.")


def plotDetailedPhysics(results_tuple, power_val):
    """
    Generates scatter plots for physics validation for a specific scenario.
    """
    # Create specific folder for this power level
    folder = f"simulation_results/details_{int(power_val)}W"
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Unpack tuple
    flow_m3h = results_tuple[2]
    drawdown = results_tuple[3]
    sun = results_tuple[5]
    power_watts = results_tuple[8] * 1000  # Convert kW back to W

    # Align sun array
    sim_steps = len(flow_m3h)
    sun = sun[:sim_steps]

    # Filter for daylight
    daylight_mask = sun > 10
    sun_day = sun[daylight_mask]
    flow_day = flow_m3h[daylight_mask]
    drawdown_day = drawdown[daylight_mask]
    power_day = power_watts[daylight_mask]

    # 1. HYSTERESIS PLOT
    fig, ax = matplotlibPlotter.subplots(figsize=(10, 8))
    scatter = ax.scatter(sun_day, flow_day, c=power_day, cmap='inferno', alpha=0.7)
    ax.set_title(f'PHYSICS: Start-Up Hysteresis ({power_val}W)', fontsize=16)
    ax.set_xlabel('Solar Irradiance ($W/m^2$)', fontsize=14)
    ax.set_ylabel('Water Flow Rate ($m^3/h$)', fontsize=14)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Power Used (W)')
    ax.grid(True, alpha=0.3)
    fig.savefig(f"{folder}/Physics_Flow_vs_Sun.png", dpi=300)
    matplotlibPlotter.close(fig)

    # 2. AQUIFER STRESS PLOT
    fig, ax = matplotlibPlotter.subplots(figsize=(10, 8))
    ax.scatter(power_day, drawdown_day, color='purple', alpha=0.6)
    ax.set_title(f'AQUIFER RESPONSE: Drawdown vs Power ({power_val}W)', fontsize=16)
    ax.set_xlabel('Pump Power (Watts)', fontsize=14)
    ax.set_ylabel('Well Drawdown (m)', fontsize=14)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    fig.savefig(f"{folder}/Physics_Drawdown_vs_Power.png", dpi=300)
    matplotlibPlotter.close(fig)

    print(f"   -> Saved Detailed Physics plots for {power_val}W.")


def saveDailyZoomPlots(results_tuple, fullDemandProfile, filename_prefix):
    """
    Slices the big simulation into 24-hour zoomed plots for each day.
    """
    # 1. Extract Power for folder name
    try:
        pwr_str = filename_prefix.split('_')[1]
        pwr_val = pwr_str.replace('W', '')
        folder = f"simulation_results/daily_zooms/PV_{pwr_val}W"
    except:
        folder = "simulation_results/daily_zooms/UNKNOWN"

    if not os.path.exists(folder):
        os.makedirs(folder)

    print(f"   -> Generatng Daily Zoom plots in: {folder}...")

    # 2. Unpack Data
    res_vol_full = results_tuple[0][:-1]  # trim last
    drawdown_full = results_tuple[3]
    avail_pwr_full = results_tuple[4] * 1000  # to Watts
    irrad_full = results_tuple[5][:len(res_vol_full)]
    used_pwr_full = results_tuple[8] * 1000  # to Watts
    state_full = results_tuple[9]

    # --- CHANGE: Volumes in m3 ---
    demand_vol_full = fullDemandProfile[:len(res_vol_full)] * 3600  # m3/s * 3600s = m3
    supply_vol_full = results_tuple[7]  # Already m3
    # -----------------------------

    total_days = simulationParameters['simulationDays']

    # 3. Loop through days
    for day in range(total_days):
        start_idx = day * 24
        end_idx = start_idx + 24

        # Check bounds
        if end_idx > len(res_vol_full): break

        # Slicing
        hours_x = numpy.arange(0, 24)

        r_slice = res_vol_full[start_idx:end_idx]
        irr_slice = irrad_full[start_idx:end_idx]
        dem_slice = demand_vol_full[start_idx:end_idx]
        state_slice = state_full[start_idx:end_idx]
        dd_slice = drawdown_full[start_idx:end_idx]
        pwr_avail_slice = avail_pwr_full[start_idx:end_idx]
        pwr_used_slice = used_pwr_full[start_idx:end_idx]
        sup_slice = supply_vol_full[start_idx:end_idx]

        # Plotting
        with matplotlibPlotter.style.context('seaborn-v0_8-whitegrid'):
            fig, axes = matplotlibPlotter.subplots(7, 1, figsize=(15, 20), sharex=True)
            fig.suptitle(f'DAY {day + 1:02d} ZOOM: PV {pwr_val}W', fontsize=22, fontweight='bold')

            # Subplot 1: Reservoir
            axes[0].plot(hours_x, r_slice, '#0277bd', linewidth=3)
            axes[0].set_ylabel('Reservoir\n($m^3$)', fontsize=12, fontweight='bold')
            axes[0].set_ylim(0, reservoirParameters['maxReservoirVolume'] * 1.1)

            # Subplot 2: Sun
            axes[1].plot(hours_x, irr_slice, 'orange', linewidth=2, fillstyle='bottom')
            axes[1].fill_between(hours_x, irr_slice, 0, color='orange', alpha=0.3)
            axes[1].set_ylabel('Sun\n($W/m^2$)', fontsize=12, fontweight='bold')

            # Subplot 3: Demand (VOLUME)
            axes[2].plot(hours_x, dem_slice, 'green', linewidth=3)
            axes[2].fill_between(hours_x, dem_slice, 0, color='green', alpha=0.1)
            axes[2].set_ylabel('Target Vol\n($m^3$)', fontsize=12, fontweight='bold')

            # Subplot 4: State
            axes[3].step(hours_x, state_slice, 'k', where='post', linewidth=2)
            axes[3].set_ylabel('Pump\nState', fontsize=12, fontweight='bold')
            axes[3].set_yticks([0, 1])
            axes[3].set_yticklabels(['OFF', 'ON'])

            # Subplot 5: Drawdown
            axes[4].plot(hours_x, dd_slice, 'purple', linewidth=2)
            axes[4].invert_yaxis()
            axes[4].set_ylabel('Drawdown\n(m)', fontsize=12, fontweight='bold')

            # Subplot 6: Power
            axes[5].plot(hours_x, pwr_avail_slice, 'k--', label='Solar Limit')
            axes[5].plot(hours_x, pwr_used_slice, 'r', linewidth=2, label='Used')
            axes[5].fill_between(hours_x, pwr_used_slice, 0, color='red', alpha=0.2)
            axes[5].set_ylabel('Power\n(W)', fontsize=12, fontweight='bold')

            # Subplot 7: Flow (VOLUME)
            axes[6].plot(hours_x, sup_slice, '#0d47a1', linewidth=3, label='Pumped')
            axes[6].plot(hours_x, dem_slice, 'g--', linewidth=2, label='Target')
            axes[6].set_ylabel('Hourly Vol\n($m^3$)', fontsize=12, fontweight='bold')
            axes[6].set_xlabel('Hour of Day (0-24)', fontsize=16)
            axes[6].set_xticks(numpy.arange(0, 25, 2))

            # Save
            filename = f"{folder}/Day_{day + 1:02d}.png"
            fig.tight_layout()
            fig.savefig(filename, dpi=150)  # Lower DPI for speed since there are many
            matplotlibPlotter.close(fig)


def saveResultsAsImage(reservoirVolume, controlAction, pumpFlowRate,
                       drawdown, availablePower, irradiance,
                       deficit, volumeDrawn, usedPower, isPumpRunning,
                       fullDemandProfile, filename_prefix):
    # --- UPDATED: Save into specific folder ---
    try:
        pwr_str = filename_prefix.split('_')[1]
        pwr_val = pwr_str.replace('W', '')
        folder = f"simulation_results/details_{pwr_val}W"
    except:
        folder = "simulation_results"

    if not os.path.exists(folder):
        os.makedirs(folder)

    # --- UPDATED: Save a copy to the "All Timelines" folder ---
    all_timelines_folder = "simulation_results/ALL_SCENARIOS_OVERVIEW"
    if not os.path.exists(all_timelines_folder):
        os.makedirs(all_timelines_folder)

    filename = f"{folder}/{filename_prefix}_Timeline.png"
    common_filename = f"{all_timelines_folder}/{filename_prefix}_Timeline.png"

    print(f"   -> Generating Timeline: {filename}...")

    # 0. SETUP & STYLE
    with matplotlibPlotter.style.context('seaborn-v0_8-whitegrid'):

        timeArray = numpy.arange(simulationParameters['totalSimulationSteps']) / 24.0
        n_steps = len(timeArray)

        # Align arrays
        irr_aligned = irradiance[:n_steps]

        # --- NEW CUMULATIVE LOGIC (MATCHING YOUR SKETCH) ---
        # 1. Prepare arrays for the resetting cumulative data
        cum_demand_plot = numpy.zeros(n_steps)
        cum_supply_plot = numpy.zeros(n_steps)

        # 2. Get block info
        block_days = simulationParameters['demandBlockDays']
        steps_per_block = block_days * 24

        # 3. Calculate Cumulative Sums (Resetting every block)
        num_blocks = int(numpy.ceil(n_steps / steps_per_block))

        hourly_demand_raw = fullDemandProfile[:n_steps] * 3600  # m3
        hourly_supply_raw = volumeDrawn  # m3

        for b in range(num_blocks):
            s_idx = b * steps_per_block
            e_idx = min((b + 1) * steps_per_block, n_steps)

            # Slice this block's data
            block_dem = hourly_demand_raw[s_idx:e_idx]
            block_sup = hourly_supply_raw[s_idx:e_idx]

            # Calculate cumulative sum for this specific block (starts at 0)
            cum_demand_plot[s_idx:e_idx] = numpy.cumsum(block_dem)
            cum_supply_plot[s_idx:e_idx] = numpy.cumsum(block_sup)

        # ----------------------------

        # Create Figure - WIDE CANVAS
        fig, axes = matplotlibPlotter.subplots(7, 1, figsize=(36, 24), sharex=True, constrained_layout=True)

        # --- FONT SIZES (MASSIVE BOOST) ---
        TITLE_SIZE = 28
        AXIS_LABEL_SIZE = 20
        TICK_SIZE = 16
        ANNOTATION_SIZE = 16
        LEGEND_SIZE = 16
        LINE_WIDTH = 3.5

        # Dynamic Title
        clean_title = filename_prefix.replace("_", " ").upper()
        fig.suptitle(f'MPC RESULTS: {clean_title}', fontsize=TITLE_SIZE, fontweight='bold', y=0.99)

        # --- 1. BACKGROUND SHADING ---
        states = numpy.zeros(n_steps)
        states[irr_aligned > 0] = 1  # Day
        states[hourly_demand_raw > 0] = 2  # Irrigation

        color_night = 'black'
        color_day = '#ffff00'
        color_irrig = '#00e676'

        def get_spans(state_arr, target_state):
            is_state = (state_arr == target_state).astype(int)
            diffs = numpy.diff(is_state, prepend=0, append=0)
            starts = numpy.where(diffs == 1)[0]
            ends = numpy.where(diffs == -1)[0]
            return zip(starts, ends)

        for ax in axes:
            for s, e in get_spans(states, 0):
                t_end = timeArray[e] if e < n_steps else timeArray[-1]
                ax.axvspan(timeArray[s], t_end, color=color_night, alpha=0.1, zorder=0)
            for s, e in get_spans(states, 1):
                t_end = timeArray[e] if e < n_steps else timeArray[-1]
                ax.axvspan(timeArray[s], t_end, color=color_day, alpha=0.2, zorder=0)
            for s, e in get_spans(states, 2):
                t_end = timeArray[e] if e < n_steps else timeArray[-1]
                ax.axvspan(timeArray[s], t_end, color=color_irrig, alpha=0.2, zorder=0)

        # --- PLOT 1: RESERVOIR ---
        res_vol = reservoirVolume[:-1]
        min_vol = numpy.min(res_vol)
        max_vol = reservoirParameters['maxReservoirVolume']

        axes[0].plot(timeArray, res_vol, '#0277bd', linewidth=LINE_WIDTH)
        axes[0].axhline(max_vol, color='red', linestyle=':', linewidth=LINE_WIDTH)
        axes[0].axhline(min_vol, color='black', linestyle='--', linewidth=2, alpha=0.8)
        axes[0].text(timeArray[-1], max_vol, ' MAX', verticalalignment='center', fontsize=ANNOTATION_SIZE, color='red',
                     fontweight='bold')
        axes[0].text(timeArray[-1], min_vol, f' MIN: {min_vol:.1f}', verticalalignment='center',
                     fontsize=ANNOTATION_SIZE, color='black', fontweight='bold')
        axes[0].set_ylim(0, max_vol * 1.2)
        axes[0].set_ylabel('Reservoir\n($m^3$)', fontsize=AXIS_LABEL_SIZE, fontweight='bold', labelpad=15)

        # --- PLOT 2: IRRADIANCE ---
        axes[1].plot(timeArray, irr_aligned, '#ef6c00', linewidth=2.5)
        axes[1].fill_between(timeArray, irr_aligned, 0, color='orange', alpha=0.3)
        axes[1].set_ylabel('Sun\n($W/m^2$)', fontsize=AXIS_LABEL_SIZE, fontweight='bold', labelpad=15)

        # --- PLOT 3: DEMAND (Hourly Rate) ---
        axes[2].plot(timeArray, hourly_demand_raw, 'green', linewidth=LINE_WIDTH)
        axes[2].fill_between(timeArray, hourly_demand_raw, 0, color='green', alpha=0.1)
        axes[2].set_ylabel('Target Rate\n($m^3/h$)', fontsize=AXIS_LABEL_SIZE, fontweight='bold', labelpad=15)

        # --- PLOT 4: PUMP STATE ---
        axes[3].step(timeArray, isPumpRunning, 'k', where='post', linewidth=LINE_WIDTH)
        axes[3].fill_between(timeArray, isPumpRunning, step='post', color='#2ca02c', alpha=0.5)
        axes[3].set_ylabel('Pump\nState', fontsize=AXIS_LABEL_SIZE, fontweight='bold', labelpad=15)
        axes[3].set_yticks([0, 1])
        axes[3].set_yticklabels(['OFF', 'ON'], fontsize=TICK_SIZE, fontweight='bold')

        # --- PLOT 5: DRAWDOWN ---
        axes[4].plot(timeArray, drawdown, 'purple', linewidth=LINE_WIDTH)
        axes[4].fill_between(timeArray, drawdown, 0, color='purple', alpha=0.1)
        axes[4].set_ylabel('Drawdown\n(m)', fontsize=AXIS_LABEL_SIZE, fontweight='bold', labelpad=15)
        axes[4].invert_yaxis()

        # --- PLOT 6: POWER ---
        axes[5].plot(timeArray, availablePower, 'black', linestyle='--', linewidth=2, alpha=0.7)
        axes[5].plot(timeArray, usedPower, '#d32f2f', linewidth=LINE_WIDTH)
        axes[5].fill_between(timeArray, usedPower, 0, color='red', alpha=0.2)
        axes[5].text(timeArray[0], numpy.max(availablePower) * 0.9, ' Solar Limit', color='black',
                     fontsize=ANNOTATION_SIZE, fontweight='bold')
        axes[5].set_ylabel('Power\n(W)', fontsize=AXIS_LABEL_SIZE, fontweight='bold', labelpad=15)

        # --- PLOT 7: CUMULATIVE BLOCK PERFORMANCE (THE SKETCH) ---

        # Plot Demand (Target)
        axes[6].plot(timeArray, cum_demand_plot, 'k', linestyle='--', linewidth=3, label='Target (Cumulative)')

        # Plot Supply (Actual)
        axes[6].plot(timeArray, cum_supply_plot, '#0d47a1', linewidth=4, label='Pumped (Cumulative)')
        axes[6].fill_between(timeArray, cum_supply_plot, 0, color='#1976d2', alpha=0.3)

        # Fill Deficit Area (Red Hatching where Supply < Demand)
        # We use a small epsilon for float comparison safety
        deficit_mask = cum_demand_plot > (cum_supply_plot + 0.1)
        if numpy.any(deficit_mask):
            axes[6].fill_between(timeArray, cum_demand_plot, cum_supply_plot, where=deficit_mask,
                                 color='red', alpha=0.3, hatch='X', edgecolor='red', label='Deficit Gap')

            # Add label to the first deficit area
            def_idx = numpy.where(deficit_mask)[0][0]
            # axes[6].text(timeArray[def_idx], cum_demand_plot[def_idx], ' DEFICIT', color='red', fontsize=ANNOTATION_SIZE, fontweight='bold')

        # Add Vertical Block Lines
        for b in range(num_blocks + 1):
            day_boundary = b * block_days
            axes[6].axvline(day_boundary, color='black', linewidth=2)
            if b < num_blocks:
                mid_point = day_boundary + (block_days / 2)
                # Label the block (e.g., "BLOCK 1")
                axes[6].text(mid_point, numpy.max(cum_demand_plot) * 0.9, f'BLOCK {b + 1}',
                             ha='center', fontsize=ANNOTATION_SIZE, fontweight='bold',
                             bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

        axes[6].set_ylabel('Block Vol.\n($m^3$)', fontsize=AXIS_LABEL_SIZE, fontweight='bold', labelpad=15)
        axes[6].set_xlabel('Simulation Time (Days)', fontsize=22, fontweight='bold', labelpad=15)
        axes[6].legend(loc='upper left', fontsize=LEGEND_SIZE, frameon=True)

        # --- TICKS FORMATTING ---
        total_days = int(numpy.ceil(timeArray[-1]))
        daily_ticks = numpy.arange(0, total_days + 1, 1)
        for ax in axes:
            ax.set_xlim([0, timeArray[-1]])
            ax.set_xticks(daily_ticks)
            ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE, pad=8)
            ax.grid(True, which='both', linestyle='-', linewidth=1, color='gray', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # Make left/bottom spines thicker
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)

        # Bold X-Axis Ticks
        for label in axes[-1].get_xticklabels():
            label.set_fontweight('bold')

        # SAVE (Two copies)
        fig.savefig(filename, dpi=350)
        fig.savefig(common_filename, dpi=350)  # Save copy to common folder

        matplotlibPlotter.close(fig)
        print(f"   -> Saved to {filename}")
        print(f"   -> Copy saved to {common_filename}")
if __name__ == '__main__':
    # 1. Define Scenarios
    pv_scenarios = [1500,2000,3000,5000,10000,20000,50000]
    #pv_scenarios = [3000]
    # Store aggregate data for the comparison plots
    aggregate_data = []

    print("==================================================")
    print(" BATCH SIMULATION STARTED: EXHAUSTIVE ANALYSIS MODE")
    print("==================================================")

    for power in pv_scenarios:
        # A. Run the simulation for this specific power level
        results = runSimulation(power)

        # B. Calculate simple stats for the summary table
        total_deficit = numpy.sum(results[6])
        min_res = numpy.min(results[0])
        total_water_pumped = numpy.sum(results[2])  # Summing hourly flow
        total_demand = numpy.sum(results[10]) * 3600  # Convert m3/s to m3/h then sum = total m3

        # Save to aggregate list
        aggregate_data.append({
            'power': power,
            'total_deficit': total_deficit,
            'total_water': total_water_pumped,
            'total_demand': total_demand
        })

        # C. Save the plot using the specific results from this run
        saveResultsAsImage(*results, filename_prefix=f"PV_{power}W")

        # D.Save Detailed Physics Plots
        plotDetailedPhysics(results, power)

        # E.Save 24h Zooms for every day
        saveDailyZoomPlots(results, results[10], f"PV_{power}W")

    # 3. Print Summary Table
    print("\n==================================================")
    print(f"{'PV Size (W)':<15} | {'Total Deficit':<15} | {'Min Reservoir'}")
    print("-" * 50)
    for i, res in enumerate(aggregate_data):
        print(f"{res['power']:<15} | {res['total_deficit']:<15.2f} | N/A")
    print("==================================================")

    # 4. Plot Comparison Curves
    print("\n[Post-Processing] Generating Comparative Sizing Curves...")
    plotComparisonCurves(aggregate_data)

    print("DONE. Check 'simulation_results' folder.")
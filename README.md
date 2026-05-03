# Insurgency Dynamics Model

A mathematical model for insurgency-counterinsurgency dynamics with an interactive GUI.

## Overview

This project implements a system of ordinary differential equations modeling the dynamics between civilian populations (C), insurgents (I), and counterinsurgency forces (F). The model includes key parameters for recruitment, mortality, and policy effects.

## Model Equations

```
dC/dt = r*C*(1 - C/K) - α*I*C - β*F*C + γ*F*C
dI/dt = α*I*C - μ_I*I - δ*F*I
dF/dt = φ - μ_F*F + σ*I
```

Where:
- **C**: Civilian population
- **I**: Insurgent population
- **F**: Counterinsurgency (COIN) forces

## Key Parameters

- **α**: Insurgent recruitment rate
- **β**: COIN collateral damage rate
- **γ**: Hearts-and-minds legitimacy rate
- **R₀**: Basic reproduction number = αC*/(μ_I + δF*)

## GUI Application

The project includes a minimalist web-based GUI styled after Palantir's dark green and black theme.

### Running the GUI

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   streamlit run gui.py
   ```

3. Open your browser to `http://localhost:8501`

### Features

- Interactive parameter adjustment
- Real-time simulation
- R₀ threshold analysis
- Population dynamics visualization
- Dark theme with green accents

## Files

- `insurgency_model.py`: Core model implementation
- `visualize_summary.py`: Static visualization script
- `gui.py`: Interactive web GUI
- `requirements.txt`: Python dependencies

## Usage

Adjust parameters in the sidebar to explore different scenarios:
- Low R₀ (< 1): Insurgency collapses
- High R₀ (> 1): Endemic insurgency persists

The GUI provides immediate feedback on population dynamics and stability.
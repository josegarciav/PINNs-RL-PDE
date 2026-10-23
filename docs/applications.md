# Applications: digital twins

A **digital twin** is a model of a specific physical asset that is kept in sync with that asset through its sensor data. It is used to estimate what can't be measured, to forecast, and to test decisions before applying them. Most industrial twins today are either detailed simulators that are too slow or too rigid to calibrate continuously, or data-driven models that don't respect physics and extrapolate poorly.

PINNs sit between the two. They combine the conservation laws you already know with the sensor streams you already have, and they solve the inverse problem of learning unknown parameters, closures and hidden states as part of training. That is the gap this project targets.

## Where a PINN fits

The pitch is the same in every row: *your plant already logs X. A physics-informed model can turn X into an estimate of Y, which you can't measure continuously, and keep that estimate calibrated as the equipment ages.*

| Industry | Typical twin | Physics (PDEs) | Where a PINN helps |
|---|---|---|---|
| **Food and dairy** | Spray dryers, evaporators, pasteurizers and UHT lines, fermenters | Heat and mass transfer, drying kinetics, fouling growth | Soft sensors for residual moisture and fouling resistance from routine temperatures and flows; predicting when cleaning-in-place is due |
| **Chemicals and petrochemicals** | Reactors, distillation columns, heat exchangers | Reaction–diffusion–convection, heat transfer | Identifying kinetic parameters; estimating hot spots and catalyst deactivation from sparse thermocouples |
| **Pharma and biopharma** | Freeze dryers, bioreactors, continuous lines | Heat and mass transfer, moving sublimation fronts | Product temperature and drying front in freeze-drying from shelf temperature and chamber pressure (process analytical technology) |
| **Pulp, paper and textiles** | Drying sections, coating lines | Coupled heat and moisture transport | Moisture profiles across the web from a few scanner readings |
| **Metals, mining and materials** | Furnaces, casting, heat treatment, additive manufacturing | Heat conduction with phase change (Stefan problems), thermal stress | Solidification front and refractory wear from shell temperatures; melt-pool calibration |
| **Power and energy** | Boilers, turbines, nuclear thermal hydraulics, wind farms | Heat transfer, fluid flow, structural dynamics | Heat-flux and remaining-life estimation; fast wake surrogates for layout and control |
| **Batteries and storage** | EV packs, grid batteries | Electrochemistry (single-particle, Doyle–Fuller–Newman models), heat | State of health and parameter identification from voltage and temperature curves |
| **District heating and HVAC** | Pipe networks, building energy | Transport (advection) in pipes, heat in walls | Hidden temperatures and leaks from supply and return sensors |
| **Oil, gas and subsurface** | Reservoirs, pipelines, geothermal | Porous-media (Darcy) flow, multiphase pipe flow | History matching (permeability inversion); leak localization |
| **Aerospace and defense** | Engines, airframes, thermal protection | Fluid flow, structural mechanics, heat | Reconstructing full fields from sparse sensors; identifying loads and damage for structural health monitoring |
| **Automotive** | Thermal management, battery packs, aerodynamics | Heat, fluid flow | Calibrated thermal models; aerodynamic surrogates for design |
| **Semiconductors** | CVD and epitaxy reactors, rapid thermal processing | Reaction–diffusion, heat | Wafer temperature maps from a few pyrometers; recipe calibration |
| **Built environment and infrastructure** | Buildings, bridges, tunnels, concrete curing | Heat and moisture in porous media, structural dynamics | Concrete maturity and strength estimation; structural health monitoring |
| **Water and environment** | Treatment plants, rivers, sewers, groundwater | Advection–dispersion–reaction, shallow water | Identifying contaminant sources; calibrated flood and dispersion models |
| **Healthcare** | Cardiac, vascular and tumor-growth models | Reaction–diffusion, blood flow | Patient-specific tissue parameters (conductivity, stiffness) from imaging |
| **Agriculture** | Soil, grain storage, greenhouses | Richards equation (soil water), heat and moisture | Soil hydraulic parameters from probes; hot spots in grain silos |

**Out of scope:** twins without field physics, such as supply-chain, logistics, IT-network and discrete-event production-scheduling twins. Graph, queueing or discrete-event models serve those better than PDEs.

## What a PINN-based twin is, and isn't

- It is an **estimation layer**. It sits above the control system (PLC/DCS), reads the historian or OPC UA, and returns estimates with uncertainty and advisory setpoints. It does not replace real-time control loops.
- It is **not a faster CFD**. When the physics is fully known and all you need is a forward simulation, use a classical solver.
- It earns its keep where something important is **unmeasured or unknown**: a fouling layer, a drying-kinetics curve, a permeability field, a tissue property. Credible deployment also needs a classical baseline (Kalman filters, moving-horizon estimation) for comparison, and calibrated uncertainty.

## First target: food-powder lines

The first concrete application is the dairy and food-powder line: evaporators feeding spray dryers, as in cheese- and milk-powder plants. The plan:

1. falling-film evaporator fouling, estimated from effect-level temperatures and flows;
2. a spray-dryer soft sensor for residual moisture, with a stickiness-margin advisory;
3. a coupled line-level twin with measurement design (deciding when to pull a lab sample).

This is the [research program](research.md); its experiments live in `experiments/` (track J of the backlog).

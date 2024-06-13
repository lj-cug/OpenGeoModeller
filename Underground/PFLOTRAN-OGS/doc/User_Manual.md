# User Manual
[pflotran_ogs¥˙¬Î≤÷ø‚¡¥Ω”](https://bitbucket.org/opengosim/pflotran_ogs_1.8/src/pflotran_ogs_1.8/)

[User Manual¡¥Ω”](https://docs.opengosim.com/)

This User Manual provides instructions on how to use the PFLOTRAN-OGS flow models:

Gas-Water for CO2 Storage in saline aquifer (GAS_WATER).

Three-component Multi-Gas for CO2 Storage in depleted gas fields (COMP3).

Four-component Multi-Gas for CO2 Storage in depleted hydrocarbon fields in cases of small quantities of residual oil (COMP4).

Multi-component for CO2 Storage in more complex depleted hydrocarbon fields (COMP).

Black-Oil (BLACK_OIL), Todd-Longstaff (TODD_LONGSTAFF), Solvent (SOLVENT_TL) and Oil-Water (TOIL_IMS), for other hydrocarbon applications.

GAS_WATER, BLACK_OIL, TODD_LONGSTAFF, SOLVENT_TL, COMP3 and COMP4 can run both in isothermal and thermal mode. At present, COMP can account for time-constant temperature-gradient effects, but run only in isothermal mode.

```
Introduction

Multi Component Mode (COMP)

Multigas Component Mode (Oil/Gas/Solvent/Water) (COMP4)

Multigas Component Mode (Gas/Solvent/Water) (COMP3)

Gas-Water (GAS_WATER)

Black Oil Model (BLACK_OIL)

The Todd-Longstaff Model (TODD_LONGSTAFF)

Solvent Model (SOLVENT_TL)

Oil-Water Model (TOIL)

Thermal Option

Structure of this Manual

Multi Component Mode (COMP)
The multi component mode (COMP) also describes a flow model flow model consiting of water and any number of additional compnents.

Multigas Component Mode (Oil/Gas/Solvent/Water) (COMP4)
Describes a flow model with OIL, GAS, SOLVENT and WATER. Note that SOLVENT will often be CO2, but this is not a requirement. The SOLVENT and GAS components share the gas phase, both may dissolve in the oil, and the solvent component may dissolve in the aqueous phase.

Multigas Component Mode (Gas/Solvent/Water) (COMP3)
Describes a flow model with, GAS, SOLVENT and WATER. Note that SOLVENT will often be CO2, but this is not a requirement. The SOLVENT and GAS components share the reservoir vapour phase, and the solvent component may dissolve in the aqueous phase. This is a generalisation of the GAS_WATER mode to handle carbon storage in depleted gas reservoirs in which both original reservoir gas (GAS component) and injected CO2 (SOLVENT) are present.

Similar to COMP4 but with no oil data to specify. The SOLVENT data can be set with a Span Wagner CO2 properties database as in GAS_WATER mode. The gas properties can be set with a PVDG properties table or a external database.

In the COMP3 and COMP4, pressure and temperature dependent K-values are used to determine the split of components between phases, and then mixing rules are applied to determine phase properties such as molar density and viscosity. For the molar densities, the molar volumes are interpolated and then the reciprocal of the molar volume is taken to construct a mixture molar density. This additive treatment of molar volumes is known as Amagat°Øs Law.

Gas-Water (GAS_WATER)
The Gas-water module (GAS_WATER) describes a two-phase flow model. In this case each phase may contain both components: the aqueous phase may contain dissolved gas, and the gas phase may contains vapourised water. The module solves a molar balance equation for each component and an energy equation to account for thermal effects. For more details see the Mathematical formulation of Gas-Water in the theory guide.

Black Oil Model (BLACK_OIL)
The Black Oil Model is a common fluid treatment in reservoir simulation. In this treatment, gas is allowed to dissolve in the reservoir oil phase. The amount of dissolved gas in the oil is parameterized by the oil saturation pressure, commonly known as the bubble point. Oil properties such as the formation volume factor (volume of oil at reservoir conditions/volume of oil at surface conditions), enthalpy and viscosity will generally be functions of pressure and bubble point. See Mathematical formulation of the Black Oil Model in the theory guide.

The Todd-Longstaff Model (TODD_LONGSTAFF)
The simple Todd-Longstaff model is suitable for cases in which the reservoir is undersaturated with no free gas. In this case the solvent is represented by the gas phase. See Mathematical formulation of the simple Todd-Longstaff Model in the theory guide.

Solvent Model (SOLVENT_TL)
The Solvent model is an extension of black oil model which adds an additional solvent phase, commonly used to represent an injected fluid such as carbon dioxide which is fully or partially miscible with the oil. See Mathematical formulation of the Solvent Model in the theory guide.

Oil-Water Model (TOIL)
The Thermal Oil Immiscible (TOIL_IMS) module describes a two-phase flow model, in which the oil and water phases are considered immiscible. The module solves a molar balance equation for each phase and an energy equation to account for thermal effects. Each phase is assumed to contain only one component: the water phase contains 100% H2O, the oil CO2 phase contains 100% of a user-defined oil component. For more details see the Mathematical formulation of toil-ims in the theory guide.

Thermal Option
In the above flow models, if the ISOTHERMAL mode has not been selected, temperature variation will be modeled for each cell. Within a single cell, the fluids and rock are assumed to be in thermal equilibrium, and to share the same cell temperature. In isothermal mode the energy equation is not solved and the temperature is assumed not to vary with time. The thermal option is not currently available for COMP.

Structure of this Manual
This manual gives instructions on how to: (i) build an input file, (ii) monitor the convergence and evolution of a simulation, (iii) view the resulting PFLOTRAN output files.
```
# 粘性沙河床演变模型2（CSTM-葛建忠）

Referring cohesive sediment module(Delft3D-Flow User manual, 2014), the
theory about sand-mud transport and interaction had been integrated into
CSTM sediment model by Dr. Ge Jianzhong of China East Normal University.

The cohesive sediment module in ROMS model is in development (Contact
Chiris Sherwood)

The sediment calculation theory was adopted from the user manual of
Delft3d software as following:

## 1. General Formulations

For schematization we distinguish "mud" (cohesive suspended load
transport), "sand" (non-cohesive bedload and suspended load transport)
and "bedload" (non-cohesive bedload only or total load transport)
fractions.

Sediment interactions are taken into account although especially in the
field of sand-mud interactions still a couple of processes are lacking.

### 1.1 Suspended sediment 

Follow the advection-diffusion equation (mass balance):

![](./media/image1.emf){width="5.281615266841645in"
height="1.08043416447944in"}

### 1.2 Effect of sediment on fluid density

By default Delft3D-Flow uses the empirical formula by UNESCO (1981):

![](./media/image2.emf){width="3.2807042869641294in"
height="0.4997976815398075in"}

泥沙浓度高时，底部流速计算值减小，拖拽力也减小，导致计算水位偏小。

### 1.3 Sediment settling velocity

The hindered settling effect follow Richardson and Zaki(1954):

![](./media/image3.emf){width="1.8043121172353456in"
height="0.5154265091863517in"}

### 1.4 Dispersive transport

Eddy diffusivities depend on the flow characteristics and the influence
of waves.

Constant coefficient, Algebric eddy viscosity closure model,
K-L,k-epsilon closure model

### 1.5 Three-dimensional wave effects

### 1.6 Initial and boundary condition

## 2. Cohesive sediment

### 2.1 Cohesive sediment settling velocity

Flocculate depending the salinity

![](./media/image4.emf){width="5.531219378827647in"
height="1.9229790026246718in"}

Remark: 没有考虑湍流对絮凝的作用。

### 2.2 Cohesive sediment dispersion

![](./media/image5.emf){width="0.7270833333333333in"
height="0.27291666666666664in"}

### 2.3 Cohesive sediment erosion and deposition

Calculated with the well-known Partheniades --Krone formulations
(Partheniades, 1965):

![](./media/image6.emf){width="2.1284722222222223in"
height="1.1972222222222222in"}

### 2.4 interaction of sediment fractions

## 3. Fluid mud(2-layer approach) (Wang Z.B. and Winterwerp, 1992)

This is an Eulerian method.

This feature (beta functionality) concerns the simulation of a fluid mud
layer together with an overlying suspension layer using the 3D sediment
transport.

Fluid mud layers may be formed from deposition of sediment from the
water column. Once formed, the fluid mud layer may be forced to flow by
gravity or shear stresses induced by the overlying water. If shear
stresses are high enough, material from the mud layer may be entrained
into the water column. When the flow in the mud layer is sufficiently
high, the mud layer may erode the consolidated bed. Material thus eroded
contributes to the fluid mud layer. When the fluid mud layer is at rest,
the fluid mud layer is subjected to consolidation. The processes
described above, are depicted in Figure B.12.

![](./media/image7.emf){width="5.159344925634295in"
height="2.7231977252843396in"}

Winterwerp, J. C., Z. B. Wang, J. A. T. M. van Kester and F. J. Verweij,
1999. "On the far-field impact of Water Injection Dredging." J.
Waterway, Port, Coastal and Ocean Engineering

### 3.1 Two-layer system

The system of a suspension layer and a fluid mud layer is considered as
a two layer system. A suspension layer is driven by tidal forces and
possibly by wind shear stresses. Sediment may be transferred from the
bottom to the fluid mud layer by erosion and the other way around by
consolidation and sediment may be transferred from the fluid mud layer
to the suspension layer by means of entrainment and the other way around
by deposition.

### 3.2 Suspension layer

It is assumed that the velocities in the suspension layer are much
larger than the velocities in the fluid mud layer. In addition it is
assumed that the thickness of the fluid mud layer is negligible compared
to the thickness of the water layer. Consequently, the fluid mud layer
is therefore assumed not to influence the suspension layer
hydrodynamically. For the present system, the bed roughness is also
independent of the presence of the fluid mud layer. The fluid mud layer
acts as a sediment source or sink for the suspension layer.

### 3.3 Fluid mud layer

The fluid mud flow is assumed to be driven by the shear stresses acting
in the interface of suspension layer and fluid mud. The non-Newtonian
rheological properties of the fluid mud layer are accounted for by the
formulation of the shear stresses in the interfaces of suspension and
fluid mud layer, and of fluid mud layer and the consolidated bed, if the
fluid mud layer is modeled as 2DH problem.

The shear stresses in the fluid mud-water interface depend on the
velocity difference between the two layers using a material specific
friction coefficient. The shear stresses in the bed-fluid mud layer
depend on the speed of the fluid mud flow, again using a material
specific friction coefficient and a Bingham yield stress. Sediment is
transported from the bed to the fluid mud layer, to the suspension layer
and vice versa. The density of the fluid mud layer is assumed to be
constant, as is the density of the consolidated bed. In the present
formulations, the thickness of the consolidated bed is assumed not to
change with time. Furthermore, the mud bed is assumed to be an infinite
source of mud.

### 3.4 Mathematical modeling of fluid mud layer

The mathematical model for the fluid mud layer is given by Wang and
Winterwerp (1992). Herein only a concise description is provided. The
mass balance reads:

![](./media/image8.emf){width="2.7576388888888888in"
height="0.4847222222222222in"}

Where *t* is time, *x* and *y* are spatial coordinates, *d~m~* is the
thickness of the fluid mud layer, *u~m~* and *v~m~* are the speed of
fluid mud layer in *x*-direction and *y*-direction, respectively. *c~m~*
the (constant) sediment concentration in the mud layer(kg/m^3^), the
right-hand term in Eq. B14 represents the source term in the mass
balance.

The equations of motion for both directions read:

![](./media/image9.emf){width="5.768055555555556in"
height="1.0712193788276465in"}

![](./media/image10.emf){width="5.768055555555556in"
height="1.2673009623797025in"}

The shear stresses are given by:

![](./media/image11.emf){width="2.689583333333333in"
height="1.257638888888889in"}

![](./media/image12.emf){width="3.0909722222222222in"
height="0.8180555555555555in"}

The sediment exchange rate between the suspension layer and the fluid
mud layer includes:

-   Settling

-   entrainment

and between the fluid mud layer and the (consolidated) bed:

-   erosion

-   consolidation, when the fluid mud layer is at rest (um = vm = 0).

The source term in the mass balance equation B.14 is given by:

![](./media/image13.emf){width="3.9923611111111112in"
height="0.4395833333333333in"}

The settling term is given by:

![](./media/image14.emf){width="3.12961832895888in"
height="0.5383934820647419in"}

The entrainment term is given by Winterwerp et al. (1999)

![](./media/image15.emf){width="5.257638888888889in"
height="0.6284722222222222in"}

The erosion term is given by:

![](./media/image16.emf){width="3.0229166666666667in"
height="0.5229166666666667in"}

The dewatering term is given by:

![](./media/image17.emf){width="1.4243055555555555in"
height="0.28055555555555556in"}

浮泥层为平面2D求解。

### 3.5 Applying fluid mud

The same code can be applied to both the suspension layer (possibly in
3D) and the fluid mud layer (only 2DH).

A simulation of a fluid mud problem requires the execution of two
modules, one for the suspension layer and one for the fluid mud layer.

![](./media/image18.emf){width="2.9618055555555554in"
height="4.083333333333333in"}

**Figure B.13:** A schematic representation of two Delft3D-FLOW modules
running simultaneously simulating a fluid mud problem

# 四、粘性-非粘性混合沙模型（Delft3D-SandMud module）

**From the draft (Thijs van Kessel et al., 2012)**

Sediment transport is often not solely determined by the carrying
capacity of the flow, but also by the sediment supply from the bed. The
sink and source terms from and towards the bed depend on sediment
composition. It is therefore important that information on sediment
composition is tracked within a model. Formulations should describe the
erosion behavior of the bed as a function of bed composition.

If the mud fraction of the bed remains below a critical value, the
erosion behavior of the bed is non-cohesive. However, if the critical
value is exceeded, the behavior switches to cohesive. Erosion
formulations change accordingly.

## 4.1 Introduction

Fine sediments penetrate into or are released from a sandy substrate,
depending on the hydrodynamic conditions. Release of fine sediments
results in an increasing sediment concentration in the upper water
column, while penetration of fines into the bed lowers the suspended
sediment concentration. This so-called buffering of fines into a sandy
substrate has been demonstrated to influence the annual variation of the
suspended sediment concentration in the North Sea (Kleinhans et al.,
2005).

The physical processes responsible for the buffering of fine sediments
still remain to be further explored.

This literature review aims to summarize the mud infiltration mechanisms
in sub-tidal marine sediment beds, estimates of mixing rates and
residence times of mud in sand beds, and state-of-the-art modeling
efforts in North Sea. The infiltration mechanisms in intertidal areas
such as swash on sloping beaches (Turner and Masselink, 1998) or sand
banks (Gibbes et al. 2008a, 2008b) are not evaluated here.

## 4.2 Mud infiltration mechanisms

In muddy, cohesive sediment, solutes are mainly transported by molecular
diffusion, and possibly by benthic fauna. Interstitial flow is much
faster in permeable sands, resulting in higher advective transport rates
(Figure 4.1). Advective transport is flow-induced material transport
through sediment pores, driven by pressure gradients. Especially in the
presence of bedforms, advective transport in permeable sediments may be
high. Furthermore, fine sediment may be mixed or released from a sand
bed through bedform migration. These mechanisms can be summarized as:

1.  sedimentological processes;

2.  biological processes;

3.  advective transport;

4.  molecular diffusion.

![](./media/image19.emf){width="2.714080271216098in"
height="2.508260061242345in"}

Figure 4.1 Mixing process as a function of depth and permeability (D, in
10^-13^ m^2^). From Huettel and Gust, 1992

### 4.2.1 Sedimentological processes

The entrainment of muds into suspension is coupled to bedform dynamics.
Mobile bedforms create an active layer of sediment with a thickness
approximately equal to the bedform height, to be mobilised during one or
a few storms. On a timescale of 1-10 years, bedforms smaller than
sandwaves, macrobenthos, and fishing determine the mobility depth
(Kleinhans et al, 2005).

Mud infiltrates a sand bed by bedform dynamics through two mechanisms:

**(1) Mud drapes (consolidated or not) are covered by bedforms**

Mud drapes are typically 1-5 mm thick in North Sea. Burial by large
bedforms, such as sand waves, may lead to residence times of the mud in
the sand bed of 1 century.

**(2) The upper part of a sediment bed is permanently mobilised by
ripples (the active layer), even during fairly tranquil conditions
during which a substantial amount of fine sediment may settle towards
the bed.**

In the North Sea, ripples generated by tidal currents bury fine sediment
up to a depth of 5 cm (Kleinhans et al., 2005). Most organisms must
migrate below the active layer to survive highly energetic conditions
(Kleinhans et al., 2005).

### 4.2.2 Bioturbation

Organisms influence sediment beds mainly through biodeposition,
biostabilization, and bioturbation. Biodeposition results in increased
sedimentation rates, biostabilization in reduced erosion rates, and
bioturbation in increased mixing rates (which, usually, increases the
erosion rate).

Modelling results by Paarlberg et al. (2004) suggest that destabilizing
organisms reduce the mud content in a bed, whereas stabilizing organisms
*may* cause an increase in the mud content.

Biodeposition through faeces production is different for deposit
feeders, which only change the characteristics of existing sediment, and
filter feeders, which add new material to the sediment (Lee and Swartz,
1980). Bed sediment is stabilized through microbial binding
(micro-organisms, especially polysaccharides), tubes (especially species
that burrow firm tubes), roots by aquatic vegetation, and by benthic
macroalgal mats. Bioturbation results in the vertical mixing of surface
sediment. Mixing rate and burial depth depend on the species that occur
at a typical site, which in turn relate to the hydrodynamic conditions,
sediment available, nutrients and temperature. Hence, the variation of
mixing by bioturbation is large.

An extensive review of reworking rate and penetration depth is given by
Lee and Swartz (1980).

### 4.2.3 Advective transport

Advective transport is the transport of fines by interstitial porewater
flow in stationary sediment beds. Fines and the coarser sediment matrix
are therefore not physically mixed: the sand matrix through which the
fines percolate remains unchanged. Mechanisms of advective transport are
(Huettel and Webster, 2001):

-   Current-topography interaction by steady flow as well as by
    oscillating wave-generated currents. (Current-topography
    interaction)

![](./media/image20.emf){width="4.138447069116361in"
height="2.4927088801399826in"}

Figure 4.2 Infiltration and outflow due to pressure variations over
bed-topography interaction in steady uniform flow. From Huettel et al.,
1998

-   Wave-generated pressure gradients.

Alternatively, mud infiltration into sand beds may be substantially
increased by wave generated pressure gradients.

-   Haline or thermal convection.

Mixing rates by haline convection are substantially larger than mixing
rates by molecular diffusion (Huettel and Webster, 2001).

### 4.2.4 Turbulent diffusion

In absence of other mixing processes, fines transported in pore waters
may be mixed within the sand matrix by turbulent diffusion. However,
turbulent diffusion is probably several orders of magnitude lower than
any of the other mechanism, and is therefore of minor importance.

### 4.2.5 Modeling method

Several numerical models exist that compute the grain size distribution
of sediment beds using different sediment fractions. In these models,
sediment is deposited as distinct layers which are vertically mixed
within the bed through a diffusion coefficient representing
bioturbation. (还是非粘性沙的模拟思路)

Harris and Wiberg (1996), for instance, apply a double-layer sediment
transport model to account for the effect of bed armouring. The upper
layer may be resuspended, while sediment from the lower layer may be
mixed upward by biodiffusion. The depth of the active layer is related
to the excess shear strength and the grain size.
（CSTM非粘性模型中活动层厚度的计算方法）

Sanford (2008) developed a 1DV numerical (MATLAB) model simulating the
effect of bioturbation on the vertical distribution of sand and mud.

Van Ledden (2003, 2004) developed a sand-mud model as part of the
Delft3D model suite that modeled the vertical and spatial segregation of
sand-mud mixtures. A distinguishing aspect of van Ledden's model is that
it also models the effect of mixtures on erosion rates. (Wang Z.B. and
Winterwerp, 1992; Van Ledden，2004)

Paarlberg et al (2004) extended the sand-mud model developed by Van
Ledden (2003) with biotic stabilization and mixing.

A two layer mud buffering model is developed by van Prooijen et al.
(2007) to simulate fine sediment transport in a sandy environment.

**The first layer** is representative for the thin fluff layer on the
bed surface that forms during slack tide and that is easily re-suspended
by tidal currents. The total sediment mass in this layer tends to be
small and the residence time of sediment in this layer is short because
of the large flux between the fluff layer and the water column.

**The second layer** is representative for the sandy seabed into which
fines may entrain and temporarily be stored. Re-suspension from this
buffer layer is only significant during highly dynamic conditions, such
as spring tide or storms.

A user-defined fraction of the fine sediment is transported from the
water column into the layer 1 and 2, and eroded when the bed shear
stress exceeds a critical value *τ*cr. Since*τ*cr of the layer 2 (lower
layer) is higher than the layer 1 (upper layer), sediment is transported
from layer 1 to layer 2 when*τ*cr, 1 \<*τ*\<*τ*cr,2. An alternative
method to transport sediment from layer 1 to layer 2 is through a
so-called burial term in which a user defined percentage of fines in the
upper layer is transported to the deeper layer. This latter is more in
analogy to diffusion coefficients representing bioturbation used in
other models. The erosion rates in van Kessel's and van Prooijen's model
is not influenced by the composition of the sand-mud mixtures. Also, it
only simulates the transport of mud, and not of sand.

## 4.3 Residence time & infiltration rates

### 4.3.1 infiltration rates

Infiltration rates are highest for fines with a grain size of 1μm
because smaller grain sizes adhere to particles, while coarser are
blocked by the pores (Huettel et al., 1996).

The intrusion velocity is episodic, depending on the wave height and
current velocity. Intrusion velocities rapidly decline with depth and
time.

Infiltration rates are determined by hydrodynamics and sediment
dynamics. Especially the permeability is important for the inflow
velocity. The inflow velocity scales linearly with permeability, but at
different locations around the world the permeability differs several
orders of magnitude.

### 4.3.2 Mixing rates & diffusion coefficients

*Ds* between 0.18 10^-6^ and 1.43 10^-6^ cm^2^/s

### 4.3.3 Residence time

The time that fines remain buried in the bed can be expressed in terms
of a half life, and as a residence time.

This residence time is estimated by dividing the mixing depth by the
accumulation rate. Hence, the residence time is defined as the time
required for the sediments to be permanently deposited.

From a geological point of view, bioturbation is a slow and continuous
process, and therefore biological mixing dominates sedimentary strata on
longer timescales (Dellapenna, 1998). The duration, depth, and degree of
sediment disturbance by biological processes depend on the benthic
community structure and its temporal and spatial variability. Physical
mixing is episodic (event-driven), and modulated on a variety of
frequencies and depths, depending on the driving forces (e.g. wind,
waves, tides, surges).

### 4.3.4 Measurement methods

Measurement methods which can be used to measure the flux of mud into or
out of the sediment bed are basically:

\(1\) to determine the mixing rates from sediment cores, using some sort
of natural tracer;

\(2\) active field measurements during which the decay of a substance
injected into the bed is monitored;

\(3\) indirect measurements such as sediment concentration time series.

For example: radioactive tracers, contaminants, particulate organic
matter concentrations; redox potential discontinuity (RPD) layer

## 4.4 Erosion rate of sand-mud mixtures

The network structure of a sediment bed determines the erosion type and
rate significantly. A conceptual framework for the erosion behaviour of
sand--mud mixtures was proposed by van Ledden et al. (2004), identifying
a cohesive and non-cohesive sand-dominated network structure, a cohesive
and non-cohesive silt-dominated network structure, a non-cohesive mixed
structure, and a cohesive clay-dominated structure. The transitions
between silt--sand--clay domination are determined by the volume
fractions of the different sediment types, and the total water content
*n*. The behavior of the sediment bed is dominated by a sand skeleton
when at least 40% of the volume fraction (including sediment and water)
consists of sand.

The critical clay or mud content for a sediment bed to become cohesive
can be determined through experiments. Van Ledden et al. (2004)
concluded that a clay fraction of 7.5% results in a transition of
cohesive to non-cohesive properties.

In addition to the clay content or mud content, the erosion rate is also
strongly influenced by biological activity. Biostabilization reduce
erosion rates, while bioturbation increases the mixing rate (which,
usually, increasesthe erosion rate). The biological activity therefore
also influence the sediment composition of the bed: destabilizing
organisms reduce the mud content in a bed, whereas stabilizing organisms
*may* cause an increase in the mud content (Paarlberg et al., 2004).

## 4.5 The bed module of sand-mud mixure

### 4.5.1 Introduction

Those wanting to integrate the bed module into their own sediment
transport code should use the Fortran modules.

### 4.5.2 Mixed Eulerian-Lagrangian approach

Aggradation and degradation of the bed can be treated in a Lagrangian or
Eulerian framework, see Figure 4.3. In the Lagrangian framework, the
thickness of the layers is constant and the set of layers moves with the
aggradation/degradation by means of an artificial advection velocity.
The advantage of this method is that the grid itself does not change.
This means that the thicknesses of the cells can vary over the depth. In
many cases it is desirable to have a high resolution near the bed
surface and a coarser resolution at a larger depth. The drawback is
that, depending on the numerical scheme, the movement of the grid
results in artificial diffusion between the layers. Stratification will
then be smoothed out.

In the Eulerian framework, the position of the layers is kept constant.
The aggradation/degradation is accounted for by changing the thickness
of the top layer. In case the top layer becomes too thick due to
deposition, it will be split. In case the thickness tends to zero due to
erosion, the layer is merged with the second layer. The drawback of the
Eulerian framework is that the top layer has no fixed thickness. As this
thickness has a significant effect on the time scales of the system, it
is desirable to have a top layer with a constant thickness, or a even
multiple top layers with constant thickness. This implies a combination
of the Eulerian and the Lagrangian framework.

![](./media/image21.emf){width="4.970811461067367in"
height="3.120291994750656in"}

Figure 3.1 Change of grid for erosion and deposition for the Eulerian
and Lagrangian framework

In the following three examples, the difference between using a
Lagrangian, Eulerian and mixed framework is shown. The simulation
involves a quasi-1D channel with a trench.

![](./media/image22.emf){width="2.742100831146107in"
height="2.0321686351706036in"}![](./media/image23.emf){width="2.8318252405949256in"
height="2.1082436570428698in"}

Initial condition Fully-Euler

![](./media/image24.emf){width="2.7890529308836394in"
height="2.112186132983377in"}
![](./media/image25.emf){width="2.7612248468941383in"
height="2.1113188976377955in"}

Fully-Lagrange Mixed

It can be seen that using a Lagrangian or mixed framework indeed results
in more artificial diffusion between the layers.

Fully-Euler: The thickness of the top layer is constant and there are no
Lagrangian layers

Fully-Lagrange: Only Lagrangian layers are defined.

Mixed: There are two Lagrangian layers (beneath the fixed top layer) and
the other layers are Eulerian.

### 4.5.3 Mixing between layers

### 4.5.4 Fluff layer concept

As the fluff layer is very thin, it does hardly contribute to the bed
level. The amount of sediment in the fluff layer should be in the order
of magnitude of the amount of sediment in the water during spring tide
(O(0.1-1 g/m2, depending on the location).the fluff layer is treated
separately.

The properties of the fluff layer are different from the ones in the
bed. For example, the critical shear stress will be much smaller. There
are two different possibilities to define the fluxes (see Figure 3.20)

For both methods holds:

-   The fluff layer contains no sand. The sand fluxes are only between
    bed and water.

-   The fluff layer is in between the bed and the water column, but has
    no contribution to the water depth or bed level.

![](./media/image26.emf){width="4.1647670603674545in"
height="2.8182338145231847in"}

Figure 3.20 Fluff layer

### 4.5.5 Formulations for sand-mud interaction

a distinction between a cohesive and non-cohesive regime

\(1\) non-cohesive regime
(![](./media/image27.emf){width="0.7205161854768154in"
height="0.18482502187226596in"})

\(2\) cohesive regime

## 五、絮凝动力学(Verney et al., 2011)

## 参考文献

L. Pinto, A.B. Fortunato, Y. Zhang, et al. 2012. Development and
validation of a three-dimensional morphodynamic modeling system for
non-cohesive sediments. Ocean Modelling, 57-58: 1-14.

Warner, J.C., Sherwood, C.R., Signell, R.P., Harris, C.K., Arango, H.G.,
2008. Development of a three-dimensional, regional, coupled wave,
current, and sediment-transport model. Computers and Geosciences 34,
1284-1306.

Roelvink, J.A., 2006. Coastal morphodynamic evolution techniques.
Coastal Engineering 53, 277-287.

Urs Neumeier, Christian Ferrarin, Carl L. Amos, et al. 2008. Sedtrans05:
An improved sediment-transport model for continental shelves and coastal
waters with a new algorithm for cohesive sediments. Computers &
Geosciences, 34: 1223-1242.

Umgiesser G., Depascalis F., Ferrarin C., Amos C.L., 2006. A model of
sand transport in Treporti channel: northern Venice lagoon. Ocean
Dynamics, 56: 339-351.

Delft3D-FLOW: Simulation of multi-dimensional hydrodynamic flows and
transport phenomena, including sediments. Hydro-Morphodynamics User
Manual, Version: 3.15.34158, 2014-5-28

Thijs van Kessel et al., 2012. Bed module for sand-mud mixtures in
framework of BwN project NTW 1.3 mud dynamics (Draft)

[Janette L.
Cookman](http://www.sciencedirect.com/science/article/pii/S0098300400001217), [Peter
B.
Flemings](http://www.sciencedirect.com/science/article/pii/S0098300400001217).
2001. STORMSED1.0: hydrodynamics and sediment transport in a 2-D,
steady-state, wind- and wave-driven coastal circulation model.
[Computers &
Geosciences](http://www.sciencedirect.com/science/journal/00983004),
27(6): 647-674.

## 3. Fluid mud(2-layer approach) (Wang Z.B. and Winterwerp, 1992)

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

![](./media/image1.emf){width="5.159344925634295in"
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

![](./media/image2.emf){width="2.4558016185476816in"
height="0.431667760279965in"}

Where *t* is time, *x* and *y* are spatial coordinates, *d~m~* is the
thickness of the fluid mud layer, *u~m~* and *v~m~* are the speed of
fluid mud layer in *x*-direction and *y*-direction, respectively. *c~m~*
the (constant) sediment concentration in the mud layer(kg/m^3^), the
right-hand term in Eq. B14 represents the source term in the mass
balance.

The equations of motion for both directions read:

![](./media/image3.emf){width="5.509945319335083in"
height="1.0232841207349082in"}

![](./media/image4.emf){width="5.465707567804024in"
height="1.200871609798775in"}

The shear stresses are given by:

![](./media/image5.emf){width="2.403530183727034in"
height="1.1238812335958006in"}

![](./media/image6.emf){width="2.6908748906386704in"
height="0.7121653543307087in"}

The sediment exchange rate between the suspension layer and the fluid
mud layer includes:

-   Settling

-   entrainment

and between the fluid mud layer and the (consolidated) bed:

-   erosion

-   consolidation, when the fluid mud layer is at rest (um = vm = 0).

The source term in the mass balance equation B.14 is given by:

![](./media/image7.emf){width="3.9923611111111112in"
height="0.4395833333333333in"}

The settling term is given by:

![](./media/image8.emf){width="2.655949256342957in"
height="0.4569083552055993in"}

The entrainment term is given by Winterwerp et al. (1999)

![](./media/image9.emf){width="4.750240594925634in"
height="0.5678204286964129in"}

The erosion term is given by:

![](./media/image10.emf){width="3.0229166666666667in"
height="0.5229166666666667in"}

The dewatering term is given by:

![](./media/image11.emf){width="1.4243055555555555in"
height="0.28055555555555556in"}

浮泥层为平面2D求解。

### 3.5 Applying fluid mud

The same code can be applied to both the suspension layer (possibly in
3D) and the fluid mud layer (only 2DH).

A simulation of a fluid mud problem requires the execution of two
modules, one for the suspension layer and one for the fluid mud layer.

![](./media/image12.emf){width="2.7064391951006126in"
height="3.7312685914260717in"}

**Figure B.13:** A schematic representation of two Delft3D-FLOW modules
running simultaneously simulating a fluid mud problem

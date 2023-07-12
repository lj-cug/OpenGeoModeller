# Installation Overview

**Table of Contents**

- [Installation Overview](#installation-overview)
    * [Steps to Installing the Lustre Software](#steps-to-installing-the-lustre-software)

This chapter provides on overview of the procedures required to set up, install and configure a Lustre file system.

### Note

If the Lustre file system is new to you, you may find it helpful to refer to [Introducing the Lustre* File System](02-Introducing%20the%20Lustre%20File%20System.md) for a description of the Lustre architecture, file system components and terminology before proceeding with the installation procedure.

## Steps to Installing the Lustre Software

To set up Lustre file system hardware and install and configure the Lustre software, refer the the chapters below in the order listed:

1. *(Required)* **Set up your Lustre file system hardware.**

   See [*Determining Hardware Configuration Requirements and Formatting Options*](02.02-Determining%20Hardware%20Configuration%20Requirements%20and%20Formatting%20Options.md) - Provides guidelines for configuring hardware for a Lustre file system including storage, memory, and networking requirements.

2. *(Optional - Highly Recommended)* **Configure storage on Lustre storage devices.**

   See [*Configuring Storage on a Lustre File System*](02.03-Configuring%20Storage%20on%20a%20Lustre%20File%20System.md) - Provides instructions for setting up hardware RAID on Lustre storage devices.

3. *(Optional)* **Set up network interface bonding.**

   See [*Setting Up Network Interface Bonding*](02.04-Setting%20Up%20Network%20Interface%20Bonding.md) - Describes setting up network interface bonding to allow multiple network interfaces to be used in parallel to increase bandwidth or redundancy.

4. *(Required)* **Install Lustre software.**

   See [*Installing the Lustre Software*](02.05-Installing%20the%20Lustre%20Software.md) - Describes preparation steps and a procedure for installing the Lustre software.

5. *(Optional)* **Configure Lustre Networking (LNet).**

   See [*Configuring Lustre Networking (LNet)*](02.06-Configuring%20Lustre%20Networking%20(LNet).md) - Describes how to configure LNet if the default configuration is not sufficient. By default, LNet will use the first TCP/IP interface it discovers on a system. LNet configuration is required if you are using InfiniBand or multiple Ethernet interfaces.

6. *(Required)* **Configure the Lustre file system.**

   See [*Configuring a Lustre File System*](02.07-Configuring%20a%20Lustre%20File%20System.md) - Provides an example of a simple Lustre configuration procedure and points to tools for completing more complex configurations.

7. *(Optional)* **Configure Lustre failover.**

   See [*Configuring Failover in a Lustre File System*](02.08-Configuring%20Failover%20in%20a%20Lustre%20File%20System.md) - Describes how to configure Lustre failover.


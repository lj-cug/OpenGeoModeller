# Setting Up Network Interface Bonding

**Table of Contents**

- [Setting Up Network Interface Bonding](#setting-up-network-interface-bonding)
    * [Network Interface Bonding Overview](#network-interface-bonding-overview)
  * [Requirements](#requirements)
  * [Bonding Module Parameters](#bonding-module-parameters)
  * [Setting Up Bonding](#setting-up-bonding)
    + [Examples](#examples)
  * [Configuring a Lustre File System with Bonding](#configuring-a-lustre-file-system-with-bonding)
  * [Bonding References](#bonding-references)

This chapter describes how to use multiple network interfaces in parallel to increase bandwidth and/or redundancy. Topics include:

- [the section called “Network Interface Bonding Overview”](#network-interface-bonding-overview)
- [the section called “Requirements”](#requirements)
- [the section called “Bonding Module Parameters”](#bonding-module-parameters)
- [the section called “Setting Up Bonding”](#setting-up-bonding)
- [the section called “Configuring a Lustre File System with Bonding”](#configuring-a-lustre-file-system-with-bonding)
- [the section called “Bonding References”](#bonding-references)

### Note

Using network interface bonding is optional.

## Network Interface Bonding Overview

Bonding, also known as link aggregation, trunking and port trunking, is a method of aggregating multiple physical network links into a single logical link for increased bandwidth.

Several different types of bonding are available in the Linux distribution. All these types are referred to as 'modes', and use the bonding kernel module.

Modes 0 to 3 allow load balancing and fault tolerance by using multiple interfaces. Mode 4 aggregates a group of interfaces into a single virtual interface where all members of the group share the same speed and duplex settings. This mode is described under IEEE spec 802.3ad, and it is referred to as either 'mode 4' or '802.3ad.'

## Requirements

The most basic requirement for successful bonding is that both endpoints of the connection must be capable of bonding. In a normal case, the non-server endpoint is a switch. (Two systems connected via crossover cables can also use bonding.) Any switch used must explicitly handle 802.3ad Dynamic Link Aggregation.

The kernel must also be configured with bonding. All supported Lustre kernels have bonding functionality. The network driver for the interfaces to be bonded must have the ethtool functionality to determine slave speed and duplex settings. All recent network drivers implement it.

To verify that your interface works with ethtool, run:

```
# which ethtool
/sbin/ethtool
 
# ethtool eth0
Settings for eth0:
           Supported ports: [ TP MII ]
           Supported link modes:   10baseT/Half 10baseT/Full
                                   100baseT/Half 100baseT/Full
           Supports auto-negotiation: Yes
           Advertised link modes:  10baseT/Half 10baseT/Full
                                   100baseT/Half 100baseT/Full
           Advertised auto-negotiation: Yes
           Speed: 100Mb/s
           Duplex: Full
           Port: MII
           PHYAD: 1
           Transceiver: internal
           Auto-negotiation: on
           Supports Wake-on: pumbg
           Wake-on: d
           Current message level: 0x00000001 (1)
           Link detected: yes
 
# ethtool eth1
 
Settings for eth1:
   Supported ports: [ TP MII ]
   Supported link modes:   10baseT/Half 10baseT/Full
                           100baseT/Half 100baseT/Full
   Supports auto-negotiation: Yes
   Advertised link modes:  10baseT/Half 10baseT/Full
   100baseT/Half 100baseT/Full
   Advertised auto-negotiation: Yes
   Speed: 100Mb/s
   Duplex: Full
   Port: MII
   PHYAD: 32
   Transceiver: internal
   Auto-negotiation: on
   Supports Wake-on: pumbg
   Wake-on: d
   Current message level: 0x00000007 (7)
   Link detected: yes
   To quickly check whether your kernel supports bonding, run:     
   # grep ifenslave /sbin/ifup
   # which ifenslave
   /sbin/ifenslave
```

## Bonding Module Parameters

Bonding module parameters control various aspects of bonding.

Outgoing traffic is mapped across the slave interfaces according to the transmit hash policy. We recommend that you set the `xmit_hash_policy` option to the layer3+4 option for bonding. This policy uses upper layer protocol information if available to generate the hash. This allows traffic to a particular network peer to span multiple slaves, although a single connection does not span multiple slaves.

```
$ xmit_hash_policy=layer3+4
```

The `miimon` option enables users to monitor the link status. (The parameter is a time interval in milliseconds.) It makes an interface failure transparent to avoid serious network degradation during link failures. A reasonable default setting is 100 milliseconds; run:

```
$ miimon=100
```

For a busy network, increase the timeout.

## Setting Up Bonding

To set up bonding:

1. Create a virtual 'bond' interface by creating a configuration file:

   ```
   # vi /etc/sysconfig/network-scripts/ifcfg-bond0
   ```

2. Append the following lines to the file.

   ```
   DEVICE=bond0
   IPADDR=192.168.10.79 # Use the free IP Address of your network
   NETWORK=192.168.10.0
   NETMASK=255.255.255.0
   USERCTL=no
   BOOTPROTO=none
   ONBOOT=yes
   ```

3. Attach one or more slave interfaces to the bond interface. Modify the eth0 and eth1 configuration files (using a VI text editor).

   1. Use the VI text editor to open the eth0 configuration file.

      ```
      # vi /etc/sysconfig/network-scripts/ifcfg-eth0
      ```

   2. Modify/append the eth0 file as follows:

      ```
      DEVICE=eth0
      USERCTL=no
      ONBOOT=yes
      MASTER=bond0
      SLAVE=yes
      BOOTPROTO=none
      ```

   3. Use the VI text editor to open the eth1 configuration file.

      ```
      # vi /etc/sysconfig/network-scripts/ifcfg-eth1
      ```

   4. Modify/append the eth1 file as follows:

      ```
      DEVICE=eth1
      USERCTL=no
      ONBOOT=yes
      MASTER=bond0
      SLAVE=yes
      BOOTPROTO=none
      ```

4. Set up the bond interface and its options in `/etc/modprobe.d/bond.conf`. Start the slave interfaces by your normal network method.

   ```
   # vi /etc/modprobe.d/bond.conf
   ```

   1. Append the following lines to the file.

      ```
      alias bond0 bonding
      options bond0 mode=balance-alb miimon=100
      ```

   2. Load the bonding module.

      ```
      # modprobe bonding
      # ifconfig bond0 up
      # ifenslave bond0 eth0 eth1
      ```

5. Start/restart the slave interfaces (using your normal network method).

   ### Note

   You must `modprobe` the bonding module for each bonded interface. If you wish to create bond0 and bond1, two entries in `bond.conf` file are required.

   The examples below are from systems running Red Hat Enterprise Linux. For setup use: `/etc/sysconfig/networking-scripts/ifcfg-*` The website referenced below includes detailed instructions for other configuration methods, instructions to use DHCP with bonding, and other setup details. We strongly recommend you use this website.

   <http://www.linuxfoundation.org/collaborate/workgroups/networking/bonding>

6. Check /proc/net/bonding to determine status on bonding. There should be a file there for each bond interface.

   ```
   # cat /proc/net/bonding/bond0
   Ethernet Channel Bonding Driver: v3.0.3 (March 23, 2006)
    
   Bonding Mode: load balancing (round-robin)
   MII Status: up
   MII Polling Interval (ms): 0
   Up Delay (ms): 0
   Down Delay (ms): 0
    
   Slave Interface: eth0
   MII Status: up
   Link Failure Count: 0
   Permanent HW addr: 4c:00:10:ac:61:e0
    
   Slave Interface: eth1
   MII Status: up
   Link Failure Count: 0
   Permanent HW addr: 00:14:2a:7c:40:1d
   ```

7. Use ethtool or ifconfig to check the interface state. ifconfig lists the first bonded interface as 'bond0.'

   ```
   ifconfig
   bond0      Link encap:Ethernet  HWaddr 4C:00:10:AC:61:E0
      inet addr:192.168.10.79  Bcast:192.168.10.255 \     Mask:255.255.255.0
      inet6 addr: fe80::4e00:10ff:feac:61e0/64 Scope:Link
      UP BROADCAST RUNNING MASTER MULTICAST  MTU:1500 Metric:1
      RX packets:3091 errors:0 dropped:0 overruns:0 frame:0
      TX packets:880 errors:0 dropped:0 overruns:0 carrier:0
      collisions:0 txqueuelen:0
      RX bytes:314203 (306.8 KiB)  TX bytes:129834 (126.7 KiB)
    
   eth0       Link encap:Ethernet  HWaddr 4C:00:10:AC:61:E0
      inet6 addr: fe80::4e00:10ff:feac:61e0/64 Scope:Link
      UP BROADCAST RUNNING SLAVE MULTICAST  MTU:1500 Metric:1
      RX packets:1581 errors:0 dropped:0 overruns:0 frame:0
      TX packets:448 errors:0 dropped:0 overruns:0 carrier:0
      collisions:0 txqueuelen:1000
      RX bytes:162084 (158.2 KiB)  TX bytes:67245 (65.6 KiB)
      Interrupt:193 Base address:0x8c00
    
   eth1       Link encap:Ethernet  HWaddr 4C:00:10:AC:61:E0
      inet6 addr: fe80::4e00:10ff:feac:61e0/64 Scope:Link
      UP BROADCAST RUNNING SLAVE MULTICAST  MTU:1500 Metric:1
      RX packets:1513 errors:0 dropped:0 overruns:0 frame:0
      TX packets:444 errors:0 dropped:0 overruns:0 carrier:0
      collisions:0 txqueuelen:1000
      RX bytes:152299 (148.7 KiB)  TX bytes:64517 (63.0 KiB)
      Interrupt:185 Base address:0x6000
   ```

### Examples

This is an example showing `bond.conf` entries for bonding Ethernet interfaces `eth1`and `eth2` to `bond0`:

```
# cat /etc/modprobe.d/bond.conf
alias eth0 8139too
alias eth1 via-rhine
alias bond0 bonding
options bond0 mode=balance-alb miimon=100
 
# cat /etc/sysconfig/network-scripts/ifcfg-bond0
DEVICE=bond0
BOOTPROTO=none
NETMASK=255.255.255.0
IPADDR=192.168.10.79 # (Assign here the IP of the bonded interface.)
ONBOOT=yes
USERCTL=no
 
ifcfg-ethx 
# cat /etc/sysconfig/network-scripts/ifcfg-eth0
TYPE=Ethernet
DEVICE=eth0
HWADDR=4c:00:10:ac:61:e0
BOOTPROTO=none
ONBOOT=yes
USERCTL=no
IPV6INIT=no
PEERDNS=yes
MASTER=bond0
SLAVE=yes
```

In the following example, the `bond0` interface is the master (MASTER) while `eth0`and `eth1` are slaves (SLAVE).

### Note

All slaves of `bond0` have the same MAC address (Hwaddr) - `bond0`. All modes, except TLB and ALB, have this MAC address. TLB and ALB require a unique MAC address for each slave.

```
$ /sbin/ifconfig
 
bond0Link encap:EthernetHwaddr 00:C0:F0:1F:37:B4
inet addr:XXX.XXX.XXX.YYY Bcast:XXX.XXX.XXX.255 Mask:255.255.252.0
UP BROADCAST RUNNING MASTER MULTICAST MTU:1500  Metric:1
RX packets:7224794 errors:0 dropped:0 overruns:0 frame:0
TX packets:3286647 errors:1 dropped:0 overruns:1 carrier:0
collisions:0 txqueuelen:0
 
eth0Link encap:EthernetHwaddr 00:C0:F0:1F:37:B4
inet addr:XXX.XXX.XXX.YYY Bcast:XXX.XXX.XXX.255 Mask:255.255.252.0
UP BROADCAST RUNNING SLAVE MULTICAST MTU:1500  Metric:1
RX packets:3573025 errors:0 dropped:0 overruns:0 frame:0
TX packets:1643167 errors:1 dropped:0 overruns:1 carrier:0
collisions:0 txqueuelen:100
Interrupt:10 Base address:0x1080
 
eth1Link encap:EthernetHwaddr 00:C0:F0:1F:37:B4
inet addr:XXX.XXX.XXX.YYY Bcast:XXX.XXX.XXX.255 Mask:255.255.252.0
UP BROADCAST RUNNING SLAVE MULTICAST MTU:1500  Metric:1
RX packets:3651769 errors:0 dropped:0 overruns:0 frame:0
TX packets:1643480 errors:0 dropped:0 overruns:0 carrier:0
collisions:0 txqueuelen:100
Interrupt:9 Base address:0x1400
```

## Configuring a Lustre File System with Bonding

The Lustre software uses the IP address of the bonded interfaces and requires no special configuration. The bonded interface is treated as a regular TCP/IP interface. If needed, specify `bond0` using the Lustre `networks` parameter in `/etc/modprobe`.

```
options lnet networks=tcp(bond0)
```

## Bonding References

We recommend the following bonding references:

- In the Linux kernel source tree, see`documentation/networking/bonding.txt`
- <http://linux-ip.net/html/ether-bonding.html>.
- <http://www.sourceforge.net/projects/bonding>.
- Linux Foundation bonding website: <http://www.linuxfoundation.org/collaborate/workgroups/networking/bonding>. This is the most extensive reference and we highly recommend it. This website includes explanations of more complicated setups, including the use of DHCP with bonding.


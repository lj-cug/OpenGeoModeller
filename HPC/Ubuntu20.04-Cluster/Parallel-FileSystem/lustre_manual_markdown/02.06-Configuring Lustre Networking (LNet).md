## Configuring Lustre Networking (LNet)

**Table of Contents**

- [Configuring LNet via `lnetctl`](#configuring-lnet-via-lnetctl)L 2.7
  * [Configuring LNet](#configuring-lnet)
  * [Displaying Global Settings](#displaying-global-settings)
  * [Adding, Deleting and Showing Networks](#adding-deleting-and-showing-networks)
  * [Manual Adding, Deleting and Showing Peers](#manual-adding-deleting-and-showing-peers)L 2.10
  * [Dynamic Peer Discovery](#dynamic-peer-discovery)L 2.11
    + [Overview](#overview)
    + [Protocol](#protocol)
    + [Dynamic Discovery and User-space Configuration](#dynamic-discovery-and-user-space-configuration)
    + [Configuration](#configuration)
    + [Initiating Dynamic Discovery on Demand](#initiating-dynamic-discovery-on-demand)
  * [Adding, Deleting and Showing routes](#adding-deleting-and-showing-routes)
  * [Enabling and Disabling Routing](#enabling-and-disabling-routing)
  * [Showing routing information](#showing-routing-information)
  * [Configuring Routing Buffers](#configuring-routing-buffers)
  * [Asymmetrical Routes](#asymmetrical-routes)L 2.13
    + [Overview](#overview-1)
    + [Configuration](#configuration-1)
  * [Importing YAML Configuration File](#importing-yaml-configuration-file)
  * [Exporting Configuration in YAML format](#exporting-configuration-in-yaml-format)
  * [Showing LNet Traffic Statistics](#showing-lnet-traffic-statistics)
  * [YAML Syntax](#yaml-syntax)
    + [Network Configuration](#network-configuration)
    + [Enable Routing and Adjust Router Buffer Configuration](#enable-routing-and-adjust-router-buffer-configuration)
    + [Show Statistics](#show-statistics)
    + [Route Configuration](#route-configuration)
- [Overview of LNet Module Parameters](#overview-of-lnet-module-parameters)
  * [Using a Lustre Network Identifier (NID) to Identify a Node](#using-a-lustre-network-identifier-nid-to-identify-a-node)
- [Setting the LNet Module networks Parameter](#setting-the-lnet-module-networks-parameter)
  * [Multihome Server Example](#multihome-server-example)
- [Setting the LNet Module ip2nets Parameter](#setting-the-lnet-module-ip2nets-parameter)
- [Setting the LNet Module routes Parameter](#setting-the-lnet-module-routes-parameter)
  * [Routing Example](#routing-example)
- [Testing the LNet Configuration](#testing-the-lnet-configuration)
- [Configuring the Router Checker](#configuring-the-router-checker)
- [Best Practices for LNet Options](#best-practices-for-lnet-options)
  * [Escaping commas with quotes](#escaping-commas-with-quotes)
  * [Including comments](#including-comments)



This chapter describes how to configure Lustre Networking (LNet). It includes the following sections:

- [the section called “Configuring LNet via `lnetctl`”](#configuring-lnet-via-lnetctl)
- [the section called “ Overview of LNet Module Parameters”](#overview-of-lnet-module-parameters)
- [the section called “Setting the LNet Module networks Parameter”](#setting-the-lnet-module-networks-parameter)
- [the section called “Setting the LNet Module ip2nets Parameter”](#setting-the-lnet-module-ip2nets-parameter)
- [the section called “Setting the LNet Module routes Parameter”](#setting-the-lnet-module-routes-parameter)
- [the section called “Testing the LNet Configuration”](#testing-the-lnet-configuration)
- [the section called “Configuring the Router Checker”](#configuring-the-router-checker)
- [the section called “Best Practices for LNet Options”](#best-practices-for-lnet-options)

### Note

Configuring LNet is optional.

LNet will use the first TCP/IP interface it discovers on a system (`eth0`) if it's loaded using the `lctl network up`. If this network configuration is sufficient, you do not need to configure LNet. LNet configuration is required if you are using Infiniband or multiple Ethernet interfaces.

Introduced in Lustre 2.7The `lnetctl` utility can be used to initialize LNet without bringing up any network interfaces. Network interfaces can be added after configuring LNet via `lnetctl`. `lnetctl` can also be used to manage an operational LNet. However, if it wasn't initialized by `lnetctl` then `lnetctl lnet configure` must be invoked before `lnetctl` can be used to manage LNet.

Introduced in Lustre 2.7DLC also introduces a C-API to enable configuring LNet programatically. See [*LNet Configuration C-API*](06.08-LNet%20Configuration%20C-API.md)



Introduced in Lustre 2.7

## Configuring LNet via `lnetctl`

The `lnetctl` utility can be used to initialize and configure the LNet kernel module after it has been loaded via `modprobe`. In general the lnetctl format is as follows:

```
lnetctl cmd subcmd [options]
```

The following configuration items are managed by the tool:

- Configuring/unconfiguring LNet
- Adding/removing/showing Networks
- Adding/removing/showing Routes
- Enabling/Disabling routing
- Configuring Router Buffer Pools

### Configuring LNet

After LNet has been loaded via `modprobe`, `lnetctl` utility can be used to configure LNet without bringing up networks which are specified in the module parameters. It can also be used to configure network interfaces specified in the module prameters by providing the `--all` option.

```
lnetctl lnet configure [--all]
# --all: load NI configuration from module parameters
```

The `lnetctl` utility can also be used to unconfigure LNet.

```
lnetctl lnet unconfigure
```

### Displaying Global Settings

The active LNet global settings can be displayed using the `lnetctl` command shown below:

```
lnetctl global show
```

For example:

```
# lnetctl global show
        global:
        numa_range: 0
        max_intf: 200
        discovery: 1
        drop_asym_route: 0
```

### Adding, Deleting and Showing Networks

Networks can be added, deleted, or shown after the LNet kernel module is loaded.

The **lnetctl net add** command is used to add networks:

```
lnetctl net add: add a network
        --net: net name (ex tcp0)
        --if: physical interface (ex eth0)
        --peer_timeout: time to wait before declaring a peer dead
        --peer_credits: defines the max number of inflight messages
        --peer_buffer_credits: the number of buffer credits per peer
        --credits: Network Interface credits
        --cpts: CPU Partitions configured net uses
        --help: display this help text

Example:
lnetctl net add --net tcp2 --if eth0
                --peer_timeout 180 --peer_credits 8
```

Introduced in Lustre 2.10NoteWith the addition of Software based Multi-Rail in Lustre 2.10, the following should be noted:--net: no longer needs to be unique since multiple interfaces can be added to the same network.--if: The same interface per network can be added only once, however, more than one interface can now be specified (separated by a comma) for a node. For example: eth0,eth1,eth2.For examples on adding multiple interfaces via `lnetctl net add`and/or YAML, please see [the section called “Configuring Multi-Rail”](03.05-LNet%20Software%20Multi-Rail%202.10.md)

Networks can be deleted with the **lnetctl net del** command:

```
net del: delete a network
        --net: net name (ex tcp0)
        --if:  physical inerface (e.g. eth0)

Example:
lnetctl net del --net tcp2
```

Introduced in Lustre 2.10NoteIn a Software Multi-Rail configuration, specifying only the `--net`argument will delete the entire network and all interfaces under it. The new `--if` switch should also be used in conjunction with `--net` to specify deletion of a specific interface.

All or a subset of the configured networks can be shown with the **lnetctl net show**command. The output can be non-verbose or verbose.

```
net show: show networks
        --net: net name (ex tcp0) to filter on
        --verbose: display detailed output per network

Examples:
lnetctl net show
lnetctl net show --verbose
lnetctl net show --net tcp2 --verbose
```

Below are examples of non-detailed and detailed network configuration show.

```
# non-detailed show
> lnetctl net show --net tcp2
net:
    - nid: 192.168.205.130@tcp2
      status: up
      interfaces:
          0: eth3

# detailed show
> lnetctl net show --net tcp2 --verbose
net:
    - nid: 192.168.205.130@tcp2
      status: up
      interfaces:
          0: eth3
      tunables:
          peer_timeout: 180
          peer_credits: 8
          peer_buffer_credits: 0
          credits: 256
```

Introduced in Lustre 2.10

### Manual Adding, Deleting and Showing Peers

The **lnetctl peer add** command is used to manually add a remote peer to a software multi-rail configuration. For the dynamic peer discovery capability introduced in Lustre Release 2.11.0, please see [the section called “Dynamic Peer Discovery”](02.06-Configuring%20Lustre%20Networking%20(LNet).md#dynamic-peer-discovery).

When configuring peers, use the `–-prim_nid` option to specify the key or primary nid of the peer node. Then follow that with the `--nid` option to specify a set of comma separated NIDs.

```
peer add: add a peer
            --prim_nid: primary NID of the peer
            --nid: comma separated list of peer nids (e.g. 10.1.1.2@tcp0)
            --non_mr: if specified this interface is created as a non mulit-rail
            capable peer. Only one NID can be specified in this case.
```

For example:

```
            lnetctl peer add --prim_nid 10.10.10.2@tcp --nid 10.10.3.3@tcp1,10.4.4.5@tcp2
        
```

The `--prim-nid` (primary nid for the peer node) can go unspecified. In this case, the first listed NID in the `--nid` option becomes the primary nid of the peer. For example:

```
            lnetctl peer_add --nid 10.10.10.2@tcp,10.10.3.3@tcp1,10.4.4.5@tcp2
```

YAML can also be used to configure peers:

```
peer:
            - primary nid: <key or primary nid>
            Multi-Rail: True
            peer ni:
            - nid: <nid 1>
            - nid: <nid 2>
            - nid: <nid n>
```

As with all other commands, the result of the `lnetctl peer show` command can be used to gather information to aid in configuring or deleting a peer:

```
lnetctl peer show -v
```

Example output from the `lnetctl peer show` command:

```
peer:
            - primary nid: 192.168.122.218@tcp
            Multi-Rail: True
            peer ni:
            - nid: 192.168.122.218@tcp
            state: NA
            max_ni_tx_credits: 8
            available_tx_credits: 8
            available_rtr_credits: 8
            min_rtr_credits: -1
            tx_q_num_of_buf: 0
            send_count: 6819
            recv_count: 6264
            drop_count: 0
            refcount: 1
            - nid: 192.168.122.78@tcp
            state: NA
            max_ni_tx_credits: 8
            available_tx_credits: 8
            available_rtr_credits: 8
            min_rtr_credits: -1
            tx_q_num_of_buf: 0
            send_count: 7061
            recv_count: 6273
            drop_count: 0
            refcount: 1
            - nid: 192.168.122.96@tcp
            state: NA
            max_ni_tx_credits: 8
            available_tx_credits: 8
            available_rtr_credits: 8
            min_rtr_credits: -1
            tx_q_num_of_buf: 0
            send_count: 6939
            recv_count: 6286
            drop_count: 0
            refcount: 1
```

Use the following `lnetctl` command to delete a peer:

```
peer del: delete a peer
            --prim_nid: Primary NID of the peer
            --nid: comma separated list of peer nids (e.g. 10.1.1.2@tcp0)
```

`prim_nid` should always be specified. The `prim_nid` identifies the peer. If the`prim_nid` is the only one specified, then the entire peer is deleted.

Example of deleting a single nid of a peer (10.10.10.3@tcp):

```
lnetctl peer del --prim_nid 10.10.10.2@tcp --nid 10.10.10.3@tcp
```

Example of deleting the entire peer:

```
lnetctl peer del --prim_nid 10.10.10.2@tcp
```

Introduced in Lustre 2.11

### Dynamic Peer Discovery

#### Overview

Dynamic Discovery (DD) is a feature that allows nodes to dynamically discover a peer's interfaces without having to explicitly configure them. This is very useful for Multi-Rail (MR) configurations. In large clusters, there could be hundreds of nodes and having to configure MR peers on each node becomes error prone. Dynamic Discovery is enabled by default and uses a new protocol based on LNet pings to discover the interfaces of the remote peers on first message.

#### Protocol

When LNet on a node is requested to send a message to a peer it first attempts to ping the peer. The reply to the ping contains the peer's NIDs as well as a feature bit outlining what the peer supports. Dynamic Discovery adds a Multi-Rail feature bit. If the peer is Multi-Rail capable, it sets the MR bit in the ping reply. When the node receives the reply it checks the MR bit, and if it is set it then pushes its own list of NIDs to the peer using a new PUT message, referred to as a "push ping". After this brief protocol, both the peer and the node will have each other's list of interfaces. The MR algorithm can then proceed to use the list of interfaces of the corresponding peer.

If the peer is not MR capable, it will not set the MR feature bit in the ping reply. The node will understand that the peer is not MR capable and will only use the interface provided by upper layers for sending messages.

#### Dynamic Discovery and User-space Configuration

It is possible to configure the peer manually while Dynamic Discovery is running. Manual peer configuration always takes precedence over Dynamic Discovery. If there is a discrepancy between the manual configuration and the dynamically discovered information, a warning is printed.

#### Configuration

Dynamic Discovery is very light on the configuration side. It can only be turned on or turned off. To turn the feature on or off, the following command is used:

```
lnetctl set discovery [0 | 1]
```

To check the current `discovery` setting, the `lnetctl global show` command can be used as shown in [the section called “Displaying Global Settings”](02.06-Configuring%20Lustre%20Networking%20(LNet).md#displaying-global-settings).

#### Initiating Dynamic Discovery on Demand

It is possible to initiate the Dynamic Discovery protocol on demand without having to wait for a message to be sent to the peer. This can be done with the following command:

```
lnetctl discover <peer_nid> [<peer_nid> ...]
```

### Adding, Deleting and Showing routes

A set of routes can be added to identify how LNet messages are to be routed.

```
lnetctl route add: add a route
        --net: net name (ex tcp0) LNet message is destined to.
               The can not be a local network.
        --gateway: gateway node nid (ex 10.1.1.2@tcp) to route
                   all LNet messaged destined for the identified
                   network
        --hop: number of hops to final destination
               (1 < hops < 255)
        --priority: priority of route (0 - highest prio)

Example:
lnetctl route add --net tcp2 --gateway 192.168.205.130@tcp1 --hop 2 --prio 1
```

Routes can be deleted via the following `lnetctl` command.

```
lnetctl route del: delete a route
        --net: net name (ex tcp0)
        --gateway: gateway nid (ex 10.1.1.2@tcp)

Example:
lnetctl route del --net tcp2 --gateway 192.168.205.130@tcp1
```

Configured routes can be shown via the following `lnetctl` command.

```
lnetctl route show: show routes
        --net: net name (ex tcp0) to filter on
        --gateway: gateway nid (ex 10.1.1.2@tcp) to filter on
        --hop: number of hops to final destination
               (1 < hops < 255) to filter on
        --priority: priority of route (0 - highest prio)
                    to filter on
        --verbose: display detailed output per route

Examples:
# non-detailed show
lnetctl route show

# detailed show
lnetctl route show --verbose
```

When showing routes the `--verbose` option outputs more detailed information. All show and error output are in YAML format. Below are examples of both non-detailed and detailed route show output.

```
#Non-detailed output
> lnetctl route show
route:
    - net: tcp2
      gateway: 192.168.205.130@tcp1

#detailed output
> lnetctl route show --verbose
route:
    - net: tcp2
      gateway: 192.168.205.130@tcp1
      hop: 2
      priority: 1
      state: down
```

### Enabling and Disabling Routing

When an LNet node is configured as a router it will route LNet messages not destined to itself. This feature can be enabled or disabled as follows.

```
lnetctl set routing [0 | 1]
# 0 - disable routing feature
# 1 - enable routing feature
```

### Showing routing information

When routing is enabled on a node, the tiny, small and large routing buffers are allocated. See [the section called “ Tuning LNet Parameters”](04.03-Tuning%20a%20Lustre%20File%20System.md#tuning-lnet-parameters) for more details on router buffers. This information can be shown as follows:

```
lnetctl routing show: show routing information

Example:
lnetctl routing show
```

An example of the show output:

```
> lnetctl routing show
routing:
    - cpt[0]:
          tiny:
              npages: 0
              nbuffers: 2048
              credits: 2048
              mincredits: 2048
          small:
              npages: 1
              nbuffers: 16384
              credits: 16384
              mincredits: 16384
          large:
              npages: 256
              nbuffers: 1024
              credits: 1024
              mincredits: 1024
    - enable: 1
```

### Configuring Routing Buffers

The routing buffers values configured specify the number of buffers in each of the tiny, small and large groups.

It is often desirable to configure the tiny, small and large routing buffers to some values other than the default. These values are global values, when set they are used by all configured CPU partitions. If routing is enabled then the values set take effect immediately. If a larger number of buffers is specified, then buffers are allocated to satisfy the configuration change. If fewer buffers are configured then the excess buffers are freed as they become unused. If routing is not set the values are not changed. The buffer values are reset to default if routing is turned off and on.

The `lnetctl` 'set' command can be used to set these buffer values. A VALUE greater than 0 will set the number of buffers accordingly. A VALUE of 0 will reset the number of buffers to system defaults.

```
set tiny_buffers:
      set tiny routing buffers
               VALUE must be greater than or equal to 0

set small_buffers: set small routing buffers
        VALUE must be greater than or equal to 0

set large_buffers: set large routing buffers
        VALUE must be greater than or equal to 0
```

Usage examples:

```
> lnetctl set tiny_buffers 4096
> lnetctl set small_buffers 8192
> lnetctl set large_buffers 2048
```

The buffers can be set back to the default values as follows:

```
> lnetctl set tiny_buffers 0
> lnetctl set small_buffers 0
> lnetctl set large_buffers 0
```

Introduced in Lustre 2.13

### Asymmetrical Routes

#### Overview

An asymmetrical route is when a message from a remote peer is coming through a router that is not known by this node to reach the remote peer.

Asymmetrical routes can be an issue when debugging network, and allowing them also opens the door to attacks where hostile clients inject data to the servers.

So it is possible to activate a check in LNet, that will detect any asymmetrical route message and drop it.

#### Configuration

In order to switch asymmetric route detection on or off, the following command is used:

```
lnetctl set drop_asym_route [0 | 1]
```

This command works on a per-node basis. This means each node in a Lustre cluster can decide whether it accepts asymmetrical route messages.

To check the current `drop_asym_route` setting, the `lnetctl global show` command can be used as shown in [the section called “Displaying Global Settings”](#displaying-global-settings).

By default, asymmetric route detection is off.

### Importing YAML Configuration File

Configuration can be described in YAML format and can be fed into the `lnetctl`utility. The `lnetctl` utility parses the YAML file and performs the specified operation on all entities described there in. If no operation is defined in the command as shown below, the default operation is 'add'. The YAML syntax is described in a later section.

```
lnetctl import FILE.yaml
lnetctl import < FILE.yaml
```

The '`lnetctl` import' command provides three optional parameters to define the operation to be performed on the configuration items described in the YAML file.

```
# if no options are given to the command the "add" command is assumed
              # by default.
lnetctl import --add FILE.yaml
lnetctl import --add < FILE.yaml

# to delete all items described in the YAML file
lnetctl import --del FILE.yaml
lnetctl import --del < FILE.yaml

# to show all items described in the YAML file
lnetctl import --show FILE.yaml
lnetctl import --show < FILE.yaml
```

### Exporting Configuration in YAML format

`lnetctl` utility provides the 'export' command to dump current LNet configuration in YAML format

```
lnetctl export FILE.yaml
lnetctl export > FILE.yaml
```

### Showing LNet Traffic Statistics

`lnetctl` utility can dump the LNet traffic statistiscs as follows

```
lnetctl stats show
```

### YAML Syntax

The `lnetctl` utility can take in a YAML file describing the configuration items that need to be operated on and perform one of the following operations: add, delete or show on the items described there in.

Net, routing and route YAML blocks are all defined as a YAML sequence, as shown in the following sections. The stats YAML block is a YAML object. Each sequence item can take a seq_no field. This seq_no field is returned in the error block. This allows the caller to associate the error with the item that caused the error. The `lnetctl`utilty does a best effort at configuring items defined in the YAML file. It does not stop processing the file at the first error.

Below is the YAML syntax describing the various configuration elements which can be operated on via DLC. Not all YAML elements are required for all operations (add/delete/show). The system ignores elements which are not pertinent to the requested operation.

#### Network Configuration

```
net:
   - net: <network.  Ex: tcp or o2ib>
     interfaces:
         0: <physical interface>
     detail: <This is only applicable for show command.  1 - output detailed info.  0 - basic output>
     tunables:
        peer_timeout: <Integer. Timeout before consider a peer dead>
        peer_credits: <Integer. Transmit credits for a peer>
        peer_buffer_credits: <Integer. Credits available for receiving messages>
        credits: <Integer.  Network Interface credits>
	SMP: <An array of integers of the form: "[x,y,...]", where each
	integer represents the CPT to associate the network interface
	with> seq_no: <integer.  Optional.  User generated, and is
	passed back in the YAML error block>
```

Both seq_no and detail fields do not appear in the show output.

#### Enable Routing and Adjust Router Buffer Configuration

```
routing:
    - tiny: <Integer. Tiny buffers>
      small: <Integer. Small buffers>
      large: <Integer. Large buffers>
      enable: <0 - disable routing.  1 - enable routing>
      seq_no: <Integer.  Optional.  User generated, and is passed back in the YAML error block>
```

The seq_no field does not appear in the show output

#### Show Statistics

```
statistics:
    seq_no: <Integer. Optional.  User generated, and is passed back in the YAML error block>
```

The seq_no field does not appear in the show output

#### Route Configuration

```
route:
  - net: <network. Ex: tcp or o2ib>
    gateway: <nid of the gateway in the form <ip>@<net>: Ex: 192.168.29.1@tcp>
    hop: <an integer between 1 and 255. Optional>
    detail: <This is only applicable for show commands.  1 - output detailed info.  0. basic output>
    seq_no: <integer. Optional. User generated, and is passed back in the YAML error block>
```

Both seq_no and detail fields do not appear in the show output.

## Overview of LNet Module Parameters

LNet kernel module (lnet) parameters specify how LNet is to be configured to work with Lustre, including which NICs will be configured to work with Lustre and the routing to be used with Lustre.

Parameters for LNet can be specified in the `/etc/modprobe.d/lustre.conf` file. In some cases the parameters may have been stored in `/etc/modprobe.conf`, but this has been deprecated since before RHEL5 and SLES10, and having a separate`/etc/modprobe.d/lustre.conf` file simplifies administration and distribution of the Lustre networking configuration. This file contains one or more entries with the syntax:

```
options lnet parameter=value
```

To specify the network interfaces that are to be used for Lustre, set either the `networks` parameter or the `ip2nets` parameter (only one of these parameters can be used at a time):

- `networks` - Specifies the networks to be used.
- `ip2nets` - Lists globally-available networks, each with a range of IP addresses. LNet then identifies locally-available networks through address list-matching lookup.

See [the section called “Setting the LNet Module networks Parameter”](#setting-the-lnet-module-networks-parameter) and [the section called “Setting the LNet Module ip2nets Parameter”](#setting-the-lnet-module-ip2nets-parameter) for more details.

To set up routing between networks, use:

- `routes` - Lists networks and the NIDs of routers that forward to them.

See [the section called “Setting the LNet Module routes Parameter”](#setting-the-lnet-module-routes-parameter) for more details.

A `router` checker can be configured to enable Lustre nodes to detect router health status, avoid routers that appear dead, and reuse those that restore service after failures. See [the section called “Configuring the Router Checker”](#configuring-the-router-checker) for more details.

For a complete reference to the LNet module parameters, see *Configuration Files and Module ParametersLNet Options*.

### Note

We recommend that you use 'dotted-quad' notation for IP addresses rather than host names to make it easier to read debug logs and debug configurations with multiple interfaces.

### Using a Lustre Network Identifier (NID) to Identify a Node

A Lustre network identifier (NID) is used to uniquely identify a Lustre network endpoint by node ID and network type. The format of the NID is:

```
network_id@network_type
```

Examples are:

```
10.67.73.200@tcp0
10.67.75.100@o2ib
```

The first entry above identifies a TCP/IP node, while the second entry identifies an InfiniBand node.

When a mount command is run on a client, the client uses the NID of the MDS to retrieve configuration information. If an MDS has more than one NID, the client should use the appropriate NID for its local network.

To determine the appropriate NID to specify in the mount command, use the `lctl`command. To display MDS NIDs, run on the MDS :

```
lctl list_nids
```

To determine if a client can reach the MDS using a particular NID, run on the client:

```
lctl which_nid MDS_NID
```

## Setting the LNet Module networks Parameter

If a node has more than one network interface, you'll typically want to dedicate a specific interface to Lustre. You can do this by including an entry in the `lustre.conf`file on the node that sets the LNet module `networks` parameter:

```
options lnet networks=comma-separated list of
    networks
```

This example specifies that a Lustre node will use a TCP/IP interface and an InfiniBand interface:

```
options lnet networks=tcp0(eth0),o2ib(ib0)
```

This example specifies that the Lustre node will use the TCP/IP interface `eth1`:

```
options lnet networks=tcp0(eth1)
```

Depending on the network design, it may be necessary to specify explicit interfaces. To explicitly specify that interface `eth2` be used for network `tcp0` and `eth3` be used for `tcp1` , use this entry:

```
options lnet networks=tcp0(eth2),tcp1(eth3)
```

When more than one interface is available during the network setup, Lustre chooses the best route based on the hop count. Once the network connection is established, Lustre expects the network to stay connected. In a Lustre network, connections do not fail over to another interface, even if multiple interfaces are available on the same node.

### Note

LNet lines in `lustre.conf` are only used by the local node to determine what to call its interfaces. They are not used for routing decisions.

### Multihome Server Example

If a server with multiple IP addresses (multihome server) is connected to a Lustre network, certain configuration setting are required. An example illustrating these setting consists of a network with the following nodes:

- Server svr1 with three TCP NICs (`eth0`, `eth1`, and `eth2`) and an InfiniBand NIC.
- Server svr2 with three TCP NICs (`eth0`, `eth1`, and `eth2`) and an InfiniBand NIC. Interface eth2 will not be used for Lustre networking.
- TCP clients, each with a single TCP interface.
- InfiniBand clients, each with a single Infiniband interface and a TCP/IP interface for administration.

To set the `networks` option for this example:

- On each server, `svr1` and `svr2`, include the following line in the `lustre.conf`file:

```
options lnet networks=tcp0(eth0),tcp1(eth1),o2ib
```

- For TCP-only clients, the first available non-loopback IP interface is used for `tcp0`. Thus, TCP clients with only one interface do not need to have options defined in the `lustre.conf` file.
- On the InfiniBand clients, include the following line in the `lustre.conf` file:

```
options lnet networks=o2ib
```

### Note

By default, Lustre ignores the loopback (`lo0`) interface. Lustre does not ignore IP addresses aliased to the loopback. If you alias IP addresses to the loopback interface, you must specify all Lustre networks using the LNet networks parameter.

### Note

If the server has multiple interfaces on the same subnet, the Linux kernel will send all traffic using the first configured interface. This is a limitation of Linux, not Lustre. In this case, network interface bonding should be used. For more information about network interface bonding, see [*Setting Up Network Interface Bonding*](02.04-Setting%20Up%20Network%20Interface%20Bonding.md).

## Setting the LNet Module ip2nets Parameter

The `ip2nets` option is typically used when a single, universal `lustre.conf` file is run on all servers and clients. Each node identifies the locally available networks based on the listed IP address patterns that match the node's local IP addresses.

Note that the IP address patterns listed in the `ip2nets` option are *only* used to identify the networks that an individual node should instantiate. They are *not* used by LNet for any other communications purpose.

For the example below, the nodes in the network have these IP addresses:

- Server svr1: `eth0` IP address `192.168.0.2`, IP over Infiniband (`o2ib`) address`132.6.1.2`.
- Server svr2: `eth0` IP address `192.168.0.4`, IP over Infiniband (`o2ib`) address`132.6.1.4`.
- TCP clients have IP addresses `192.168.0.5-255.`
- Infiniband clients have IP over Infiniband (`o2ib`) addresses `132.6.[2-3].2, .4, .6, .8`.

The following entry is placed in the `lustre.conf` file on each server and client:

```
options lnet 'ip2nets="tcp0(eth0) 192.168.0.[2,4]; \
tcp0 192.168.0.*; o2ib0 132.6.[1-3].[2-8/2]"'
```

Each entry in `ip2nets` is referred to as a 'rule'.

The order of LNet entries is important when configuring servers. If a server node can be reached using more than one network, the first network specified in `lustre.conf` will be used.

Because `svr1` and `svr2` match the first rule, LNet uses `eth0` for `tcp0` on those machines. (Although `svr1` and `svr2` also match the second rule, the first matching rule for a particular network is used).

The `[2-8/2]` format indicates a range of 2-8 stepped by 2; that is 2,4,6,8. Thus, the clients at `132.6.3.5` will not find a matching o2ib network.

Introduced in Lustre 2.10NoteMulti-rail deprecates the kernel parsing of ip2nets. ip2nets patterns are matched in user space and translated into Network interfaces to be added into the system.The first interface that matches the IP pattern will be used when adding a network interface.If an interface is explicitly specified as well as a pattern, the interface matched using the IP pattern will be sanitized against the explicitly-defined interface.For example, `tcp(eth0) 192.168.*.3` and there exists in the system `eth0 == 192.158.19.3` and `eth1 == 192.168.3.3`, then the configuration will fail, because the pattern contradicts the interface specified.A clear warning will be displayed if inconsistent configuration is encountered.You could use the following command to configure ip2nets:`lnetctl import < ip2nets.yaml`For example:`ip2nets:   - net-spec: tcp1     interfaces:          0: eth0          1: eth1     ip-range:          0: 192.168.*.19          1: 192.168.100.105   - net-spec: tcp2     interfaces:          0: eth2     ip-range:          0: 192.168.*.*`

## Setting the LNet Module routes Parameter

The LNet module routes parameter is used to identify routers in a Lustre configuration. These parameters are set in `modprobe.conf` on each Lustre node.

Routes are typically set to connect to segregated subnetworks or to cross connect two different types of networks such as tcp and o2ib

The LNet routes parameter specifies a colon-separated list of router definitions. Each route is defined as a network number, followed by a list of routers:

```
routes=net_type router_NID(s)
```

This example specifies bi-directional routing in which TCP clients can reach Lustre resources on the IB networks and IB servers can access the TCP networks:

```
options lnet 'ip2nets="tcp0 192.168.0.*; \
  o2ib0(ib0) 132.6.1.[1-128]"' 'routes="tcp0   132.6.1.[1-8]@o2ib0; \
  o2ib0 192.16.8.0.[1-8]@tcp0"'
```

All LNet routers that bridge two networks are equivalent. They are not configured as primary or secondary, and the load is balanced across all available routers.

The number of LNet routers is not limited. Enough routers should be used to handle the required file serving bandwidth plus a 25 percent margin for headroom.

### Routing Example

On the clients, place the following entry in the `lustre.conf` file

```
lnet networks="tcp" routes="o2ib0 192.168.0.[1-8]@tcp0"
```

On the router nodes, use:

```
lnet networks="tcp o2ib" forwarding=enabled 
```

On the MDS, use the reverse as shown below:

```
lnet networks="o2ib0" routes="tcp0 132.6.1.[1-8]@o2ib0" 
```

To start the routers, run:

```
modprobe lnet
lctl network configure
```

## Testing the LNet Configuration

After configuring Lustre Networking, it is highly recommended that you test your LNet configuration using the LNet Self-Test provided with the Lustre software. For more information about using LNet Self-Test, see [*Testing Lustre Network Performance (LNet Self-Test)*](04.01-Testing%20Lustre%20Network%20Performance%20(LNet%20Self-Test).md).

## Configuring the Router Checker

In a Lustre configuration in which different types of networks, such as a TCP/IP network and an Infiniband network, are connected by routers, a router checker can be run on the clients and servers in the routed configuration to monitor the status of the routers. In a multi-hop routing configuration, router checkers can be configured on routers to monitor the health of their next-hop routers.

A router checker is configured by setting LNet parameters in `lustre.conf` by including an entry in this form:

```
options lnet
    router_checker_parameter=value
```

The router checker parameters are:

- `live_router_check_interval` - Specifies a time interval in seconds after which the router checker will ping the live routers. The default value is 0, meaning no checking is done. To set the value to 60, enter:

  ```
  options lnet live_router_check_interval=60
  ```

- `dead_router_check_interval` - Specifies a time interval in seconds after which the router checker will check for dead routers. The default value is 0, meaning no checking is done. To set the value to 60, enter:

  ```
  options lnet dead_router_check_interval=60
  ```

- auto_down - Enables/disables (1/0) the automatic marking of router state as up or down. The default value is 1. To disable router marking, enter:

  ```
  options lnet auto_down=0
  ```

- `router_ping_timeout` - Specifies a timeout for the router checker when it checks live or dead routers. The router checker sends a ping message to each dead or live router once every dead_router_check_interval or live_router_check_interval respectively. The default value is 50. To set the value to 60, enter:

  ```
  options lnet router_ping_timeout=60
  ```

  ### Note

  The `router_ping_timeout` is consistent with the default LND timeouts. You may have to increase it on very large clusters if the LND timeout is also increased. For larger clusters, we suggest increasing the check interval.

- `check_routers_before_use` - Specifies that routers are to be checked before use. Set to off by default. If this parameter is set to on, the dead_router_check_interval parameter must be given a positive integer value.

  ```
  options lnet check_routers_before_use=on
  ```

The router checker obtains the following information from each router:

- Time the router was disabled
- Elapsed disable time

If the router checker does not get a reply message from the router within router_ping_timeout seconds, it considers the router to be down.

If a router is marked 'up' and responds to a ping, the timeout is reset.

If 100 packets have been sent successfully through a router, the sent-packets counter for that router will have a value of 100.

## Best Practices for LNet Options

For the `networks`, `ip2nets`, and `routes` options, follow these best practices to avoid configuration errors.

### Escaping commas with quotes

Depending on the Linux distribution, commas may need to be escaped using single or double quotes. In the extreme case, the `options` entry would look like this:

```
options
      lnet'networks="tcp0,elan0"'
      'routes="tcp [2,10]@elan0"'
```

Added quotes may confuse some distributions. Messages such as the following may indicate an issue related to added quotes:

```
lnet: Unknown parameter 'networks'
```

A `'Refusing connection - no matching NID'` message generally points to an error in the LNet module configuration.

### Including comments

*Place the semicolon terminating a comment immediately after the comment.* LNet silently ignores everything between the `#` character at the beginning of the comment and the next semicolon.

In this *incorrect* example, LNet silently ignores `pt11 192.168.0.[92,96]`, resulting in these nodes not being properly initialized. No error message is generated.

```
options lnet ip2nets="pt10 192.168.0.[89,93]; # comment
      with semicolon BEFORE comment \ pt11 192.168.0.[92,96];
```

This correct example shows the required syntax:

```
options lnet ip2nets="pt10 192.168.0.[89,93] \
# comment with semicolon AFTER comment; \
pt11 192.168.0.[92,96] # comment
```

Do not add an excessive number of comments. The Linux kernel limits the length of character strings used in module options (usually to 1KB, but this may differ between vendor kernels). If you exceed this limit, errors result and the specified configuration may not be processed correctly.


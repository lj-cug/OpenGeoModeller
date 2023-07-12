# Managing Lustre Networking (LNet)

- [Managing Lustre Networking (LNet)](#managing-lustre-networking-lnet)
  * [Updating the Health Status of a Peer or Router](#updating-the-health-status-of-a-peer-or-router)
  * [Starting and Stopping LNet](#starting-and-stopping-lnet)
    + [Starting LNet](#starting-lnet)
      - [Starting Clients](#starting-clients)
    + [Stopping LNet](#stopping-lnet)
  * [Hardware Based Multi-Rail Configurations with LNet](#hardware-based-multi-rail-configurations-with-lnet)
  * [Load Balancing with an InfiniBand* Network](#load-balancing-with-an-infiniband-network)
    + [Setting Up `lustre.conf` for Load Balancing](#setting-up-lustreconf-for-load-balancing)
  * [Dynamically Configuring LNet Routes](#dynamically-configuring-lnet-routes)L 2.4
    + [`lustre_routes_config`](#lustre_routes_config)
    + [`lustre_routes_conversion`](#lustre_routes_conversion)
    + [`Route Configuration Examples`](#route-configuration-examples)

This chapter describes some tools for managing Lustre networking (LNet) and includes the following sections:

* [the section called “Updating the Health Status of a Peer or Router”](#updating-the-health-status-of-a-peer-or-router)

* [the section called “Starting and Stopping LNet”](#starting-and-stopping-lnet)

* [the section called “Hardware Based Multi-Rail Configurations with LNet”](#hardware-based-multi-rail-configurations-with-lnet)

* [the section called “Load Balancing with an InfiniBand* Network”](#load-balancing-with-an-infiniband-network)

* [the section called “Dynamically Configuring LNet Routes”](#dynamically-configuring-lnet-routes)



## Updating the Health Status of a Peer or Router

There are two mechanisms to update the health status of a peer or a router:

- LNet can actively check health status of all routers and mark them as dead or alive automatically. By default, this is off. To enable it set `auto_down` and if desired `check_routers_before_use`. This initial check may cause a pause equal to `router_ping_timeout` at system startup, if there are dead routers in the system.
- When there is a communication error, all LNDs notify LNet that the peer (not necessarily a router) is down. This mechanism is always on, and there is no parameter to turn it off. However, if you set the LNet module parameter `auto_down` to `0`, LNet ignores all such peer-down notifications.

Several key differences in both mechanisms:

- The router pinger only checks routers for their health, while LNDs notices all dead peers, regardless of whether they are a router or not.
- The router pinger actively checks the router health by sending pings, but LNDs only notice a dead peer when there is network traffic going on.
- The router pinger can bring a router from alive to dead or vice versa, but LNDs can only bring a peer down.

 

## Starting and Stopping LNet

The Lustre software automatically starts and stops LNet, but it can also be manually started in a standalone manner. This is particularly useful to verify that your networking setup is working correctly before you attempt to start the Lustre file system.

### Starting LNet

To start LNet, run:

```
$ modprobe lnet
$ lctl network up
```

To see the list of local NIDs, run:

```
$ lctl list_nids
```

This command tells you the network(s) configured to work with the Lustre file system.

If the networks are not correctly setup, see the `modules.conf` "`networks=`" line and make sure the network layer modules are correctly installed and configured.

To get the best remote NID, run:

```
$ lctl which_nid NIDs
```

where `*NIDs*` is the list of available NIDs.

This command takes the "best" NID from a list of the NIDs of a remote host. The "best" NID is the one that the local node uses when trying to communicate with the remote node.

#### Starting Clients

To start a TCP client, run:

```
mount -t lustre mdsnode:/mdsA/client /mnt/lustre/
```

To start an Elan client, run:

```
mount -t lustre 2@elan0:/mdsA/client /mnt/lustre
```

### Stopping LNet

Before the LNet modules can be removed, LNet references must be removed. In general, these references are removed automatically when the Lustre file system is shut down, but for standalone routers, an explicit step is needed to stop LNet. Run:

```
lctl network unconfigure
```

**Note**

Attempting to remove Lustre modules prior to stopping the network may result in a crash or an LNet hang. If this occurs, the node must be rebooted (in most cases). Make sure that the Lustre network and Lustre file system are stopped prior to unloading the modules. Be extremely careful using `rmmod -f`.

To unconfigure the LNet network, run:

```
modprobe -r lnd_and_lnet_modules
```

**Note**

To remove all Lustre modules, run:

`$ lustre_rmmod`

## Hardware Based Multi-Rail Configurations with LNet

To aggregate bandwidth across both rails of a dual-rail IB cluster (o2iblnd) [1]) using LNet, consider these points:

- LNet can work with multiple rails, however, it does not load balance across them. The actual rail used for any communication is determined by the peer NID.
- Hardware multi-rail LNet configurations do not provide an additional level of network fault tolerance. The configurations described below are for bandwidth aggregation only.
- A Lustre node always uses the same local NID to communicate with a given peer NID. The criteria used to determine the local NID are:
  - Introduced in Lustre 2.5Lowest route priority number (lower number, higher priority).
  - Fewest hops (to minimize routing), and
  - Appears first in the "`networks`" or "`ip2nets`" LNet configuration strings

----------

[1]Hardware multi-rail configurations are only supported by o2iblnd; other IB LNDs do not support multiple interfaces.

-------------------

## Load Balancing with an InfiniBand* Network

A Lustre file system contains OSSs with two InfiniBand HCAs. Lustre clients have only one InfiniBand HCA using OFED-based Infiniband ''o2ib'' drivers. Load balancing between the HCAs on the OSS is accomplished through LNet.

### Setting Up `lustre.conf` for Load Balancing

To configure LNet for load balancing on clients and servers:

1. Set the `lustre.conf` options.

   Depending on your configuration, set `lustre.conf` options as follows:

   - Dual HCA OSS server

   ```
   options lnet networks="o2ib0(ib0),o2ib1(ib1)"
   ```

   - Client with the odd IP address

   ```
   options lnet ip2nets="o2ib0(ib0) 192.168.10.[103-253/2]"
   ```

   - Client with the even IP address

   ```
   options lnet ip2nets="o2ib1(ib0) 192.168.10.[102-254/2]"
   ```

2. Run the modprobe lnet command and create a combined MGS/MDT file system.

   The following commands create an MGS/MDT or OST file system and mount the targets on the servers.

   ```
   modprobe lnet
   # mkfs.lustre --fsname lustre --mgs --mdt /dev/mdt_device
   # mkdir -p /mount_point
   # mount -t lustre /dev/mdt_device /mount_point
   ```

   For example:

   ```
   modprobe lnet
   mds# mkfs.lustre --fsname lustre --mdt --mgs /dev/sda
   mds# mkdir -p /mnt/test/mdt
   mds# mount -t lustre /dev/sda /mnt/test/mdt   
   mds# mount -t lustre mgs@o2ib0:/lustre /mnt/mdt
   oss# mkfs.lustre --fsname lustre --mgsnode=mds@o2ib0 --ost --index=0 /dev/sda
   oss# mkdir -p /mnt/test/mdt
   oss# mount -t lustre /dev/sda /mnt/test/ost   
   oss# mount -t lustre mgs@o2ib0:/lustre /mnt/ost0
   ```

3. Mount the clients.

   ```
   client# mount -t lustre mgs_node:/fsname /mount_point
   ```

   This example shows an IB client being mounted.

   ```
   client# mount -t lustre
   192.168.10.101@o2ib0,192.168.10.102@o2ib1:/mds/client /mnt/lustre
   ```

As an example, consider a two-rail IB cluster running the OFED stack with these IPoIB address assignments.

```
             ib0                             ib1
Servers            192.168.0.*                     192.168.1.*
Clients            192.168.[2-127].*               192.168.[128-253].*
```

You could create these configurations:

- A cluster with more clients than servers. The fact that an individual client cannot get two rails of bandwidth is unimportant because the servers are typically the actual bottleneck.

```
ip2nets="o2ib0(ib0),    o2ib1(ib1)      192.168.[0-1].*                     \
                                            #all servers;\
                   o2ib0(ib0)      192.168.[2-253].[0-252/2]       #even cl\
ients;\
                   o2ib1(ib1)      192.168.[2-253].[1-253/2]       #odd cli\
ents"
```

This configuration gives every server two NIDs, one on each network, and statically load-balances clients between the rails.

- A single client that must get two rails of bandwidth, and it does not matter if the maximum aggregate bandwidth is only (# servers) * (1 rail).

```
ip2nets="       o2ib0(ib0)                      192.168.[0-1].[0-252/2]     \
                                            #even servers;\
           o2ib1(ib1)                      192.168.[0-1].[1-253/2]         \
                                        #odd servers;\
           o2ib0(ib0),o2ib1(ib1)           192.168.[2-253].*               \
                                        #clients"
```

This configuration gives every server a single NID on one rail or the other. Clients have a NID on both rails.

- All clients and all servers must get two rails of bandwidth.

```
ip2nets=â€   o2ib0(ib0),o2ib2(ib1)           192.168.[0-1].[0-252/2]       \
  #even servers;\
           o2ib1(ib0),o2ib3(ib1)           192.168.[0-1].[1-253/2]         \
#odd servers;\
           o2ib0(ib0),o2ib3(ib1)           192.168.[2-253].[0-252/2)       \
#even clients;\
           o2ib1(ib0),o2ib2(ib1)           192.168.[2-253].[1-253/2)       \
#odd clients"
```

This configuration includes two additional proxy o2ib networks to work around the simplistic NID selection algorithm in the Lustre software. It connects "even" clients to "even" servers with `o2ib0` on `rail0`, and "odd" servers with `o2ib3` on `rail1`. Similarly, it connects "odd" clients to "odd" servers with `o2ib1` on `rail0`, and "even" servers with `o2ib2` on `rail1`.



Introduced in Lustre 2.4

## Dynamically Configuring LNet Routes

Two scripts are provided: `lustre/scripts/lustre_routes_config` and `lustre/scripts/lustre_routes_conversion`.

`lustre_routes_config` sets or cleans up LNet routes from the specified config file. The `/etc/sysconfig/lnet_routes.conf` file can be used to automatically configure routes on LNet startup.

`lustre_routes_conversion` converts a legacy routes configuration file to the new syntax, which is parsed by `lustre_routes_config`

### `lustre_routes_config`

`lustre_routes_config` usage is as follows

```
lustre_routes_config [--setup|--cleanup|--dry-run|--verbose] config_file
         --setup: configure routes listed in config_file
         --cleanup: unconfigure routes listed in config_file
         --dry-run: echo commands to be run, but do not execute them
         --verbose: echo commands before they are executed 
```

The format of the file which is passed into the script is as follows:

`*network*: { gateway: *gateway*@*exit_network* [hop: *hop*] [priority: *priority*] }`

An LNet router is identified when its local NID appears within the list of routes. However, this can not be achieved by the use of this script, since the script only adds extra routes after the router is identified. To ensure that a router is identified correctly, make sure to add its local NID in the routes parameter in the modprobe lustre configuration file. See [*the section called “ Introduction”*](06.06-Configuration%20Files%20and%20Module%20Parameters.md#introduction).

### `lustre_routes_conversion`

`lustre_routes_conversion` usage is as follows:

```
lustre_routes_conversion legacy_file new_file
```

`lustre_routes_conversion` takes as a first parameter a file with routes configured as follows:

`*network* [*hop*] *gateway*@*exit network*[:*priority*];`

The script then converts each routes entry in the provided file to:

`*network*: { gateway: *gateway*@*exit network* [hop: *hop*] [priority: *priority*] }`

and appends each converted entry to the output file passed in as the second parameter to the script.

### `Route Configuration Examples`

Below is an example of a legacy LNet route configuration. A legacy configuration file can have multiple entries.

```
tcp1 10.1.1.2@tcp0:1;
tcp2 10.1.1.3@tcp0:2;
tcp3 10.1.1.4@tcp0;
```

Below is an example of the converted LNet route configuration. The following would be the result of the `lustre_routes_conversion` script, when run on the above legacy entries.

```
tcp1: { gateway: 10.1.1.2@tcp0 priority: 1 }
tcp2: { gateway: 10.1.1.2@tcp0 priority: 2 }
tcp1: { gateway: 10.1.1.4@tcp0 }
```

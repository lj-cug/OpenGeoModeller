# Testing Lustre Network Performance (LNet Self-Test)

- [Testing Lustre Network Performance (LNet Self-Test)](#testing-lustre-network-performance-lnet-self-test)
  * [LNet Self-Test Overview](#lnet-self-test-overview)
    + [Prerequisites](#prerequisites)
  * [Using LNet Self-Test](#using-lnet-self-test)
    + [Creating a Session](#creating-a-session)
    + [Setting Up Groups](#setting-up-groups)
    + [Defining and Running the Tests](#defining-and-running-the-tests)
    + [Sample Script](#sample-script)
  * [LNet Self-Test Command Reference](#lnet-self-test-command-reference)
    + [Session Commands](#session-commands)
    + [Group Commands](#group-commands)
    + [Batch and Test Commands](#batch-and-test-commands)
    + [Other Commands](#other-commands)

This chapter describes the LNet self-test, which is used by site administrators to confirm that Lustre Networking (LNet) has been properly installed and configured, and that underlying network software and hardware are performing according to expectations. The chapter includes:

- [the section called “ LNet Self-Test Overview”](#lnet-self-test-overview)
- [the section called “Using LNet Self-Test”](#using-lnet-self-test)
- [the section called “LNet Self-Test Command Reference”](#lnet-self-test-command-reference)

## LNet Self-Test Overview

LNet self-test is a kernel module that runs over LNet and the Lustre network drivers (LNDs). It is designed to:

- Test the connection ability of the Lustre network
- Run regression tests of the Lustre network
- Test performance of the Lustre network

After you have obtained performance results for your Lustre network, refer to [*Tuning a Lustre File System*](04.03-Tuning%20a%20Lustre%20File%20System.md) for information about parameters that can be used to tune LNet for optimum performance.

**Note**

Apart from the performance impact, LNet self-test is invisible to the Lustre file system.

An LNet self-test cluster includes two types of nodes:

- **Console node** - A node used to control and monitor an LNet self-test cluster. The console node serves as the user interface of the LNet self-test system and can be any node in the test cluster. All self-test commands are entered from the console node. From the console node, a user can control and monitor the status of the entire LNet self-test cluster (session). The console node is exclusive in that a user cannot control two different sessions from one console node.
- **Test nodes** - The nodes on which the tests are run. Test nodes are controlled by the user from the console node; the user does not need to log into them directly.

LNet self-test has two user utilities:

- **lst** - The user interface for the self-test console (run on the *console node*). It provides a list of commands to control the entire test system, including commands to create a session, create test groups, etc.
- **lstclient** - The userspace LNet self-test program (run on a *test node*). The `lstclient` utility is linked with userspace LNDs and LNet. This utility is not needed if only kernel space LNet and LNDs are used.

**Note**

*Test nodes* can be in either kernel or userspace. A *console node* can invite a kernel *test node* to join the session by running `lst add_group NID`, but the *console node* cannot actively add a userspace *test node* to the session. A *console node* can passively accept a *test node* to the session while the *test node* is running `lstclient` to connect to the *console node*.

### Prerequisites

To run LNet self-test, these modules must be loaded on both *console nodes* and *test nodes*:

- `libcfs`
- `net`
- `lnet_selftest`
- `klnds`: A kernel Lustre network driver (LND) (i.e, `ksocklnd`, `ko2iblnd`...) as needed by your network configuration.

To load the required modules, run:

```
modprobe lnet_selftest 
```

This command recursively loads the modules on which LNet self-test depends.

**Note**

While the *console node* and *test nodes* require all the prerequisite modules to be loaded, userspace test nodes do not require these modules.

## Using LNet Self-Test

This section describes how to create and run an LNet self-test. The examples shown are for a test that simulates the traffic pattern of a set of Lustre servers on a TCP network accessed by Lustre clients on an InfiniBand network connected via LNet routers. In this example, half the clients are reading and half the clients are writing.

### Creating a Session

A *session* is a set of processes that run on a *test node*. Only one session can be run at a time on a test node to ensure that the session has exclusive use of the node. The console node is used to create, change or destroy a session (`new_session`, `end_session`, `show_session`). For more about session parameters, see [*the section called “Session Commands”*](#session-commands).

Almost all operations should be performed within the context of a session. From the *console node*, a user can only operate nodes in his own session. If a session ends, the session context in all test nodes is stopped.

The following commands set the `LST_SESSION` environment variable to identify the session on the console node and create a session called `read_write`:

```
export LST_SESSION=$$
lst new_session read_write
```

### Setting Up Groups

A *group* is a named collection of nodes. Any number of groups can exist in a single LNet self-test session. Group membership is not restricted in that a *test node* can be included in any number of groups.

Each node in a group has a rank, determined by the order in which it was added to the group. The rank is used to establish test traffic patterns.

A user can only control nodes in his/her session. To allocate nodes to the session, the user needs to add nodes to a group (of the session). All nodes in a group can be referenced by the group name. A node can be allocated to multiple groups of a session.

In the following example, three groups are established on a console node:

```
lst add_group servers 192.168.10.[8,10,12-16]@tcp
lst add_group readers 192.168.1.[1-253/2]@o2ib
lst add_group writers 192.168.1.[2-254/2]@o2ib
```

These three groups include:

- Nodes that will function as 'servers' to be accessed by 'clients' during the LNet self-test session
- Nodes that will function as 'clients' that will simulate *reading* data from the 'servers'
- Nodes that will function as 'clients' that will simulate *writing* data to the 'servers'

**Note**

A *console node* can associate kernel space *test nodes* with the session by running `lst add_group *NIDs*`, but a userspace test node cannot be actively added to the session. A console node can passively "accept" a test node to associate with a test session while the test node running `lstclient`connects to the console node, i.e: `lstclient --sesid *CONSOLE_NID* --group *NAME*`).

### Defining and Running the Tests

A *test* generates a network load between two groups of nodes, a source group identified using the `--from`parameter and a target group identified using the `--to` parameter. When a test is running, each node in the `--from *group*` simulates a client by sending requests to nodes in the `--to *group*`, which are simulating a set of servers, and then receives responses in return. This activity is designed to mimic Lustre file system RPC traffic.

A *batch* is a collection of tests that are started and stopped together and run in parallel. A test must always be run as part of a batch, even if it is just a single test. Users can only run or stop a test batch, not individual tests.

Tests in a batch are non-destructive to the file system, and can be run in a normal Lustre file system environment (provided the performance impact is acceptable).

A simple batch might contain a single test, for example, to determine whether the network bandwidth presents an I/O bottleneck. In this example, the `--to *group*` could be comprised of Lustre OSSs and `--from *group*` the compute nodes. A second test could be added to perform pings from a login node to the MDS to see how checkpointing affects the `ls -l` process.

Two types of tests are available:

- **ping -** A `ping` generates a short request message, which results in a short response. Pings are useful to determine latency and small message overhead and to simulate Lustre metadata traffic.
- **brw -** In a `brw` ('bulk read write') test, data is transferred from the target to the source (`brwread`) or data is transferred from the source to the target (`brwwrite`). The size of the bulk transfer is set using the `size`parameter. A brw test is useful to determine network bandwidth and to simulate Lustre I/O traffic.

In the example below, a batch is created called `bulk_rw`. Then two `brw` tests are added. In the first test, 1M of data is sent from the servers to the clients as a simulated read operation with a simple data validation check. In the second test, 4K of data is sent from the clients to the servers as a simulated write operation with a full data validation check.

```
lst add_batch bulk_rw
lst add_test --batch bulk_rw --from readers --to servers \
  brw read check=simple size=1M
lst add_test --batch bulk_rw --from writers --to servers \
  brw write check=full size=4K
```

The traffic pattern and test intensity is determined by several properties such as test type, distribution of test nodes, concurrency of test, and RDMA operation type. For more details, see [*the section called “Batch and Test Commands”*](#batch-and-test-commands).

### Sample Script

This sample LNet self-test script simulates the traffic pattern of a set of Lustre servers on a TCP network, accessed by Lustre clients on an InfiniBand network (connected via LNet routers). In this example, half the clients are reading and half the clients are writing.

Run this script on the console node:

```
#!/bin/bash
export LST_SESSION=$$
lst new_session read/write
lst add_group servers 192.168.10.[8,10,12-16]@tcp
lst add_group readers 192.168.1.[1-253/2]@o2ib
lst add_group writers 192.168.1.[2-254/2]@o2ib
lst add_batch bulk_rw
lst add_test --batch bulk_rw --from readers --to servers \
brw read check=simple size=1M
lst add_test --batch bulk_rw --from writers --to servers \
brw write check=full size=4K
# start running
lst run bulk_rw
# display server stats for 30 seconds
lst stat servers & sleep 30; kill $!
# tear down
lst end_session
```

**Note**

This script can be easily adapted to pass the group NIDs by shell variables or command line arguments (making it good for general-purpose use).

## LNet Self-Test Command Reference

The LNet self-test (`lst`) utility is used to issue LNet self-test commands. The `lst` utility takes a number of command line arguments. The first argument is the command name and subsequent arguments are command-specific.

### Session Commands

This section describes `lst` session commands.

**LST_FEATURES**

The `lst` utility uses the `LST_FEATURES` environmental variable to determine what optional features should be enabled. All features are disabled by default. The supported values for `LST_FEATURES` are:

- **1 -** Enable the Variable Page Size feature for LNet Selftest.

Example:

```
export LST_FEATURES=1
```

**LST_SESSION**

The `lst` utility uses the `LST_SESSION` environmental variable to identify the session locally on the self-test console node. This should be a numeric value that uniquely identifies all session processes on the node. It is convenient to set this to the process ID of the shell both for interactive use and in shell scripts. Almost all `lst` commands require `LST_SESSION` to be set.

Example:

```
export LST_SESSION=$$
```

**new_session [ --timeout SECONDS ]** **[ -- force ] SESSNAME**

Creates a new session session named *SESSNAME*.

| **Parameter**         | **Description**                                              |
| --------------------- | ------------------------------------------------------------ |
| `--timeout *seconds*` | Console timeout value of the session. The session ends automatically if it remains idle (i.e., no commands are issued) for this period. |
| `--force`             | Ends conflicting sessions. This determines who 'wins' when one session conflicts with another. For example, if there is already an active session on this node, then the attempt to create a new session fails unless the `--force` flag is specified. If the `--force` flag is specified, then the active session is ended. Similarly, if a session attempts to add a node that is already 'owned' by another session, the `--force` flag allows this session to 'steal' the node. |
| `*name*`              | A human-readable string to print when listing sessions or reporting session conflicts. |

**Example:**

```
$ lst new_session --force read_write
```

`end_session`

Stops all operations and tests in the current session and clears the session's status.

```
$ lst end_session
```

`show_session`

Shows the session information. This command prints information about the current session. It does not require LST_SESSION to be defined in the process environment.

```
$ lst show_session 
```

### Group Commands

This section describes `lst` group commands.

`add_group *name* *NIDS* [*NIDs*...]`

Creates the group and adds a list of test nodes to the group.

| **Parameter** | **Description**                                              |
| ------------- | ------------------------------------------------------------ |
| `*name*`      | Name of the group.                                           |
| `*NIDs*`      | A string that may be expanded to include one or more LNet NIDs. |

**Example:**

```
$ lst add_group servers 192.168.10.[35,40-45]@tcp
$ lst add_group clients 192.168.1.[10-100]@tcp 192.168.[2,4].\
  [10-20]@tcp
```

`update_group *name* [--refresh] [--clean *status*] [--remove *NIDs*]`

Updates the state of nodes in a group or adjusts a group's membership. This command is useful if some nodes have crashed and should be excluded from the group.

| **Parameter**      | **Description**                                              |                                             |
| ------------------ | ------------------------------------------------------------ | ------------------------------------------- |
| `--refresh`        | Refreshes the state of all inactive nodes in the group.      |                                             |
| `--clean *status*` | Removes nodes with a specified status from the group. Status may be: |                                             |
|                    | active                                                       | The node is in the current session.         |
|                    | busy                                                         | The node is now owned by another session.   |
|                    | down                                                         | The node has been marked down.              |
|                    | unknown                                                      | The node's status has yet to be determined. |
|                    | invalid                                                      | Any state but active.                       |
| `--remove *NIDs*`  | Removes specified nodes from the group.                      |                                             |

**Example:**

```
$ lst update_group clients --refresh
$ lst update_group clients --clean busy
$ lst update_group clients --clean invalid // \
  invalid == busy || down || unknown
$ lst update_group clients --remove \192.168.1.[10-20]@tcp
```

`list_group [*name*] [--active] [--busy] [--down] [--unknown] [--all]`

Prints information about a group or lists all groups in the current session if no group is specified.

| **Parameter** | **Description**         |
| ------------- | ----------------------- |
| `*name*`      | The name of the group.  |
| `--active`    | Lists the active nodes. |
| `--busy`      | Lists the busy nodes.   |
| `--down`      | Lists the down nodes.   |
| `--unknown`   | Lists unknown nodes.    |
| `--all`       | Lists all nodes.        |

Example:

```
$ lst list_group
1) clients
2) servers
Total 2 groups
$ lst list_group clients
ACTIVE BUSY DOWN UNKNOWN TOTAL
3 1 2 0 6
$ lst list_group clients --all
192.168.1.10@tcp Active
192.168.1.11@tcp Active
192.168.1.12@tcp Busy
192.168.1.13@tcp Active
192.168.1.14@tcp DOWN
192.168.1.15@tcp DOWN
Total 6 nodes
$ lst list_group clients --busy
192.168.1.12@tcp Busy
Total 1 node
```

`del_group *name*`

Removes a group from the session. If the group is referred to by any test, then the operation fails. If nodes in the group are referred to only by this group, then they are kicked out from the current session; otherwise, they are still in the current session.

```
$ lst del_group clients
```

`lstclient --sesid *NID* --group *name* [--server_mode]`

Use `lstclient` to run the userland self-test client. The `lstclient` command should be executed after creating a session on the console. There are only two mandatory options for `lstclient`:

| **Parameter**    | **Description**                                              |
| ---------------- | ------------------------------------------------------------ |
| `--sesid *NID*`  | The first console's NID.                                     |
| `--group *name*` | The test group to join.                                      |
| `--server_mode`  | When included, forces LNet to behave as a server, such as starting an acceptor if the underlying NID needs it or using privileged ports. Only root is allowed to use the `--server_mode` option. |

**Example:**

```
Console $ lst new_session testsession
Client1 $ lstclient --sesid 192.168.1.52@tcp --group clients
```

**Example:**

```
Client1 $ lstclient --sesid 192.168.1.52@tcp |--group clients --server_mode
```

### Batch and Test Commands

This section describes `lst` batch and test commands.

`add_batch *name*`

A default batch test set named batch is created when the session is started. You can specify a batch name by using `add_batch`:

```
$ lst add_batch bulkperf
```

Creates a batch test called `bulkperf`.

```
add_test --batch batchname [--loop loop_count] [--concurrency active_count] [--distribute source_count:sink_count] \
         --from group --to group brw|ping test_options
        
```

Adds a test to a batch. The parameters are described below.

| **Parameter**                              | **Description**                                              |
| ------------------------------------------ | ------------------------------------------------------------ |
| `--batch *batchname*`                      | Names a group of tests for later execution.                  |
| `--loop *loop_count*`                      | Number of times to run the test.                             |
| `--concurrency *active_count*`             | The number of requests that are active at one time.          |
| `--distribute *source_count*:*sink_count*` | Determines the ratio of client nodes to server nodes for the specified test. This allows you to specify a wide range of topologies, including one-to-one and all-to-all. Distribution divides the source group into subsets, which are paired with equivalent subsets from the target group so only nodes in matching subsets communicate. |
| `--from *group*`                           | The source group (test client).                              |
| `--to *group*`                             | The target group (test server).                              |
| `ping`                                     | Sends a small request message, resulting in a small reply message. For more details, see [*the section called “Defining and Running the Tests”*](#defining-and-running-the-tests). `ping` does not have any additional options. |
| `brw`                                      | Sends a small request message followed by a bulk data transfer, resulting in a small reply message. [*the section called “Defining and Running the Tests”*](#defining-and-running-the-tests). Options are: |
| `read | write`                             | Read or write. The default is read.                          |
| `size=*bytes[KM]*`                         | I/O size in bytes, kilobytes, or Megabytes (i.e., `size=1024`, `size=4K`, `size=1M`). The default is 4 kilobytes. |
| `check=full|simple`                        | A data validation check (checksum of data). The default is that no check is done. |

**Examples showing use of the distribute parameter:**

```
Clients: (C1, C2, C3, C4, C5, C6)
Server: (S1, S2, S3)
--distribute 1:1 (C1->S1), (C2->S2), (C3->S3), (C4->S1), (C5->S2),
\(C6->S3) /* -> means test conversation */ --distribute 2:1 (C1,C2->S1), (C3,C4->S2), (C5,C6->S3)
--distribute 3:1 (C1,C2,C3->S1), (C4,C5,C6->S2), (NULL->S3)
--distribute 3:2 (C1,C2,C3->S1,S2), (C4,C5,C6->S3,S1)
--distribute 4:1 (C1,C2,C3,C4->S1), (C5,C6->S2), (NULL->S3)
--distribute 4:2 (C1,C2,C3,C4->S1,S2), (C5, C6->S3, S1)
--distribute 6:3 (C1,C2,C3,C4,C5,C6->S1,S2,S3)
```

The setting `--distribute 1:1` is the default setting where each source node communicates with one target node.

When the setting `--distribute 1: *n*` (where `*n*` is the size of the target group) is used, each source node communicates with every node in the target group.

Note that if there are more source nodes than target nodes, some source nodes may share the same target nodes. Also, if there are more target nodes than source nodes, some higher-ranked target nodes will be idle.

**Example showing a brw test:**

```
$ lst add_group clients 192.168.1.[10-17]@tcp
$ lst add_group servers 192.168.10.[100-103]@tcp
$ lst add_batch bulkperf
$ lst add_test --batch bulkperf --loop 100 --concurrency 4 \
  --distribute 4:2 --from clients brw WRITE size=16K
```

In the example above, a batch test called bulkperf that will do a 16 kbyte bulk write request. In this test, two groups of four clients (sources) write to each of four servers (targets) as shown below:

- `192.168.1.[10-13]` will write to `192.168.10.[100,101]`
- `192.168.1.[14-17]` will write to `192.168.10.[102,103]`

**list_batch [name]** **[--test index]** **[--active]** **[--invalid]** **[--server|client]**

Lists batches in the current session or lists client and server nodes in a batch or a test.

| **Parameter**     | **Description**                                              |
| ----------------- | ------------------------------------------------------------ |
| `--test *index*`  | Lists tests in a batch. If no option is used, all tests in the batch are listed. If one of these options are used, only specified tests in the batch are listed: |
| `active`          | Lists only active batch tests.                               |
| `invalid`         | Lists only invalid batch tests.                              |
| `server | client` | Lists client and server nodes in a batch test.               |

**Example:**

```
$ lst list_batchbulkperf
$ lst list_batch bulkperf
Batch: bulkperf Tests: 1 State: Idle
ACTIVE BUSY DOWN UNKNOWN TOTAL
client 8 0 0 0 8
server 4 0 0 0 4
Test 1(brw) (loop: 100, concurrency: 4)
ACTIVE BUSY DOWN UNKNOWN TOTAL
client 8 0 0 0 8
server 4 0 0 0 4
$ lst list_batch bulkperf --server --active
192.168.10.100@tcp Active
192.168.10.101@tcp Active
192.168.10.102@tcp Active
192.168.10.103@tcp Active
```

`run *name*`

Runs the batch.

```
$ lst run bulkperf

```

`stop *name*`

Stops the batch.

```
$ lst stop bulkperf

```

**query name [--test index]** **[--timeout seconds]** **[--loop loopcount]** **[--delay seconds]** **[--all]**

Queries the batch status.

| **Parameter**         | **Description**                                              |
| --------------------- | ------------------------------------------------------------ |
| `--test *index*`      | Only queries the specified test. The test index starts from 1. |
| `--timeout *seconds*` | The timeout value to wait for RPC. The default is 5 seconds. |
| `--loop *#*`          | The loop count of the query.                                 |
| `--delay *seconds*`   | The interval of each query. The default is 5 seconds.        |
| `--all`               | The list status of all nodes in a batch or a test.           |

**Example:**

```
$ lst run bulkperf
$ lst query bulkperf --loop 5 --delay 3
Batch is running
Batch is running
Batch is running
Batch is running
Batch is running
$ lst query bulkperf --all
192.168.1.10@tcp Running
192.168.1.11@tcp Running
192.168.1.12@tcp Running
192.168.1.13@tcp Running
192.168.1.14@tcp Running
192.168.1.15@tcp Running
192.168.1.16@tcp Running
192.168.1.17@tcp Running
$ lst stop bulkperf
$ lst query bulkperf
Batch is idle

```

### Other Commands

This section describes other `lst` commands.

`ping [-session] [--group *name*] [--nodes *NIDs*] [--batch *name*] [--server] [--timeout *seconds*]`

Sends a 'hello' query to the nodes.

| **Parameter**         | **Description**                                              |
| --------------------- | ------------------------------------------------------------ |
| `--session`           | Pings all nodes in the current session.                      |
| `--group *name*`      | Pings all nodes in a specified group.                        |
| `--nodes *NIDs*`      | Pings all specified nodes.                                   |
| `--batch *name*`      | Pings all client nodes in a batch.                           |
| `--server`            | Sends RPC to all server nodes instead of client nodes. This option is only used with `--batch *name*`. |
| `--timeout *seconds*` | The RPC timeout value.                                       |

**Example:**

```
# lst ping 192.168.10.[15-20]@tcp
192.168.1.15@tcp Active [session: liang id: 192.168.1.3@tcp]
192.168.1.16@tcp Active [session: liang id: 192.168.1.3@tcp]
192.168.1.17@tcp Active [session: liang id: 192.168.1.3@tcp]
192.168.1.18@tcp Busy [session: Isaac id: 192.168.10.10@tcp]
192.168.1.19@tcp Down [session: <NULL> id: LNET_NID_ANY]
192.168.1.20@tcp Down [session: <NULL> id: LNET_NID_ANY]

```

`stat [--bw] [--rate] [--read] [--write] [--max] [--min] [--avg] " " [--timeout *seconds*] [--delay *seconds*] *group|NIDs* [*group|NIDs*]`

The collection performance and RPC statistics of one or more nodes.

| **Parameter**         | **Description**                                              |
| --------------------- | ------------------------------------------------------------ |
| `--bw`                | Displays the bandwidth of the specified group/nodes.         |
| `--rate`              | Displays the rate of RPCs of the specified group/nodes.      |
| `--read`              | Displays the read statistics of the specified group/nodes.   |
| `--write`             | Displays the write statistics of the specified group/nodes.  |
| `--max`               | Displays the maximum value of the statistics.                |
| `--min`               | Displays the minimum value of the statistics.                |
| `--avg`               | Displays the average of the statistics.                      |
| `--timeout *seconds*` | The timeout of the statistics RPC. The default is 5 seconds. |
| `--delay *seconds*`   | The interval of the statistics (in seconds).                 |

**Example:**

```
$ lst run bulkperf
$ lst stat clients
[LNet Rates of clients]
[W] Avg: 1108 RPC/s Min: 1060 RPC/s Max: 1155 RPC/s
[R] Avg: 2215 RPC/s Min: 2121 RPC/s Max: 2310 RPC/s
[LNet Bandwidth of clients]
[W] Avg: 16.60 MB/s Min: 16.10 MB/s Max: 17.1 MB/s
[R] Avg: 40.49 MB/s Min: 40.30 MB/s Max: 40.68 MB/s

```

Specifying a group name ( *group* ) causes statistics to be gathered for all nodes in a test group. For example:

```
$ lst stat servers

```

where servers is the name of a test group created by `lst add_group`

Specifying a `*NID*` range (`*NIDs*`) causes statistics to be gathered for selected nodes. For example:

```
$ lst stat 192.168.0.[1-100/2]@tcp

```

Only LNet performance statistics are available. By default, all statistics information is displayed. Users can specify additional information with these options.

`show_error [--session] [*group*|*NIDs*]...`

Lists the number of failed RPCs on test nodes.

| **Parameter** | **Description**                                              |
| ------------- | ------------------------------------------------------------ |
| `--session`   | Lists errors in the current test session. With this option, historical RPC errors are not listed. |

**Example:**

```
$ lst show_error client
sclients
12345-192.168.1.15@tcp: [Session: 1 brw errors, 0 ping errors] \
  [RPC: 20 errors, 0 dropped,
12345-192.168.1.16@tcp: [Session: 0 brw errors, 0 ping errors] \
  [RPC: 1 errors, 0 dropped, Total 2 error nodes in clients
$ lst show_error --session clients
clients
12345-192.168.1.15@tcp: [Session: 1 brw errors, 0 ping errors]
Total 1 error nodes in clients

```

 

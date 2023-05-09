# Glossary

### A

- ACL

  Access control list. An extended attribute associated with a file that contains enhanced authorization directives.

- Administrative OST failure

  A manual configuration change to mark an OST as unavailable, so that operations intended for that OST fail immediately with an I/O error instead of waiting indefinitely for OST recovery to complete

### C

- Completion callback

  An RPC made by the lock server on an OST or MDT to another system, usually a client, to indicate that the lock is now granted.

- configlog

  An llog file used in a node, or retrieved from a management server over the network with configuration instructions for the Lustre file system at startup time.

- Configuration lock

  A lock held by every node in the cluster to control configuration changes. When the configuration is changed on the MGS, it revokes this lock from all nodes. When the nodes receive the blocking callback, they quiesce their traffic, cancel and re-enqueue the lock and wait until it is granted again. They can then fetch the configuration updates and resume normal operation.

### D

- Default stripe pattern

  Information in the LOV descriptor that describes the default stripe count, stripe size, and layout pattern used for new files in a file system. This can be amended by using a directory stripe descriptor or a per-file stripe descriptor.

- Direct I/O

  A mechanism that can be used during read and write system calls to avoid memory cache overhead for large I/O requests. It bypasses the data copy between application and kernel memory, and avoids buffering the data in the client memory.

- Directory stripe descriptor

  An extended attribute that describes the default stripe pattern for new files created within that directory. This is also inherited by new subdirectories at the time they are created.

### E

- EA

  Extended attribute. A small amount of data that can be retrieved through a name (EA or attr) associated with a particular inode. A Lustre file system uses EAs to store striping information (indicating the location of file data on OSTs). Examples of extended attributes are ACLs, striping information, and the FID of the file.

- Eviction

  The process of removing a client's state from the server if the client is unresponsive to server requests after a timeout or if server recovery fails. If a client is still running, it is required to flush the cache associated with the server when it becomes aware that it has been evicted.

- Export

  The state held by a server for a client that is sufficient to transparently recover all in-flight operations when a single failure occurs.

- Extent

  A range of contiguous bytes or blocks in a file that are addressed by a {start, length} tuple instead of individual block numbers.

- Extent lock

  An LDLM lock used by the OSC to protect an extent in a storage object for concurrent control of read/write, file size acquisition, and truncation operations.

### F

- Failback

  The failover process in which the default active server regains control from the backup server that had taken control of the service.

- Failout OST

  An OST that is not expected to recover if it fails to answer client requests. A failout OST can be administratively failed, thereby enabling clients to return errors when accessing data on the failed OST without making additional network requests or waiting for OST recovery to complete.

- Failover

  The process by which a standby computer server system takes over for an active computer server after a failure of the active node. Typically, the standby computer server gains exclusive access to a shared storage device between the two servers.

- FID

  Lustre File Identifier. A 128-bit file system-unique identifier for a file or object in the file system. The FID structure contains a unique 64-bit sequence number (see FLDB), a 32-bit object ID (OID), and a 32-bit version number. The sequence number is unique across all Lustre targets (OSTs and MDTs).

- Fileset

  A group of files that are defined through a directory that represents the start point of a file system.

- FLDB

  FID location database. This database maps a sequence of FIDs to a specific target (MDT or OST), which manages the objects within the sequence. The FLDB is cached by all clients and servers in the file system, but is typically only modified when new servers are added to the file system.

- Flight group

  Group of I/O RPCs initiated by the OSC that are concurrently queued or processed at the OST. Increasing the number of RPCs in flight for high latency networks can increase throughput and reduce visible latency at the client.

### G

- Glimpse callback

  An RPC made by an OST or MDT to another system (usually a client) to indicate that a held extent lock should be surrendered. If the system is using the lock, then the system should return the object size and timestamps in the reply to the glimpse callback instead of cancelling the lock. Glimpses are introduced to optimize the acquisition of file attributes without introducing contention on an active lock.

### I

- Import

  The state held held by the client for each target that it is connected to. It holds server NIDs, connection state, and uncommitted RPCs needed to fully recover a transaction sequence after a server failure and restart.

- Intent lock

  A special Lustre file system locking operation in the Linux kernel. An intent lock combines a request for a lock with the full information to perform the operation(s) for which the lock was requested. This offers the server the option of granting the lock or performing the operation and informing the client of the operation result without granting a lock. The use of intent locks enables metadata operations (even complicated ones) to be implemented with a single RPC from the client to the server.

### L

- LBUG

  A fatal error condition detected by the software that halts execution of the kernel thread to avoid potential further corruption of the system state. It is printed to the console log and triggers a dump of the internal debug log. The system must be rebooted to clear this state.

- LDLM

  Lustre Distributed Lock Manager.

- lfs

  The Lustre file system command-line utility that allows end users to interact with Lustre software features, such as setting or checking file striping or per-target free space. For more details, see [the section called “ `lfs`”](06.03-User%20Utilities.md#lfs).

- LFSCK

  Lustre file system check. A distributed version of a disk file system checker. Normally, `lfsck` does not need to be run, except when file systems are damaged by events such as multiple disk failures and cannot be recovered using file system journal recovery.

- llite

  Lustre lite. This term is in use inside code and in module names for code that is related to the Linux client VFS interface.

- llog

  Lustre log. An efficient log data structure used internally by the file system for storing configuration and distributed transaction records. An `llog` is suitable for rapid transactional appends of records and cheap cancellation of records through a bitmap.

- llog catalog

  Lustre log catalog. An `llog` with records that each point at an `llog`. Catalogs were introduced to give `llogs`increased scalability. `llogs` have an originator which writes records and a replicator which cancels records when the records are no longer needed.

- LMV

  Logical metadata volume. A module that implements a DNE client-side abstraction device. It allows a client to work with many MDTs without changes to the llite module. The LMV code forwards requests to the correct MDT based on name or directory striping information and merges replies into a single result to pass back to the higher `llite` layer that connects the Lustre file system with Linux VFS, supports VFS semantics, and complies with POSIX interface specifications.

- LND

  Lustre network driver. A code module that enables LNet support over particular transports, such as TCP and various kinds of InfiniBand networks.

- LNet

  Lustre networking. A message passing network protocol capable of running and routing through various physical layers. LNet forms the underpinning of LNETrpc.

- Lock client

  A module that makes lock RPCs to a lock server and handles revocations from the server.

- Lock server

  A service that is co-located with a storage target that manages locks on certain objects. It also issues lock callback requests, calls while servicing or, for objects that are already locked, completes lock requests.

- LOV

  Logical object volume. The object storage analog of a logical volume in a block device volume management system, such as LVM or EVMS. The LOV is primarily used to present a collection of OSTs as a single device to the MDT and client file system drivers.

- LOV descriptor

  A set of configuration directives which describes which nodes are OSS systems in the Lustre cluster and providing names for their OSTs.

- Lustre client

  An operating instance with a mounted Lustre file system.

- Lustre file

  A file in the Lustre file system. The implementation of a Lustre file is through an inode on a metadata server that contains references to a storage object on OSSs.

### M

- mballoc

  Multi-block allocate. Functionality in ext4 that enables the `ldiskfs` file system to allocate multiple blocks with a single request to the block allocator.

- MDC

  Metadata client. A Lustre client component that sends metadata requests via RPC over LNet to the metadata target (MDT).

- MDD

  Metadata disk device. Lustre server component that interfaces with the underlying object storage device to manage the Lustre file system namespace (directories, file ownership, attributes).

- MDS

  Metadata server. The server node that is hosting the metadata target (MDT).

- MDT

  Metadata target. A storage device containing the file system namespace that is made available over the network to a client. It stores filenames, attributes, and the layout of OST objects that store the file data.

- MGS

  Management service. A software module that manages the startup configuration and changes to the configuration. Also, the server node on which this system runs.

- mountconf

  The Lustre configuration protocol that formats disk file systems on servers with the `mkfs.lustre` program, and prepares them for automatic incorporation into a Lustre cluster. This allows clients to be configured and mounted with a simple `mount` command.

### N

- NID

  Network identifier. Encodes the type, network number, and network address of a network interface on a node for use by the Lustre file system.

- NIO API

  A subset of the LNet RPC module that implements a library for sending large network requests, moving buffers with RDMA.

- Node affinity

  Node affinity describes the property of a multi-threaded application to behave sensibly on multiple cores. Without the property of node affinity, an operating scheduler may move application threads across processors in a sub-optimal way that significantly reduces performance of the application overall.

- NRS

  Network request scheduler. A subcomponent of the PTLRPC layer, which specifies the order in which RPCs are handled at servers. This allows optimizing large numbers of incoming requests for disk access patterns, fairness between clients, and other administrator-selected policies.

- NUMA

  Non-uniform memory access. Describes a multi-processing architecture where the time taken to access given memory differs depending on memory location relative to a given processor. Typically machines with multiple sockets are NUMA architectures.

### O

- OBD

  Object-based device. The generic term for components in the Lustre software stack that can be configured on the client or server. Examples include MDC, OSC, LOV, MDT, and OST.

- OBD API

  The programming interface for configuring OBD devices. This was formerly also the API for accessing object IO and attribute methods on both the client and server, but has been replaced by the OSD API in most parts of the code.

- OBD type

  Module that can implement the Lustre object or metadata APIs. Examples of OBD types include the LOV, OSC and OSD.

- Obdfilter

  An older name for the OBD API data object operation device driver that sits between the OST and the OSD. In Lustre software release 2.4 this device has been renamed OFD."

- Object storage

  Refers to a storage-device API or protocol involving storage objects. The two most well known instances of object storage are the T10 iSCSI storage object protocol and the Lustre object storage protocol (a network implementation of the Lustre object API). The principal difference between the Lustre protocol and T10 protocol is that the Lustre protocol includes locking and recovery control in the protocol and is not tied to a SCSI transport layer.

- opencache

  A cache of open file handles. This is a performance enhancement for NFS.

- Orphan objects

  Storage objects to which no Lustre file points. Orphan objects can arise from crashes and are automatically removed by an `llog` recovery between the MDT and OST. When a client deletes a file, the MDT unlinks it from the namespace. If this is the last link, it will atomically add the OST objects into a per-OST `llog`(if a crash has occurred) and then wait until the unlink commits to disk. (At this point, it is safe to destroy the OST objects. Once the destroy is committed, the MDT `llog` records can be cancelled.)

- OSC

  Object storage client. The client module communicating to an OST (via an OSS).

- OSD

  Object storage device. A generic, industry term for storage devices with a more extended interface than block-oriented devices such as disks. For the Lustre file system, this name is used to describe a software module that implements an object storage API in the kernel. It is also used to refer to an instance of an object storage device created by that driver. The OSD device is layered on a file system, with methods that mimic create, destroy and I/O operations on file inodes.

- OSS

  Object storage server. A server OBD that provides access to local OSTs.

- OST

  Object storage target. An OSD made accessible through a network protocol. Typically, an OST is associated with a unique OSD which, in turn is associated with a formatted disk file system on the server containing the data objects.

### P

- pdirops

  A locking protocol in the Linux VFS layer that allows for directory operations performed in parallel.

- Pool

  OST pools allows the administrator to associate a name with an arbitrary subset of OSTs in a Lustre cluster. A group of OSTs can be combined into a named pool with unique access permissions and stripe characteristics.

- Portal

  A service address on an LNet NID that binds requests to a specific request service, such as an MDS, MGS, OSS, or LDLM. Services may listen on multiple portals to ensure that high priority messages are not queued behind many slow requests on another portal.

- PTLRPC

  An RPC protocol layered on LNet. This protocol deals with stateful servers and has exactly-once semantics and built in support for recovery.

### R

- Recovery

  The process that re-establishes the connection state when a client that was previously connected to a server reconnects after the server restarts.

- Replay request

  The concept of re-executing a server request after the server has lost information in its memory caches and shut down. The replay requests are retained by clients until the server(s) have confirmed that the data is persistent on disk. Only requests for which a client received a reply and were assigned a transaction number by the server are replayed. Requests that did not get a reply are resent.

- Resent request

  An RPC request sent from a client to a server that has not had a reply from the server. This might happen if the request was lost on the way to the server, if the reply was lost on the way back from the server, or if the server crashes before or after processing the request. During server RPC recovery processing, resent requests are processed after replayed requests, and use the client RPC XID to determine if the resent RPC request was already executed on the server.

- Revocation callback

  Also called a "blocking callback". An RPC request made by the lock server (typically for an OST or MDT) to a lock client to revoke a granted DLM lock.

- Root squash

  A mechanism whereby the identity of a root user on a client system is mapped to a different identity on the server to avoid root users on clients from accessing or modifying root-owned files on the servers. This does not prevent root users on the client from assuming the identity of a non-root user, so should not be considered a robust security mechanism. Typically, for management purposes, at least one client system should not be subject to root squash.

- Routing

  LNet routing between different networks and LNDs.

- RPC

  Remote procedure call. A network encoding of a request.

### S

- Stripe

  A contiguous, logical extent of a Lustre file written to a single OST. Used synonymously with a single OST data object that makes up part of a file visible to user applications.

- Stripe size

  The maximum number of bytes that will be written to an OST object before the next object in a file's layout is used when writing sequential data to a file. Once a full stripe has been written to each of the objects in the layout, the first object will be written to again in round-robin fashion.

- Stripe count

  The number of OSTs holding objects for a RAID0-striped Lustre file.

### T

- T10 object protocol

  An object storage protocol tied to the SCSI transport layer. The Lustre file system does not use T10.

### W

- Wide striping

  Strategy of using many OSTs to store stripes of a single file. This obtains maximum bandwidth access to a single file through parallel utilization of many OSTs. For more information about wide striping, see [*the section called “Lustre Striping Internals”*](03.08-Managing%20File%20Layout%20(Striping)%20and%20Free%20Space.md#lustre-striping-internals).


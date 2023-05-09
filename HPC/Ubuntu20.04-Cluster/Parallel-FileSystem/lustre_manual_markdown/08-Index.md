# Index
## A
Access Control List (ACL), Using ACLs
	examples, Examples
	how they work, How ACLs Work
	using, Using ACLs with the Lustre Software
audit
	change logs, Audit with Changelogs

## B
backup, Backing up a File System
	aborting recovery, Aborting Recovery
	index objects, Backing Up an OST or MDT (Backend File System Level)
	MDT file system, Backing Up an OST or MDT (Backend File System Level)
	MDT/OST device level, Backing Up and Restoring an MDT or OST (ldiskfs Device Level)
	new/changed files, Backing up New/Changed Files to the Backup File System
	OST and MDT, Backing Up an OST or MDT
	OST config, Backing Up OST Configuration Files
	OST file system, Backing Up an OST or MDT (Backend File System Level)
	restoring file system backup, Restoring a File-Level Backup
	restoring OST config, Restoring OST Configuration Files
	rsync, Lustre_rsync

​		examples, lustre_rsync Examples
​		using, Using Lustre_rsync

​	using LVM, Using LVM Snapshots with the Lustre File System

​		creating, Creating an LVM-based Backup File System
​		creating snapshots, Creating Snapshot Volumes
​		deleting, Deleting Old Snapshots
​		resizing, Changing Snapshot Volume Size
​		restoring, Restoring the File System From a Snapshot

​	ZFS to ldiskfs, Migrate from a ZFS to an ldiskfs based filesystem, Migrate from an ldiskfs to a ZFS based 		filesystem
​	ZFS ZPL, Migration Between ZFS and ldiskfs Target Filesystems

barrier, Global Write Barriers

​	impose, Impose Barrier
​	query, Query Barrier
​	remove, Remove Barrier
​	rescan, Rescan Barrier

benchmarking
	

​	application profiling, Collecting Application Profiling Information (stats-collect)
​	local disk, Testing Local Disk Performance
​	MDS performance, Testing MDS Performance (mds-survey)
​	network, Testing Network Performance
​	OST I/O, Testing OST I/O Performance (ost-survey)
​	OST performance, Testing OST Performance (obdfilter-survey)
​	raw hardware with sgpdd-survey, Testing I/O Performance of Raw Hardware (sgpdd-survey)
​	remote disk, Testing Remote Disk Performance
​	tuning storage, Tuning Linux Storage Devices
​	with Lustre I/O Kit, Using Lustre I/O Kit Tools

## C
change logs (see monitoring)
commit on share, Commit on Share
	

​	tuning, Tuning Commit On Share
​	working with, Working with Commit on Share

configlogs, Lustre Configuration Logs
configuring, Introduction
	

​	adaptive timeouts, Configuring Adaptive Timeouts
​	LNet options, LNet Options
​	module options, Module Options
​	multihome, Multihome Server Example
​	network

​		forwarding, forwarding ("")
​		rnet_htable_size, rnet_htable_size
​		routes, routes ("")
​		SOCKLND, SOCKLND Kernel TCP/IP LND
​		tcp, networks ("tcp")

​	network topology, Network Topology

## D
debug

​	utilities, Testing / Debugging Utilities

debugging, Diagnostic and Debugging Tools

​	admin tools, Tools for Administrators and Developers
​	developer tools, Tools for Developers
​	developers tools, Lustre Debugging for Developers
​	disk contents, Looking at Disk Content
​	external tools, External Debugging Tools
​	kernel debug log, Controlling Information Written to the Kernel Debug Log
​	lctl example, Sample lctl Run
​	memory leaks, Finding Memory Leaks Using leak_finder.pl
​	message format, Understanding the Lustre Debug Messaging Format
​	procedure, Lustre Debugging Procedures
​	tools, Lustre Debugging Tools
​	using lctl, Using the lctl Tool to View Debug Messages
​	using strace, Troubleshooting with strace

design (see setup)
DLC

​	Code Example, Adding a route code example

dom, Introduction to Data on MDT (DoM), User Commands, DoM Stripe Size Restrictions, lfs getstripe for DoM files, lfs find for DoM files, The dom_stripesize parameter, Disable DoM

​	disabledom, Disable DoM
​	domstripesize, DoM Stripe Size Restrictions
​	dom_stripesize, The dom_stripesize parameter
​	intro, Introduction to Data on MDT (DoM)
​	lfsfind, lfs find for DoM files
​	lfsgetstripe, lfs getstripe for DoM files
​	lfssetstripe, lfs setstripe for DoM files
​	usercommands, User Commands

## E
e2scan, e2scan
errors (see troubleshooting)
## F
failover, What is Failover?, Setting Up a Failover Environment

​	and Lustre, Failover Functionality in a Lustre File System
​	capabilities, Failover Capabilities
​	configuration, Types of Failover Configurations
​	high-availability (HA) software, Selecting High-Availability (HA) Software
​	MDT, MDT Failover Configuration (Active/Passive), MDT Failover Configuration (Active/Active)
​	OST, OST Failover Configuration (Active/Active)
​	power control device, Selecting Power Equipment
​	power management software, Selecting Power Management Software
​	setup, Preparing a Lustre File System for Failover

feature overview

​	configuration, Configuration

file layout

​	See striping, Lustre File Layout (Striping) Considerations

filefrag, filefrag
fileset, Fileset Feature
fragmentation, Description

## H
Hierarchical Storage Management (HSM)

​	introduction, Introduction
High availability (see failover)
HSM

​	agents, Agents
​	agents and copytools, Agents and copytool
​	archiveID backends, Archive ID, multiple backends
​	automatic restore, Automatic restore
​	changelogs, change logs
​	commands, Commands
​	coordinator, Coordinator
​	file states, File states
​	grace_delay, grace_delay
​	hsm_control, hsm_controlpolicy
​	max_requests, max_requests
​	policy, policy
​	policy engine, Policy engine
​	registered agents, Registered agents
​	request monitoring, Request monitoring
​	requests, Requests
​	requirements, Requirements
​	robinhood, Robinhood
​	setup, Setup
​	timeout, Timeout
​	tuning, Tuning

## I
I/O, Handling Full OSTs

​	adding an OST, Adding an OST to a Lustre File System
​	bringing OST online, Returning an Inactive OST Back Online
​	direct, Performing Direct I/O
​	disabling OST creates, Disabling creates on a Full OST
​	full OSTs, Handling Full OSTs
​	migrating data, Migrating Data within a File System
​	OST space usage, Checking OST Space Usage
​	pools, Creating and Managing OST Pools

imperative recovery, Imperative Recovery

​	Configuration Suggestions, Configuration Suggestions for Imperative Recovery
​	MGS role, MGS role
​	Tuning, Tuning Imperative Recovery

inodes

​	MDS, Setting Formatting Options for an ldiskfs MDT
​	OST, Setting Formatting Options for an ldiskfs OST

installing, Steps to Installing the Lustre Software

​	preparation, Preparing to Install the Lustre Software

Introduction, Introduction

​	Requirements, Requirements

ior-survey, ior-survey
Isolation, Isolating Clients to a Sub-directory Tree

​	client identification, Identifying Clients
​	configuring, Configuring Isolation
​	making permanent, Making Isolation Permanent

## J
jobstats (see monitoring)
## L
large_xattr

​	ea_inode, File and File System Limits, Upgrading to Lustre Software Release 2.x (Major Release)

lctl, lctl
ldiskfs

​	formatting options, Setting ldiskfs File System Formatting Options

lfs, lfs
lfs_migrate, lfs_migrate
llodbstat, llobdstat
llog_reader, llog_reader
llstat, llstat
llverdev, llverdev
ll_decode_filter_fid, ll_decode_filter_fid
ll_recover_lost_found_objs, ll_recover_lost_found_objs
LNet, Introducing LNet, Overview of LNet Module Parameters, Dynamically Configuring LNet Routes, lustre_routes_config, lustre_routes_conversion, Route Configuration Examples (see configuring)

​	best practice, Best Practices for LNet Options
​	buffer yaml syntax, Enable Routing and Adjust Router Buffer Configuration
​	capi general information, General API Information
​	capi input params, API Common Input Parameters
​	capi output params, API Common Output Parameters
​	capi return code, API Return Code
​	cli, Configuring LNet, Displaying Global Settings, Adding, Deleting and Showing Networks, Manual 		      						Adding, Deleting and Showing Peers, Adding, Deleting and Showing routes, Enabling and Disabling Routing, Showing routing information, Configuring Routing Buffers, Importing YAML Configuration File, Exporting Configuration in YAML format, Showing LNet Traffic Statistics

​		asymmetrical route, Asymmetrical Routes
​		dynamic discovery, Dynamic Peer Discovery

​	comments, Including comments
​	Configuring LNet, Configuring LNet via lnetctl
​	cyaml, Internal YAML Representation (cYAML)
​	error block, Error Block
​	escaping commas with quotes, Escaping commas with quotes
​	features, Key Features of LNet
​	hardware multi-rail configuration, Hardware Based Multi-Rail Configurations with LNet
​	InfiniBand load balancing, Load Balancing with an InfiniBand* Network
​	ip2nets, Setting the LNet Module ip2nets Parameter
​	lustre.conf, Setting Up lustre.conf for Load Balancing
​	lustre_lnet_config_buf, Adjusting Router Buffer Pools
​	lustre_lnet_config_net, Adding a Network Interface
​	lustre_lnet_config_ni_system, Configuring LNet
​	lustre_lnet_config_route, Adding Routes
​	lustre_lnet_del_net, Deleting a Network Interface
​	lustre_lnet_del_route, Deleting Routes
​	lustre_lnet_enable_routing, Enabling and Disabling Routing
​	lustre_lnet_show stats, Showing LNet Traffic Statistics
​	lustre_lnet_show_buf, Showing Routing information
​	lustre_lnet_show_net, Showing Network Interfaces
​	lustre_lnet_show_route, Showing Routes
​	lustre_yaml, Adding/Deleting/Showing Parameters through a YAML Block
​	management, Updating the Health Status of a Peer or Router
​	module parameters, Setting the LNet Module networks Parameter
​	network yaml syntax, Network Configuration
​	proc, Monitoring LNet
​	route checker, Configuring the Router Checker
​	router yaml syntax, Route Configuration
​	routes, Setting the LNet Module routes Parameter
​	routing example, Routing Example
​	self-test, LNet Self-Test Overview
​	show block, Show Block, The LNet Configuration C-API
​	starting/stopping, Starting and Stopping LNet
​	statistics yaml syntax, Show Statistics
​	supported networks, Supported Network Types
​	testing, Testing the LNet Configuration
​	tuning, Tuning LNet Parameters
​	understanding, Introducing LNet
​	using NID, Using a Lustre Network Identifier (NID) to Identify a Node
​	yaml syntax, YAML Syntax

logs, Snapshot Logs
lr_reader, lr_reader
lshowmount, lshowmount
lsom

​	enablelsom, Enable LSoM
​	intro, Introduction to Lazy Size on MDT (LSoM)
​	lfsgetsom, lfs getsom for LSoM data
​	usercommands, User Commands

lst, lst
Lustre, What a Lustre File System Is (and What It Isn't)

​	at scale, Lustre Cluster
​	cluster, Lustre Cluster
​	components, Lustre Components
​	configuring, Configuring a Simple Lustre File System

​		additional options, Additional Configuration Options
​		for scale, Scaling the Lustre File System
​		simple example, Simple Lustre Configuration Example
​		striping, Changing Striping Defaults
​		utilities, Using the Lustre Configuration Utilities

​	features, Lustre Features
​	fileset, Fileset Feature
​	I/O, Lustre File System Storage and I/O
​	LNet, Lustre Networking (LNet)	
​	MGS, Management Server (MGS)
​	Networks, Lustre Networks
​	requirements, Lustre File System Components
​	storage, Lustre File System Storage and I/O
​	striping, Lustre File System and Striping
​	upgrading (see upgrading)

lustre

​	errors (see troubleshooting)
​	recovery (see recovery)
​	troubleshooting (see troubleshooting)

lustre_rmmod.sh, lustre_rmmod.sh
lustre_rsync, lustre_rsync
LVM (see backup)
l_getidentity, l_getidentity

## M
maintenance, Working with Inactive OSTs, Working with Inactive MDTs

​	aborting recovery, Aborting Recovery
​	adding a OST, Adding a New OST to a Lustre File System
​	adding an MDT, Adding a New MDT to a Lustre File System
​	backing up OST config, Backing Up OST Configuration Files
​	bringing OST online, Returning an Inactive OST Back Online
​	changing a NID, Changing a Server NID
​	changing failover node address, Changing the Address of a Failover Node
​	Clearing a config, Clearing configuration
​	finding nodes, Finding Nodes in the Lustre File System
​	full OSTs, Migrating Data within a File System
​	identifying OST host, Determining Which Machine is Serving an OST
​	inactive MDTs, Working with Inactive MDTs
​	inactive OSTs, Working with Inactive OSTs
​	mounting a server, Mounting a Server Without Lustre Service
​	pools, Creating and Managing OST Pools
​	regenerating config logs, Regenerating Lustre Configuration Logs
​	reintroducing an OSTs, Returning a Deactivated OST to Service
​	removing an MDT, Removing an MDT from the File System
​	removing an OST, Removing and Restoring MDTs and OSTs, Removing an OST from the File System
​	restoring an OST, Removing and Restoring MDTs and OSTs
​	restoring OST config, Restoring OST Configuration Files
​	separate a combined MGS/MDT, Separate a combined MGS/MDT

MDT

​	multiple MDSs, Upgrading to Lustre Software Release 2.x (Major Release)

migrating metadata, Migrating Metadata within a Filesystem, Whole Directory Migration, Striped Directory Migration
mkfs.lustre, mkfs.lustre
monitoring, Lustre Changelogs, Lustre Jobstats

​	additional tools, Other Monitoring Options
​	change logs, Lustre Changelogs, Working with Changelogs
​	jobstats, Lustre Jobstats, How Jobstats Works, Enable/Disable Jobstats, Check Job Stats, Clear Job Stats, 			Configure Auto-cleanup Interval
​	Lustre Monitoring Tool, Lustre Monitoring Tool (LMT)

mount, mount
mount.lustre, mount.lustre
MR

​	addremotepeers, Adding Remote Peers that are Multi-Rail Capable
​	configuring, Configuring Multi-Rail
​	deleteinterfaces, Deleting Network Interfaces
​	deleteremotepeers, Deleting Remote Peers
​	health, LNet Health
​	mrhealth

​		display, Displaying Information
​		failuretypes, Failure Types and Behavior
​		initialsetup, Initial Settings Recommendations
​		interface, User Interface
​		value, Health Value

​	mrrouting, Notes on routing with Multi-Rail

​		routingex, Multi-Rail Cluster Example
​		routingmixed, Mixed Multi-Rail/Non-Multi-Rail Cluster
​		routingresiliency, Utilizing Router Resiliency

​	multipleinterfaces, Configure Multiple Interfaces on the Local Node
​	overview, Multi-Rail Overview

multiple-mount protection, Overview of Multiple-Mount Protection

## O
obdfilter-survey, obdfilter-survey
operations, Mounting by Label, Snapshot Operations

​	create, Creating a Snapshot
​	degraded OST RAID, Handling Degraded OST RAID Arrays
​	delete, Delete a Snapshot
​	erasing a file system, Erasing a File System
​	failover, Specifying Failout/Failover Mode for OSTs, Specifying NIDs and Failover
​	identifying OSTs, Identifying To Which Lustre File an OST Object Belongs
​	list, List Snapshots
​	mkdir, Creating a directory striped across multiple MDTs
​	modify, Modify Snapshot Attributes
​	mount, Mounting a Snapshot
​	mounting, Mounting a Server
​	mounting by label, Mounting by Label	
​	multiple file systems, Running Multiple Lustre File Systems
​	parameters, Setting and Retrieving Lustre Parameters
​	reclaiming space, Reclaiming Reserved Disk Space
​	remote directory, Creating a sub-directory on a given MDT
​	replacing an OST or MDS, Replacing an Existing OST or MDT
​	setdirstripe, Creating a directory striped across multiple MDTs	
​	shutdownLustre, Stopping the Filesystem
​	starting, Starting Lustre
​	striped directory, Creating a directory striped across multiple MDTs
​	unmount, Unmounting a Snapshot
​	unmounting, Unmounting a Specific Target on a Server

ost-survey, ost-survey

## P
performance (see benchmarking)
pings

​	evict_client, Client Death Notification
​	suppress_pings, "suppress_pings" Kernel Module Parameter

plot-llstat, plot-llstat
pools, Creating and Managing OST Pools

​	usage tips, Tips for Using OST Pools

proc

​	adaptive timeouts, Configuring Adaptive Timeouts
​	block I/O, Monitoring the OST Block I/O Stream
​	client metadata performance, Tuning the Client Metadata RPC Stream
​	client stats, Monitoring Client Activity
​	configuring adaptive timeouts, Configuring Adaptive Timeouts
​	debug, Enabling and Interpreting Debugging Logs
​	free space, Allocating Free Space on OSTs
​	LNet, Monitoring LNet
​	locking, Configuring Locking
​	OSS journal, Enabling OSS Asynchronous Journal Commit
​	read cache, Tuning OSS Read Cache
​	read/write survey, Monitoring Client Read-Write Offset Statistics, Monitoring Client Read-Write Extent 	Statistics
​	readahead, Tuning File Readahead and Directory Statahead
​	RPC tunables, Tuning the Client I/O RPC Stream
​	static timeouts, Setting Static Timeouts
​	thread counts, Setting MDS and OSS Thread Counts
​	watching RPC, Monitoring the Client RPC Stream

profiling (see benchmarking)
programming

​	upcall, User/Group Upcall

## Q
Quotas

​	allocating, Quota Allocation
​	configuring, Working with Quotas
​	creating, Quota Administration
​	enabling disk, Enabling Disk Quotas
​	Interoperability, Quotas and Version Interoperability
​	known issues, Granted Cache and Quota Limits
​	statistics, Lustre Quota Statistics
​	verifying, Quota Verification

## R
recovery, Recovery Overview

​	client eviction, Client Eviction
​	client failure, Client Failure
​	commit on share (see commit on share)
​	corruption of backing ldiskfs file system, Recovering from Errors or Corruption on a Backing ldiskfs File 	System
​	corruption of Lustre file system, Recovering from Corruption in the Lustre File System
​	failed recovery, Failed Recovery
​	LFSCK, Checking the file system with LFSCK
​	locks, Lock Recovery
​	MDS failure, MDS Failure (Failover)
​	metadata replay, Metadata Replay
​	network, Network Partition
​	oiscrub, Checking the file system with LFSCK
​	orphaned objects, Working with Orphaned Objects
​	OST failure, OST Failure (Failover)
​	unavailable OST, Recovering from an Unavailable OST
​	VBR (see version-based recovery)

reporting bugs (see troubleshooting)
restoring (see backup)
root squash, Using Root Squash

​	configuring, Configuring Root Squash
​	enabling, Enabling and Tuning Root Squash
​	tips, Tips on Using Root Squash

round-robin algorithm, Managing Free Space
routerstat, routerstat
rsync (see backup)

## S
selinux policy check, Checking SELinux Policy Enforced by Lustre Clients

​	determining, Determining SELinux Policy Info
​	enforcing, Enforcing SELinux Policy Check
​	making permanent, Making SELinux Policy Check Permanent
​	sending client, Sending SELinux Status Info from Clients

setup, Hardware Considerations

​	hardware, Hardware Considerations
​	inodes, Setting Formatting Options for an ldiskfs MDT
​	ldiskfs, Setting ldiskfs File System Formatting Options
​	limits, File and File System Limits
​	MDT, MGT and MDT Storage Hardware Considerations, Determining MDT Space Requirements
​	memory, Determining Memory Requirements

​		client, Client Memory Requirements
​		MDS, MDS Memory Requirements, Calculating MDS Memory Requirements
​		OSS, OSS Memory Requirements, Calculating OSS Memory Requirements

​	MGT, Determining MGT Space Requirements
​	network, Implementing Networks To Be Used by the Lustre File System
​	OST, OST Storage Hardware Considerations, Determining OST Space Requirements
​	space, Determining Space Requirements

sgpdd-survey, sgpdd-survey
space, How Lustre File System Striping Works

​	considerations, Lustre File Layout (Striping) Considerations
​	determining MDT requirements, Determining MDT Space Requirements
​	determining MGT requirements, Determining MGT Space Requirements
​	determining OST requirements, Determining OST Space Requirements
​	determining requirements, Determining Space Requirements
​	free space, Managing Free Space
​	location weighting, Adjusting the Weighting Between Free Space and Location
​	striping, How Lustre File System Striping Works

stats-collect, stats-collect
storage

​	configuring, Selecting Storage for the MDT and OSTs

​		external journal, Choosing Parameters for an External Journal
​		for best practice, Reliability Best Practices
​		for mkfs, Computing file system parameters for mkfs
​		MDT, Metadata Target (MDT)
​		OST, Object Storage Server (OST)
​		RAID options, Formatting Options for ldiskfs RAID Devices
​		SAN, Connecting a SAN to a Lustre File System

​	performance tradeoffs, Performance Tradeoffs

striping (see space)

​	allocations, Stripe Allocation Methods
​	configuration, Setting the File Layout/Striping Configuration (lfs setstripe)
​	considerations, Lustre File Layout (Striping) Considerations
​	count, Setting the Stripe Count
​	getting information, Retrieving File Layout/Striping Information (getstripe)
​	how it works, How Lustre File System Striping Works
​	metadata, Creating a directory striped across multiple MDTs
​	on specific OST, Creating a File on a Specific OST
​	overview, Lustre File System and Striping
​	per directory, Setting the Striping Layout for a Directory
​	per file system, Setting the Striping Layout for a File System
​	PFL, Progressive File Layout(PFL)
​	remote directories, Locating the MDT for a remote directory
​	round-robin algorithm, Managing Free Space
​	size, Choosing a Stripe Size
​	weighted algorithm, Managing Free Space
​	wide striping, Lustre Striping Internals

suppressing pings, Suppressing Pings

## T
troubleshooting, Lustre Error Messages

​	'Address already in use', Handling/Debugging "Bind: Address already in use" Error
​	'Error -28', Handling/Debugging Error "- 28"
​	common problems, Common Lustre File System Problems
​	error messages, Viewing Error Messages
​	error numbers, Error Numbers
​	OST out of memory, Log Message 'Out of Memory' on OST
​	reporting bugs, Reporting a Lustre File System Bug
​	slowdown during startup, Slowdown Occurs During Lustre File System Startup
​	timeouts on setup, Handling Timeouts on Initial Lustre File System Setup

tunefs.lustre, tunefs.lustre
tuning, Optimizing the Number of Service Threads (see benchmarking)

​	for small files, Improving Lustre I/O Performance for Small Files
​	Large Bulk IO, Large Bulk IO (16MB RPC)
​	libcfs, libcfs Tuning
​	LND tuning, LND Tuning
​	LNet, Tuning LNet Parameters
​	lockless I/O, Lockless I/O Tunables
​	MDS binding, Binding MDS Service Thread to CPU Partitions
​	MDS threads, Specifying the MDS Service Thread Count
​	Network interface binding, Binding Network Interface Against CPU Partitions
​	Network interface credits, Network Interface Credits
​	Network Request Scheduler (NRS) Tuning, Network Request Scheduler (NRS) Tuning

​		client round-robin over NIDs (CRR-N) policy, Client Round-Robin over NIDs (CRR-N) policy
​		Delay policy, Delay policy
​		first in, first out (FIFO) policy, First In, First Out (FIFO) policy
​		object-based round-robin (ORR) policy, Object-based Round-Robin (ORR) policy
​		Target-based round-robin (TRR) policy, Target-based Round-Robin (TRR) policy
​		Token Bucket Filter (TBF) policy, Token Bucket Filter (TBF) policy

​	OSS threads, Specifying the OSS Service Thread Count
​	portal round-robin, Portal Round-Robin
​	router buffers, Router Buffers
​	service threads, Optimizing the Number of Service Threads
​	with lfs ladvise, Server-Side Advice and Hinting
​	write performance, Understanding Why Write Performance is Better Than Read Performance

## U
upgrading, Release Interoperability and Upgrade Requirements

​	2.X.y to 2.X.y (minor release), Upgrading to Lustre Software Release 2.x.y (Minor Release)
​	major release (2.x to 2.x), Upgrading to Lustre Software Release 2.x (Major Release)

utilities

​	application profiling, Application Profiling Utilities
​	debugging, Testing / Debugging Utilities
​	system config, Additional System Configuration Utilities

## V
version

​	which version of Lustre am I running?, Revisions

Version-based recovery (VBR), Version-based Recovery

​	messages, VBR Messages
​	tips, Tips for Using VBR

## W
weighted algorithm, Managing Free Space
wide striping, File and File System Limits, Upgrading to Lustre Software Release 2.x (Major Release), Lustre Striping Internals

​	large_xattr

​		ea_inode, File and File System Limits, Upgrading to Lustre Software Release 2.x (Major Release)

## X
xattr

​	See wide striping, File and File System Limits
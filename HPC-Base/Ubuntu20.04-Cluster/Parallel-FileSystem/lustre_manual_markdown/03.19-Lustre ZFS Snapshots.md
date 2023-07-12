# Lustre ZFS Snapshots

- [Lustre ZFS Snapshots](#lustre-zfs-snapshots)
  * [Introduction](#introduction)
    + [Requirements](#requirements)
  * [Configuration](#configuration)
  * [Snapshot Operations](#snapshot-operations)
    + [Creating a Snapshot](#creating-a-snapshot)
    + [Delete a Snapshot](#delete-a-snapshot)
    + [Mounting a Snapshot](#mounting-a-snapshot)
    + [Unmounting a Snapshot](#unmounting-a-snapshot)
    + [List Snapshots](#list-snapshots)
    + [Modify Snapshot Attributes](#modify-snapshot-attributes)
  * [Global Write Barriers](#global-write-barriers)
    + [Impose Barrier](#impose-barrier)
    + [Remove Barrier](#remove-barrier)
    + [Query Barrier](#query-barrier)
    + [Rescan Barrier](#rescan-barrier)
  * [Snapshot Logs](#snapshot-logs)
  * [Lustre Configuration Logs](#lustre-configuration-logs)


This chapter describes the ZFS Snapshot feature support in Lustre and contains following sections:

- [the section called “Introduction”](#introduction)
- [the section called “Configuration”](#introduction)
- [the section called “Snapshot Operations”](#snapshot-operations)
- [the section called “Global Write Barriers”](#global-write-barriers)
- [the section called “Snapshot Logs”](#snapshot-logs)
- [the section called “Lustre Configuration Logs”](#lustre-configuration-logs)

## Introduction

Snapshots provide fast recovery of files from a previously created checkpoint without recourse to an offline backup or remote replica. Snapshots also provide a means to version-control storage, and can be used to recover lost files or previous versions of files.

Filesystem snapshots are intended to be mounted on user-accessible nodes, such as login nodes, so that users can restore files (e.g. after accidental delete or overwrite) without administrator intervention. It would be possible to mount the snapshot filesystem(s) via automount when users access them, rather than mounting all snapshots, to reduce overhead on login nodes when the snapshots are not in use.

Recovery of lost files from a snapshot is usually considerably faster than from any offline backup or remote replica. However, note that snapshots do not improve storage reliability and are just as exposed to hardware failure as any other storage volume.

### Requirements

All Lustre server targets must be ZFS file systems running Lustre version 2.10 or later. In addition, the MGS must be able to communicate via ssh or another remote access protocol, without password authentication, to all other servers.

The feature is enabled by default and cannot be disabled. The management of snapshots is done through `lctl`commands on the MGS.

Lustre snapshot is based on Copy-On-Write; the snapshot and file system may share a single copy of the data until a file is changed on the file system. The snapshot will prevent the space of deleted or overwritten files from being released until the snapshot(s) referencing those files is deleted. The file system administrator needs to establish a snapshot create/backup/remove policy according to their system’s actual size and usage.

## Configuration

The snapshot tool loads system configuration from the `/etc/ldev.conf` file on the MGS and calls related ZFS commands to maintian the Lustre snapshot pieces on all targets (MGS/MDT/OST). Please note that the `/etc/ldev.conf` file is used for other purposes as well.

The format of the file is:

```
<host> foreign/- <label> <device> [journal-path]/- [raidtab]
```

The format of `<label>` is:

```
fsname-<role><index> or <role><index>
```

The format of <device> is:

```
[md|zfs:][pool_dir/]<pool>/<filesystem>
```

Snapshot only uses the fields <host>, <label> and <device>.

Example:

```
mgs# cat /etc/ldev.conf
host-mdt1 - myfs-MDT0000 zfs:/tmp/myfs-mdt1/mdt1
host-mdt2 - myfs-MDT0001 zfs:myfs-mdt2/mdt2
host-ost1 - OST0000 zfs:/tmp/myfs-ost1/ost1
host-ost2 - OST0001 zfs:myfs-ost2/ost2
```

The configuration file is edited manually.

Once the configuration file is updated to reflect the current file system setup, you are ready to create a file system snapshot.

## Snapshot Operations

### Creating a Snapshot

To create a snapshot of an existing Lustre file system, run the following `lctl` command on the MGS:

```
lctl snapshot_create [-b | --barrier [on | off]] [-c | --comment
comment] -F | --fsname fsname> [-h | --help] -n | --name ssname>
[-r | --rsh remote_shell][-t | --timeout timeout]
```

| **Option** | **Description**                                              |
| ---------- | ------------------------------------------------------------ |
| `-b`       | set write barrier before creating snapshot. The default value is 'on'. |
| `-c`       | a description for the purpose of the snapshot                |
| `-F`       | the filesystem name                                          |
| `-h`       | help information                                             |
| `-n`       | the name of the snapshot                                     |
| `-r`       | the remote shell used for communication with remote target. The default value is 'ssh' |
| `-t`       | the lifetime (seconds) for write barrier. The default value is 30 seconds |

### Delete a Snapshot

To delete an existing snapshot, run the following `lctl` command on the MGS:

```
lctl snapshot_destroy [-f | --force] <-F | --fsname fsname>
<-n | --name ssname> [-r | --rsh remote_shell]
```

| **Option** | **Description**                                              |
| ---------- | ------------------------------------------------------------ |
| `-f`       | destroy the snapshot by force                                |
| `-F`       | the filesystem name                                          |
| `-h`       | help information                                             |
| `-n`       | the name of the snapshot                                     |
| `-r`       | the remote shell used for communication with remote target. The default value is 'ssh' |

### Mounting a Snapshot

Snapshots are treated as separate file systems and can be mounted on Lustre clients. The snapshot file system must be mounted as a read-only file system with the `-o ro` option. If the `mount` command does not include the read-only option, the mount will fail.

**Note**

Before a snapshot can be mounted on the client, the snapshot must first be mounted on the servers using the `lctl` utility.

To mount a snapshot on the server, run the following lctl command on the MGS:

```
lctl snapshot_mount <-F | --fsname fsname> [-h | --help]
<-n | --name ssname> [-r | --rsh remote_shell]
```

| **Option** | **Description**                                              |
| ---------- | ------------------------------------------------------------ |
| `-F`       | the filesystem name                                          |
| `-h`       | help information                                             |
| `-n`       | the name of the snapshot                                     |
| `-r`       | the remote shell used for communication with remote target. The default value is 'ssh' |

After the successful mounting of the snapshot on the server, clients can now mount the snapshot as a read-only filesystem. For example, to mount a snapshot named *snapshot_20170602* for a filesystem named *myfs*, the following mount command would be used:

```
mgs# lctl snapshot_mount -F myfs -n snapshot_20170602
```

After mounting on the server, use `lctl snapshot_list` to get the fsname for the snapshot itself as follows:

```
ss_fsname=$(lctl snapshot_list -F myfs -n snapshot_20170602 |
          awk '/^snapshot_fsname/ { print $2 }')
```

Finally, mount the snapshot on the client:

```
mount -t lustre -o ro $MGS_nid:/$ss_fsname $local_mount_point
```

### Unmounting a Snapshot

To unmount a snapshot from the servers, first unmount the snapshot file system from all clients, using the standard `umount` command on each client. For example, to unmount the snapshot file system named *snapshot_20170602* run the following command on each client that has it mounted:

```
client# umount $local_mount_point
```

After all clients have unmounted the snapshot file system, run the following `lctl`command on a server node where the snapshot is mounted:

```
lctl snapshot_umount [-F | --fsname fsname] [-h | --help]
<-n | -- name ssname> [-r | --rsh remote_shell]
```

| **Option** | **Description**                                              |
| ---------- | ------------------------------------------------------------ |
| `-F`       | the filesystem name                                          |
| `-h`       | help information                                             |
| `-n`       | the name of the snapshot                                     |
| `-r`       | the remote shell used for communication with remote target. The default value is 'ssh' |

For example:

```
lctl snapshot_umount -F myfs -n snapshot_20170602
```
### List Snapshots

To list the available snapshots for a given file system, use the following `lctl` command on the MGS:

```
lctl snapshot_list [-d | --detail] <-F | --fsname fsname>
[-h | -- help] [-n | --name ssname] [-r | --rsh remote_shell]
```

| **Option** | **Description**                                              |
| ---------- | ------------------------------------------------------------ |
| `-d`       | list every piece for the specified snapshot                  |
| `-F`       | the filesystem name                                          |
| `-h`       | help information                                             |
| `-n`       | the snapshot's name. If the snapshot name is not supplied, all snapshots for this file system will be displayed |
| `-r`       | the remote shell used for communication with remote target. The default value is 'ssh' |

### Modify Snapshot Attributes

Currently, Lustre snapshot has five user visible attributes; snapshot name, snapshot comment, create time, modification time, and snapshot file system name. Among them, the former two attributes can be modified. Renaming follows the general ZFS snapshot name rules, such as the maximum length is 256 bytes, cannot conflict with the reserved names, and so on.

To modify a snapshot’s attributes, use the following `lctl` command on the MGS:

```
lctl snapshot_modify [-c | --comment comment]
<-F | --fsname fsname> [-h | --help] <-n | --name ssname>
[-N | --new new_ssname] [-r | --rsh remote_shell]
```

| **Option** | **Description**                                              |
| ---------- | ------------------------------------------------------------ |
| `-c`       | update the snapshot's comment                                |
| `-F`       | the filesystem name                                          |
| `-h`       | help information                                             |
| `-n`       | the snapshot's name                                          |
| `-N`       | rename the snapshot's name as *new_ssname*                   |
| `-r`       | the remote shell used for communication with remote target. The default value is 'ssh' |

## Global Write Barriers

Snapshots are non-atomic across multiple MDTs and OSTs, which means that if there is activity on the file system while a snapshot is being taken, there may be user-visible namespace inconsistencies with files created or destroyed in the interval between the MDT and OST snapshots. In order to create a consistent snapshot of the file system, we are able to set a global write barrier, or “freeze” the system. Once set, all metadata modifications will be blocked until the write barrier is actively removed (“thawed”) or expired. The user can set a timeout parameter on a global barrier or the barrier can be explicitly removed. The default timeout period is 30 seconds.

It is important to note that snapshots are usable without the global barrier. Only files that are currently being modified by clients (write, create, unlink) may be inconsistent as noted above if the barrier is not used. Other files not curently being modified would be usable even without the barrier.

The snapshot create command will call the write barrier internally when requested using the `-b` option to `lctl snapshot_create`. So, explicit use of the barrier is not required when using snapshots but included here as an option to quiet the file system before a snapshot is created.

### Impose Barrier

To impose a global write barrier, run the `lctl barrier_freeze` command on the MGS:

```
lctl barrier_freeze <fsname> [timeout (in seconds)]
where timeout default is 30.
```

For example, to freeze the filesystem *testfs* for `15` seconds:

```
mgs# lctl barrier_freeze testfs 15
```

If the command is successful, there will be no output from the command. Otherwise, an error message will be printed.

### Remove Barrier

To remove a global write barrier, run the `lctl barrier_thaw` command on the MGS:

```
lctl barrier_thaw <fsname>
```

For example, to thaw the write barrier for the filesystem *testfs*:

```
mgs# lctl barrier_thaw testfs
```

If the command is successful, there will be no output from the command. Otherwise, an error message will be printed.

### Query Barrier

To see how much time is left on a global write barrier, run the `lctl barrier_stat` command on the MGS:

```
# lctl barrier_stat <fsname>
```

For example, to stat the write barrier for the filesystem *testfs*:

```
mgs# lctl barrier_stat testfs
The barrier for testfs is in 'frozen'
The barrier will be expired after 7 seconds
```

If the command is successful, a status from the table below will be printed. Otherwise, an error message will be printed.

The possible status and related meanings for the write barrier are as follows:

**Table 13. Write Barrier Status**

| **Status**    | **Meaning**                                                  |
| ------------- | ------------------------------------------------------------ |
| `init`        | barrier has never been set on the system                     |
| `freezing_p1` | In the first stage of setting the write barrier              |
| `freezing_p2` | the second stage of setting the write barrier                |
| `frozen`      | the write barrier has been set successfully                  |
| `thawing`     | In thawing the write barrier                                 |
| `thawed`      | The write barrier has been thawed                            |
| `failed`      | Failed to set write barrier                                  |
| `expired`     | The write barrier is expired                                 |
| `rescan`      | In scanning the MDTs status, see the command `barrier_rescan` |
| `unknown`     | Other cases                                                  |

If the barrier is in ’freezing_p1’, ’freezing_p2’ or ’frozen’ status, then the remaining lifetime will be returned also.

### Rescan Barrier

To rescan a global write barrier to check which MDTs are active, run the `lctl barrier_rescan` command on the MGS:

```
lctl barrier_rescan <fsname> [timeout (in seconds)],
where the default timeout is 30 seconds.
```

For example, to rescan the barrier for filesystem *testfs*:

```
mgs# lctl barrier_rescan testfs
1 of 4 MDT(s) in the filesystem testfs are inactive
```

If the command is successful, the number of MDTs that are unavailable against the total MDTs will be reported. Otherwise, an error message will be printed.

## Snapshot Logs

A log of all snapshot activity can be found in the following file: `/var/log/lsnapshot.log`. This file contains information on when a snapshot was created, an attribute was changed, when it was mounted, and other snapshot information.

The following is a sample `/var/log/lsnapshot` file:

```
Mon Mar 21 19:43:06 2016
(15826:jt_snapshot_create:1138:scratch:ssh): Create snapshot lss_0_0
successfully with comment <(null)>, barrier <enable>, timeout <30>
Mon Mar 21 19:43:11 2016(13030:jt_snapshot_create:1138:scratch:ssh):
Create snapshot lss_0_1 successfully with comment <(null)>, barrier
<disable>, timeout <-1>
Mon Mar 21 19:44:38 2016 (17161:jt_snapshot_mount:2013:scratch:ssh):
The snapshot lss_1a_0 is mounted
Mon Mar 21 19:44:46 2016
(17662:jt_snapshot_umount:2167:scratch:ssh): the snapshot lss_1a_0
have been umounted
Mon Mar 21 19:47:12 2016
(20897:jt_snapshot_destroy:1312:scratch:ssh): Destroy snapshot
lss_2_0 successfully with force <disable>
```

## Lustre Configuration Logs

A snapshot is independent from the original file system that it is derived from and is treated as a new file system name that can be mounted by Lustre client nodes. The file system name is part of the configuration log names and exists in configuration log entries. Two commands exist to manipulate configuration logs: `lctl fork_lcfg` and `lctl erase_lcfg`.

The snapshot commands will use configuration log functionality internally when needed. So, use of the barrier is not required to use snapshots but included here as an option. The following configuration log commands are independent of snapshots and can be used independent of snapshot use.

To fork a configuration log, run the following `lctl` command on the MGS:

```
lctl fork_lcfg
```

Usage: fork_lcfg <fsname> <newname>

To erase a configuration log, run the following `lctl` command on the MGS:

```
lctl erase_lcfg
```

Usage: erase_lcfg <fsname>

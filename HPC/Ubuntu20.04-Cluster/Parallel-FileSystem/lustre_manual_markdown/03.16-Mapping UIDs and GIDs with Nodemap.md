Introduced in Lustre 2.9

# Mapping UIDs and GIDs with Nodemap

- [Mapping UIDs and GIDs with Nodemap](#mapping-uids-and-gids-with-nodemap)
  * [Setting a Mapping](#setting-a-mapping)
    + [Defining Terms](#defining-terms)
    + [Deciding on NID Ranges](#deciding-on-nid-ranges)
    + [Describing and Deploying a Sample Mapping](#describing-and-deploying-a-sample-mapping)
  * [Altering Properties](#altering-properties)
    + [Managing the Properties](#managing-the-properties)
    + [Mixing Properties](#mixing-properties)
  * [Enabling the Feature](#enabling-the-feature)
  * [`default` Nodemap](#default-nodemap)
  * [Verifying Settings](#verifying-settings)
  * [Ensuring Consistency](#ensuring-consistency)


This chapter describes how to map UID and GIDs across a Lustre file system using the nodemap feature, and includes the following sections:

- [the section called “Setting a Mapping”](#setting-a-mapping)
- [the section called “Altering Properties”](#altering-properties)
- [the section called “Enabling the Feature”](#enabling-the-feature)
- [the section called “default Nodemap”](#default-nodemap)
- [the section called “Verifying Settings”](#verifying-settings)
- [the section called “Ensuring Consistency”](#ensuring-consistency)

## Setting a Mapping

The nodemap feature supported in Lustre 2.9 was first introduced in Lustre 2.7 as a technology preview. It allows UIDs and GIDs from remote systems to be mapped to local sets of UIDs and GIDs while retaining POSIX ownership, permissions and quota information. As a result, multiple sites with conflicting user and group identifiers can operate on a single Lustre file system without creating collisions in UID or GID space.

### Defining Terms

When the nodemap feature is enabled, client file system access to a Lustre system is filtered through the nodemap identity mapping policy engine. Lustre connectivity is governed by network identifiers, or *NIDs*, such as`192.168.7.121@tcp`. When an operation is made from a NID, Lustre decides if that NID is part of a *nodemap*, a policy group consisting of one or more NID ranges. If no policy group exists for that NID, access is squashed to user `nobody` by default. Each policy group also has several *properties*, such as `trusted` and `admin`, which determine access conditions. A collection of identity maps or *idmaps* are kept for each policy group. These idmaps determine how UIDs and GIDs on the client are translated into the canonical user space of the local Lustre file system.

In order for nodemap to function properly, the MGS, MDS, and OSS systems must all have a version of Lustre which supports nodemap. Clients operate transparently and do not require special configuration or knowledge of the nodemap setup.

### Deciding on NID Ranges

NIDs can be described as either a singleton address or a range of addresses. A single address is described in standard Lustre NID format, such as `10.10.6.120@tcp`. A range is described using a dash to separate the range, for example, `192.168.20.[0-255]@tcp`.

The range must be contiguous. The full LNet definition for a nidlist is as follows:

```
<nidlist>       :== <nidrange> [ ' ' <nidrange> ]
<nidrange>      :== <addrrange> '@' <net>
<addrrange>     :== '*' |
                        <ipaddr_range> |
                        <numaddr_range>
<ipaddr_range>  :==
        <numaddr_range>.<numaddr_range>.<numaddr_range>.<numaddr_range>
<numaddr_range> :== <number> |
                        <expr_list>
<expr_list>     :== '[' <range_expr> [ ',' <range_expr>] ']'
<range_expr>    :== <number> |
                        <number> '-' <number> |
                        <number> '-' <number> '/' <number>
<net>           :== <netname> | <netname><number>
<netname>       :== "lo" | "tcp" | "o2ib" | "gni"
<number>        :== <nonnegative decimal> | <hexadecimal> 
```

### Describing and Deploying a Sample Mapping

Deploy nodemap by first considering which users need to be mapped, and what sets of network addresses or ranges are involved. Issues of visibility between users must be examined as well.

Consider a deployment where researchers are working on data relating to birds. The researchers use a computing system which mounts Lustre from a single IPv4 address, `192.168.0.100`. Name this policy group `BirdResearchSite`. The IP address forms the NID `192.168.0.100@tcp`. Create the policy group and add the NID to that group on the MGS using the `lctl` command:

```
mgs# lctl nodemap_add BirdResearchSite
mgs# lctl nodemap_add_range --name BirdResearchSite --range 192.168.0.100@tcp
```

**Note**

A NID cannot be in more than one policy group. Assign a NID to a new policy group by first removing it from the existing group.

The researchers use the following identifiers on their host system:

- `swan` (UID 530) member of group `wetlands` (GID 600)
- `duck` (UID 531) member of group `wetlands` (GID 600)
- `hawk` (UID 532) member of group `raptor` (GID 601)
- `merlin` (UID 533) member of group `raptor` (GID 601)

Assign a set of six idmaps to this policy group, with four for UIDs, and two for GIDs. Pick a starting point, e.g. UID 11000, with room for additional UIDs and GIDs to be added as the configuration grows. Use the `lctl` command to set up the idmaps:

```
mgs# lctl nodemap_add_idmap --name BirdResearchSite --idtype uid --idmap 530:11000
mgs# lctl nodemap_add_idmap --name BirdResearchSite --idtype uid --idmap 531:11001
mgs# lctl nodemap_add_idmap --name BirdResearchSite --idtype uid --idmap 532:11002
mgs# lctl nodemap_add_idmap --name BirdResearchSite --idtype uid --idmap 533:11003
mgs# lctl nodemap_add_idmap --name BirdResearchSite --idtype gid --idmap 600:11000
mgs# lctl nodemap_add_idmap --name BirdResearchSite --idtype gid --idmap 601:11001
```

The parameter `530:11000` assigns a client UID, for example UID 530, to a single canonical UID, such as UID 11000. Each assignment is made individually. There is no method to specify a range `530-533:11000-11003`. UID and GID idmaps are assigned separately. There is no implied relationship between the two.

Files created on the Lustre file system from the `192.168.0.100@tcp` NID using UID `duck` and GID `wetlands` are stored in the Lustre file system using the canonical identifiers, in this case UID 11001 and GID 11000. A different NID, if not part of the same policy group, sees its own view of the same file space.

Suppose a previously created project directory exists owned by UID 11002/GID 11001, with mode 770. When users `hawk` and `merlin` at 192.168.0.100 place files named `hawk-file` and `merlin-file` into the directory, the contents from the 192.168.0.100 client appear as:

```
[merlin@192.168.0.100 projectsite]$ ls -la
total 34520
drwxrwx--- 2 hawk   raptor     4096 Jul 23 09:06 .
drwxr-xr-x 3 nobody nobody     4096 Jul 23 09:02 ..
-rw-r--r-- 1 hawk   raptor 10240000 Jul 23 09:05 hawk-file
-rw-r--r-- 1 merlin raptor 25100288 Jul 23 09:06 merlin-file
```

From a privileged view, the canonical owners are displayed:

```
[root@trustedSite projectsite]# ls -la
total 34520
drwxrwx--- 2 11002 11001     4096 Jul 23 09:06 .
drwxr-xr-x 3 root root     4096 Jul 23 09:02 ..
-rw-r--r-- 1 11002 11001 10240000 Jul 23 09:05 hawk-file
-rw-r--r-- 1 11003 11001 25100288 Jul 23 09:06 merlin-file
```

If UID 11002 or GID 11001 do not exist on the Lustre MDS or MGS, create them in LDAP or other data sources, or trust clients by setting `identity_upcall` to `NONE`. For more information, see *the section called “User/Group Upcall”*.

Building a larger and more complex configuration is possible by iterating through the `lctl` commands above. In short:

1. Create a name for the policy group.
2. Create a set of NID ranges used by the group.
3. Define which UID and GID translations need to occur for the group.

## Altering Properties

Privileged users access mapped systems with rights dependent on certain properties, described below. By default, root access is squashed to user `nobody`, which interferes with most administrative actions.

### Managing the Properties

Several properties exist, off by default, which change client behavior: `admin`, `trusted`, `squash_uid`, `squash_gid`, and `deny_unknown`.

- The `trusted` property permits members of a policy group to see the file system's canonical identifiers. In the above example, UID 11002 and GID 11001 will be seen without translation. This can be utilized when local UID and GID sets already map directly to the specified users.
- The property `admin` defines whether root is squashed on the policy group. By default, it is squashed, unless this property is enabled. Coupled with the `trusted` property, this will allow unmapped access for backup nodes, transfer points, or other administrative mount points.
- The property `deny_unknown` denies all access to users not mapped in a particular nodemap. This is useful if a site is concerned about unmapped users accessing the file system in order to satisfy security requirements.
- The properties `squash_uid` and `squash_gid` define the default UID and GID that users will be squashed to if unmapped, unless the deny_unknown flag is set, in which case access will still be denied.

Alter values to either true (1) or false (0) on the MGS:

```
mgs# lctl nodemap_modify --name BirdAdminSite --property trusted --value 1
mgs# lctl nodemap_modify --name BirdAdminSite --property admin --value 1
mgs# lctl nodemap_modify --name BirdAdminSite --property deny_unknown --value 1
```

Change values during system downtime to minimize the chance of any ownership or permissions problems if the policy group is active. Although changes can be made live, client caching of data may interfere with modification as there are a few seconds of lead time before the change is distributed. 

### Mixing Properties

With both `admin` and `trusted` properties set, the policy group has full access, as if nodemap was turned off, to the Lustre file system. The administrative site for the Lustre file system needs at least one group with both properties in order to perform maintenance or to perform administrative tasks.

**Warning**

MDS systems **must** be in a policy group with both these properties set to 1. It is recommended to put the MDS in a policy group labeled “TrustedSystems” or some identifier that makes the association clear.

If a policy group has the `admin` property set, but does not have the property `trusted` set, root is mapped directly to root, any explicitly specified UID and GID idmaps are honored, and other access is squashed. If root alters ownership to UIDs or GIDs which are locally known from that host but not part of an idmap, root effectively changes ownership of those files to the default squashed UID and GID.

If `trusted` is set but `admin` is not, the policy group has full access to the canonical UID and GID sets of the Lustre file system, and root is squashed.

The deny_unknown property, once enabled, prevents unmapped users from accessing the file system. Root access also is denied, if the `admin` property is off, and root is not part of any mapping.

When nodemaps are modified, the change events are queued and distributed across the cluster. Under normal conditions, these changes can take around ten seconds to propagate. During this distribution window, file access could be made via the old or new nodemap settings. Therefore, it is recommended to save changes for a maintenance window or to deploy them while the mapped nodes are not actively writing to the file system.

## Enabling the Feature

The nodemap feature is simple to enable:

```
mgs# lctl nodemap_activate 1
```

Passing the parameter 0 instead of 1 disables the feature again. After deploying the feature, validate the mappings are intact before offering the file system to be mounted by clients.

Introduced in Lustre 2.8So far, changes have been made on the MGS. Prior to Lustre 2.9, changes must also be manually set on MDS systems as well. Also, changes must be manually deployed to OSS servers if quota is enforced, utilizing `lctl set_param` instead of `lctl`. Prior to 2.9, the configuration is not persistent, requiring a script which generates the mapping to be saved and deployed after every Lustre restart. As an example, use this style to deploy settings on the OSS:`oss# lctl set_param nodemap.add_nodemap=*SiteName* oss# lctl set_param nodemap.add_nodemap_range='*SiteName 192.168.0.15@tcp*' oss# lctl set_param nodemap.add_nodemap_idmap='*SiteName* uid *510:1700*' oss# lctl set_param nodemap.add_nodemap_idmap='*SiteName* gid *612:1702*'`In Lustre 2.9 and later, nodemap configuration is saved on the MGS and distributed automatically to MGS, MDS, and OSS nodes, a process which takes approximately ten seconds in normal circumstances.

## `default` Nodemap

There is a special nodemap called `default`. As the name suggests, it is created by default and cannot be removed. It is like a fallback nodemap, setting the behaviour for Lustre clients that do not match any other nodemap.

Because of its special role, only some parameters can be set on the `default` nodemap:

- `admin`
- `trusted`
- `squash_uid`
- `squash_gid`
- `fileset`
- `audit_mode`

In particular, no UID/GID mapping can be defined on the `default` nodemap.

**Note**

Be careful when altering the `admin` and `trusted` properties of the `default` nodemap, especially if your Lustre servers fall into this nodemap.

## Verifying Settings

By using `lctl nodemap_info all`, existing nodemap configuration is listed for easy export. This command acts as a shortcut into the configuration interface for nodemap. On the Lustre MGS, the `nodemap.active` parameter contains a `1` if nodemap is active on the system. Each policy group creates a directory containing the following parameters:

- `admin` and `trusted` each contain a `1` if the values are set, and `0` otherwise.
- `idmap` contains a list of the idmaps for the policy group, while `ranges` contains a list of NIDs for the group.
- `squash_uid` and `squash_gid` determine what UID and GID users are squashed to if needed.

The expected outputs for the BirdResearchSite in the example above are:

```
mgs# lctl get_param nodemap.BirdResearchSite.idmap

 [
  { idtype: uid, client_id: 530, fs_id: 11000 },
  { idtype: uid, client_id: 531, fs_id: 11001 },
  { idtype: uid, client_id: 532, fs_id: 11002 },
  { idtype: uid, client_id: 533, fs_id: 11003 },
  { idtype: gid, client_id: 600, fs_id: 11000 },
  { idtype: gid, client_id: 601, fs_id: 11001 }
 ]

 mgs# lctl get_param nodemap.BirdResearchSite.ranges
 [
  { id: 11, start_nid: 192.168.0.100@tcp, end_nid: 192.168.0.100@tcp }
 ]
```

## Ensuring Consistency

Consistency issues may arise in a nodemap enabled configuration when Lustre clients mount from an unknown NID range, new UIDs and GIDs that were not part of a known map are added, or there are misconfigurations in the rules. Keep in mind the following when activating nodemap on a production system:

- Creating new policy groups or idmaps on a production system is allowed, but reserve a maintenance window to alter the `trusted` property to avoid metadata problems.
- To perform administrative tasks, access the Lustre file system via a policy group with `trusted` and `admin`properties set. This prevents the creation of orphaned and squashed files. Granting the `admin` property without the `trusted` property is dangerous. The root user on the client may know of UIDs and GIDs that are not present in any idmap. If root alters ownership to those identifiers, the ownership is squashed as a result. For example, tar file extracts may be flipped from an expected UID such as UID 500 to `nobody`, normally UID 99.
- To map distinct UIDs at two or more sites onto a single UID or GID on the Lustre file system, create overlapping idmaps and place each site in its own policy group. Each distinct UID may have its own mapping onto the target UID or GID.
- Introduced in Lustre 2.8In Lustre 2.8, changes must be manually kept in a script file to be re-applied after a Lustre reload, and changes must be made on each OSS, MDS, and MGS nodes, as there is no automatic synchronization between the nodes.
- If `deny_unknown` is in effect, it is possible for unmapped users to see dentries which were viewed by a mapped user. This is a result of client caching, and unmapped users will not be able to view any file contents.
- Nodemap activation status can be checked with `lctl nodemap_info`, but extra validation is possible. One way of ensuring valid deployment on a production system is to create a fingerprint of known files with specific UIDs and GIDs mapped to a test client. After bringing the Lustre system online after maintenance, the test client can validate the UIDs and GIDs map correctly before the system is mounted in user space. 
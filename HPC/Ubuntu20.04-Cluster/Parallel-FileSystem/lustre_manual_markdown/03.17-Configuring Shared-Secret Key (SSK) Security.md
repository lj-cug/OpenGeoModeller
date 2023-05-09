Introduced in Lustre 2.9

# Configuring Shared-Secret Key (SSK) Security

- [Configuring Shared-Secret Key (SSK) Security](#configuring-shared-secret-key-ssk-security)
  * [SSK Security Overview](#ssk-security-overview)
    + [Key features](#key-features)
  * [SSK Security Flavors](#ssk-security-flavors)
    + [Secure RPC Rules](#secure-rpc-rules)
      - [Defining Rules](#defining-rules)
      - [Listing Rules](#listing-rules)
      - [Deleting Rules](#deleting-rules)
  * [SSK Key Files](#ssk-key-files)
    + [Key File Management](#key-file-management)
      - [Writing Key Files](#writing-key-files)
      - [Modifying Key Files](#modifying-key-files)
      - [Reading Key Files](#reading-key-files)
      - [Loading Key Files](#loading-key-files)
  * [Lustre GSS Keyring](#lustre-gss-keyring)
    + [Setup](#setup)
    + [Server Setup](#server-setup)
    + [Debugging GSS Keyring](#debugging-gss-keyring)
    + [Revoking Keys](#revoking-keys)
  * [Role of Nodemap in SSK](#role-of-nodemap-in-ssk)
  * [SSK Examples](#ssk-examples)
    + [Securing Client to Server Communications](#securing-client-to-server-communications)
    + [Securing MGS Communications](#securing-mgs-communications)
    + [Securing Server to Server Communications](#securing-server-to-server-communications)
  * [Viewing Secure PtlRPC Contexts](#viewing-secure-ptlrpc-contexts)

This chapter describes how to configure Shared-Secret Key security and includes the following sections:

- [the section called “SSK Security Overview”](#ssk-security-overview)
- [the section called “SSK Security Flavors”](#ssk-security-flavors)
- [the section called “SSK Key Files”](#ssk-key-files)
- [the section called “Lustre GSS Keyring”](#lustre-gss-keyring)
- [the section called “Role of Nodemap in SSK”](#role-of-nodemap-in-ssk)
- [the section called “SSK Examples”](#ssk-examples)
- [the section called “Viewing Secure PtlRPC Contexts”](#viewing-secure-ptlrpc-contexts)

## SSK Security Overview

The SSK feature ensures integrity and data protection for Lustre PtlRPC traffic. Key files containing a shared secret and session-specific attributes are distributed to Lustre hosts. This authorizes Lustre hosts to mount the file system and optionally enables secure data transport, depending on which security flavor is configured. The administrator handles the generation, distribution, and installation of SSK key files, see *the section called “Key File Management”*.

### Key features

SSK provides the following key features:

- Host-based authentication
- Data Transport Privacy
  - Encrypts Lustre RPCs
  - Prevents eavesdropping
- Data Transport Integrity - Keyed-Hashing Message Authentication Code (HMAC)
  - Prevents man-in-the-middle attacks
  - Ensures RPCs cannot be altered undetected

## SSK Security Flavors

SSK is implemented as a Generic Security Services (GSS) mechanism through Lustre's support of the GSS Application Program Interface (GSSAPI). The SSK GSS mechanism supports five flavors that offer varying levels of protection.

Flavors provided:

- `skn` - SSK Null (Authentication)
- `ska` - SSK Authentication and Integrity for non-bulk RPCs
- `ski` - SSK Authentication and Integrity
- `skpi` - SSK Authentication, Privacy, and Authentication
- `gssnull` - Provides no protection. Used for testing purposes only

The table below describes the security characteristics of each flavor:

**Table 9. SSK Security Flavor Protections**

|                               | skn  | ska  | ski  | skpi |
| ----------------------------- | ---- | ---- | ---- | ---- |
| Required to mount file system | Yes  | Yes  | Yes  | Yes  |
| Provides RPC Integrity        | No   | Yes  | Yes  | Yes  |
| Provides RPC Privacy          | No   | No   | No   | Yes  |
| Provides Bulk RPC Integrity   | No   | No   | Yes  | Yes  |
| Provides Bulk RPC Privacy     | No   | No   | No   | Yes  |



Valid non-GSS flavors include:

`null` - Provides no protection. This is the default flavor.

`plain` - Plaintext with a hash on each RPC.

### Secure RPC Rules

Secure RPC configuration rules are written to the Lustre log (llog) with the `lctl` command. Rules are processed with the llog and dictate the security flavor that is used for a particular Lustre network and direction.

**Note**

Rules take affect in a matter of seconds and impact both existing and new connections.

Rule format:

*target*.srpc.flavor.*network*[.*direction*]=*flavor*

- *target* - This could be the file system name or a specific MDT/OST device name.
- *network* - LNet network name of the RPC initiator. For example `tcp1` or `o2ib0`. This can also be the keyword `default` that applies to all networks otherwise specified.
- *direction* - Direction is optional. This could be one of `mdt2mdt`, `mdt2ost`,` cli2mdt`, or `cli2ost`.

**Note**

To secure the connection to the MGS use the `mgssec=`*flavor* mount option. This is required because security rules are unknown to the initiator until after the MGS connection has been established.

The examples below are for a test Lustre file system named *testfs*.

#### Defining Rules

Rules can be defined and deleted in any order. The rule with the greatest specificity for a given connection is applied. The *fsname*`.srpc.flavor.default` rule is the broadest rule as it applies to all non-MGS connections for the file system in the absence of a more specific rule. You may tailor SSK security to your needs by further specifying a specific `target`, `network`, and/or `direction`.

The following example illustrates an approach to configuring SSK security for an environment consisting of three LNet networks. The requirements for this example are:

- All non-MGS connections must be authenticated.
- PtlRPC traffic on LNet network `tcp0` must be encrypted.
- LNet networks `tcp1` and `o2ib0` are local physically secure networks that require high performance. Do not encrypt PtlRPC traffic on these networks.

1. Ensure that all non-MGS connections are authenticated and encrypted by default.

   ```
   mgs# lctl conf_param testfs.srpc.flavor.default=skpi
   ```

2. Override the file system default security flavor on LNet networks `tcp1` and `o2ib0` with `ska`. Security flavor `ska`provides authentication but without the performance impact of encryption and bulk RPC integrity.

   ```
   mgs# lctl conf_param testfs.srpc.flavor.tcp1=ska
   mgs# lctl conf_param testfs.srpc.flavor.o2ib0=ska
   ```

**Note**

Currently the "`lctl set_param -P`" format does not work with sptlrpc.

#### Listing Rules

To view the Secure RPC Config Rules, enter:

```
mgs# lctl get_param mgs.*.live.testfs
...
Secure RPC Config Rules:
testfs.srpc.flavor.tcp.cli2mdt=skpi
testfs.srpc.flavor.tcp.cli2ost=skpi
testfs.srpc.flavor.o2ib=ski
...
```

#### Deleting Rules

To delete a security flavor for an LNet network use the `conf_param -d` command to delete the flavor for that network:

For example, to delete the `testfs.srpc.flavor.o2ib1=ski` rule, enter:

```
mgs# lctl conf_param -d testfs.srpc.flavor.o2ib1
```

## SSK Key Files

SSK key files are a collection of attributes formatted as fixed length values and stored in a file, which are distributed by the administrator to client and server nodes. Attributes include:

- **Version** - Key file schema version number. Not user-defined.

- **Type** - A mandatory attribute that denotes the Lustre role of the key file consumer. Valid key types are:

  - **mgs** - for MGS when the `mgssec` `mount.lustre` option is used.
  - **server** - for MDS and OSS servers
  - **client** - for clients as well as servers who communicate with other servers in a client context (e.g. MDS communication with OSTs).

  

- **HMAC algorithm** - The Keyed-Hash Message Authentication Code algorithm used for integrity. Valid algorithms are (Default: SHA256):

  - SHA256
  - SHA512

  

- **Cryptographic algorithm** - Cipher for encryption. Valid algorithms are (Default: AES-256-CTR).

  - AES-256-CTR

- **Session security context expiration** - Seconds before session contexts generated from key expire and are regenerated (Default: 604800 seconds (7 days)).

- **Shared key length** - Shared key length in bits (Default: 256).

- **Prime length** - Length of prime (p) in bits used for the Diffie-Hellman Key Exchange (DHKE). (Default: 2048). This is generated only for client keys and can take a while to generate. This value also sets the minimum prime length that servers and MGS will accept from a client. Clients attempting to connect with a prime length less than the minimum will be rejected. In this way servers can guarantee the minimum encryption level that will be permitted.

- **File system name** - Lustre File system name for key.

- **MGS NIDs** - Comma-separated list of MGS NIDs. Only required when `mgssec` is used (Default: "").

- **Nodemap name** - Nodemap name for key (Default: "default"). See *the section called “Role of Nodemap in SSK”*

- **Shared key** - Shared secret used by all SSK flavors to provide authentication.

- **Prime (p)** - Prime used for the DHKE. This is only used for keys with `Type=client`.

**Note**

Key files provide a means to authenticate Lustre connections; always store and transfer key files securely. Key files must not be world writable or they will fail to load.

### Key File Management

The `lgss_sk` utility is used to write, modify, and read SSK key files. `lgss_sk` can be used to load key files singularly into the kernel keyring. `lgss_sk` options include:

**Table 10. lgss_sk Parameters**

| Parameter         | Value      | Description                                                  |
| ----------------- | ---------- | ------------------------------------------------------------ |
| `-l|--load`       | *filename* | Install key from file into user's session keyring. Must be executed by *root*. |
| `-m|--modify`     | *filename* | Modify a file's key attributes                               |
| `-r|--read`       | *filename* | Show file's key attributes                                   |
| `-w|--write`      | *filename* | Generate key file                                            |
| `-c|--crypt`      | *cipher*   | Cipher for encryption (Default: AES Counter mode)AES-256-CTR |
| `-i|--hmac`       | *hash*     | Hash algorithm for intregrity (Default: SHA256)SHA256 or SHA512 |
| `-e|--expire`     | *seconds*  | Seconds before contexts from key expire (Default: 604800 (7 days)) |
| `-f|--fsname`     | *name*     | File system name for key                                     |
| `-g|--mgsnids`    | *NID(s)*   | Comma separated list of MGS NID(s). Only required when mgssec is used (Default: "") |
| `-n|--nodemap`    | *map*      | Nodemap name for key (Default: "default")                    |
| `-p|--prime-bits` | *length*   | Prime length (p) for DHKE in bits (Default: 2048)            |
| `-t|--type`       | *type*     | Key type (mgs, server, client)                               |
| `-k|--key-bits`   | *length*   | Shared key length in bits (Default: 256)                     |
| `-d|--data`       | *file*     | Shared key random data source (Default: /dev/random)         |
| `-v|--verbose`    |            | Increase verbosity for errors                                |

#### Writing Key Files

Key files are generated by the `lgss_sk` utility. Parameters are specified on the command line followed by the `--write` parameter and the filename to write to. The `lgss_sk` utility will not overwrite files so the filename must be unique. Mandatory parameters for generating key files are `--type`, either `--fsname` or `--mgsnids`, and `--write`; all other parameters are optional.

`lgss_sk` uses `/dev/random` as the default entropy data source; you may override this with the `--data` parameter. When no hardware random number generator is available on the system where `lgss_sk` is executing, you may need to press keys on the keyboard or move the mouse (if directly attached to the system) or cause disk IO (if system is remote), in order to generate entropy for the shared key. It is possible to use `/dev/urandom` for testing purposes but this may provide less security in some cases.

Example:

To create a *server* type key file for the *testfs* Lustre file system for clients in the *biology* nodemap, enter:

```
server# lgss_sk -t server -f testfs -n biology \
-w testfs.server.biology.key
```

#### Modifying Key Files

Like writing key files you modify them by specifying the paramaters on the command line that you want to change. Only key file attributes associated with the parameters provided are changed; all other attributes remain unchanged.

To modify a key file's *Type* to *client* and populate the *Prime (p)* key attribute, if it is missing, enter:

```
client# lgss_sk -t client -m testfs.client.biology.key
```

To add MGS NIDs `192.168.1.101@tcp,10.10.0.101@o2ib` to server key file `testfs.server.biology.key` and client key file `testfs.client.biology.key`, enter

```
server# lgss_sk -g 192.168.1.101@tcp,10.10.0.101@o2ib \
-m testfs.server.biology.key

client# lgss_sk -g 192.168.1.101@tcp,10.10.0.101@o2ib \
-m testfs.client.biology.key
```

To modify the `testfs.server.biology.key` on the MGS to support MGS connections from *biology* clients, modify the key file's *Type* to include *mgs* in addition to *server*, enter:

```
mgs# lgss_sk -t mgs,server -m testfs.server.biology.key
```

#### Reading Key Files

Read key files with the `lgss_sk` utility and `--read` parameter. Read the keys modified in the previous examples:

```
mgs# lgss_sk -r testfs.server.biology.key
Version:        1
Type:           mgs server
HMAC alg:       SHA256
Crypt alg:      AES-256-CTR
Ctx Expiration: 604800 seconds
Shared keylen:  256 bits
Prime length:   2048 bits
File system:    testfs
MGS NIDs:       192.168.1.101@tcp 10.10.0.101@o2ib
Nodemap name:   biology
Shared key:
  0000: 84d2 561f 37b0 4a58 de62 8387 217d c30a  ..V.7.JX.b..!}..
  0010: 1caa d39c b89f ee6c 2885 92e7 0765 c917  .......l(....e..

client# lgss_sk -r testfs.client.biology.key
Version:        1
Type:           client
HMAC alg:       SHA256
Crypt alg:      AES-256-CTR
Ctx Expiration: 604800 seconds
Shared keylen:  256 bits
Prime length:   2048 bits
File system:    testfs
MGS NIDs:       192.168.1.101@tcp 10.10.0.101@o2ib
Nodemap name:   biology
Shared key:
  0000: 84d2 561f 37b0 4a58 de62 8387 217d c30a  ..V.7.JX.b..!}..
  0010: 1caa d39c b89f ee6c 2885 92e7 0765 c917  .......l(....e..
Prime (p) :
  0000: 8870 c3e3 09a5 7091 ae03 f877 f064 c7b5  .p....p....w.d..
  0010: 14d9 bc54 75f8 80d3 22f9 2640 0215 6404  ...Tu...".&@..d.
  0020: 1c53 ba84 1267 bea2 fb05 37a4 ed2d 5d90  .S...g....7..-].
  0030: 84e3 1a67 67f0 47c7 0c68 5635 f50e 9cf0  ...gg.G..hV5....
  0040: e622 6f53 2627 6af6 9598 eeed 6290 9b1e  ."oS&'j.....b...
  0050: 2ec5 df04 884a ea12 9f24 cadc e4b6 e91d  .....J...$......
  0060: 362f a239 0a6d 0141 b5e0 5c56 9145 6237  6/.9.m.A..\V.Eb7
  0070: 59ed 3463 90d7 1cbe 28d5 a15d 30f7 528b  Y.4c....(..]0.R.
  0080: 76a3 2557 e585 a1be c741 2a81 0af0 2181  v.%W.....A*...!.
  0090: 93cc a17a 7e27 6128 5ebd e0a4 3335 db63  ...z~'a(^...35.c
  00a0: c086 8d0d 89c1 c203 3298 2336 59d8 d7e7  ........2.#6Y...
  00b0: e52a b00c 088f 71c3 5109 ef14 3910 fcf6  .*....q.Q...9...
  00c0: 0fa0 7db7 4637 bb95 75f4 eb59 b0cd 4077  ..}.F7..u..Y..@w
  00d0: 8f6a 2ebd f815 a9eb 1b77 c197 5100 84c0  .j.......w..Q...
  00e0: 3dc0 d75d 40b3 6be5 a843 751a b09c 1b20  =..]@.k..Cu....
  00f0: 8126 4817 e657 b004 06b6 86fb 0e08 6a53  .&H..W........jS
```

#### Loading Key Files

Key files can be loaded into the kernel keyring with the `lgss_sk` utility or at mount time with the `skpath` mount option. The `skpath` method has the advantage that it accepts a directory path and loads all key files within the directory into the keyring. The `lgss_sk` utility loads a single key file into the keyring with each invocation. Key files must not be world writable or they will fail to load.

Third party tools can also load the keys if desired. The only caveat is that the key must be available when the request_key upcall to userspace is made and they use the correct key descriptions for a key so that it can be found during the upcall (see Key Descriptions).

Examples:

Load the `testfs.server.biology.key` key file using `lgss_sk`, enter:

```
server# lgss_sk -l testfs.server.biology.key
```

Use the `skpath` mount option to load all of the key files in the `/secure_directory` directory when mounting a storage target, enter:

```
server# mount -t lustre -o skpath=/secure_directory \
/storage/target /mount/point
```

Use the `skpath` mount option to load key files into the keyring on a client, enter:

```
client# mount -t lustre -o skpath=/secure_directory \
mgsnode:/testfs /mnt/testfs
```

## Lustre GSS Keyring

The Lustre GSS Keyring binary `lgss_keyring` is used by SSK to handle the upcall from kernel space into user space via `request-key`. The purpose of `lgss_keyring` is to create a token that is passed as part of the security context initialization RPC (SEC_CTX_INIT).

### Setup

The Lustre GSS keyring types of flavors utilize the Linux kernel keyring infrastructure to maintain keys as well as to perform the upcall from kernel space to userspace for key negotiation/establishment. The GSS keyring establishes a key type (see “request-key(8)”) named `lgssc` when the Lustre `ptlrpc_gss` kernel module is loaded. When a security context must be established it creates a key and uses the `request-key` binary in an upcall to establish the key. This key will look for the configuration file in `/etc/request-key.d` with the name *keytype*.conf, for Lustre this is `lgssc.conf`.

Each node participating in SSK Security must have a `/etc/request-key.d/lgssc.conf` file that contains the following single line:

`create lgssc * * /usr/sbin/lgss_keyring %o %k %t %d %c %u %g %T %P %S`

The `request-key` binary will call `lgss_keyring` with the arguments following it with their substituted values (see `request-key.conf(5)`).

### Server Setup

Lustre servers do not use the Linux `request-key` mechanism as clients do. Instead servers run a daemon that uses a pipefs with the kernel to trigger events based on read/write to a file descriptor. The server-side binary is `lsvcgssd`. It can be executed in the foreground or as a daemon. Below are the parameters for the `lsvcgssd` binary which requires various security flavors (`gssnull, krb5, sk`) to be enabled explicitly. This ensures that only required functionality is enabled.

**Table 11. lsvcgssd Parameters**

| Parameter | Description                           |
| --------- | ------------------------------------- |
| `-f`      | Run in foreground                     |
| `-n`      | Do not establish Kerberos credentials |
| `-v`      | Verbosity                             |
| `-m`      | Service MDS                           |
| `-o`      | Service OSS                           |
| `-g`      | Service MGS                           |
| `-k`      | Enable Kerberos support               |
| `-s`      | Enable Shared Key support             |
| `-z`      | Enable `gssnull` support              |

 

 

A SysV style init script is installed for starting and stopping the `lsvcgssd` daemon. The init script checks the`LSVCGSSARGS` variable in the `/etc/sysconfig/lsvcgss` configuration file for startup parameters.

Keys during the upcall on the client and handling of an RPC on the server are found by using a specific key description for each key in the kernel keyring.

For each MGS NID there must be a separate key loaded. The format of the key description should be:

**Table 12. Key Descriptions**

| Type            | Key Description               | Example                      |
| --------------- | ----------------------------- | ---------------------------- |
| MGC             | lustre:MGC*NID*               | `lustre:MGC192.168.1.10@tcp` |
| MDC/OSC/OSP/LWP | lustre:*fsname*               | `lustre:testfs`              |
| MDT             | lustre:*fsname*:*NodemapName* | `lustre:testfs:biology`      |
| OST             | lustre:*fsname*:*NodemapName* | `lustre:testfs:biology`      |
| MGS             | lustre:MGS                    | `lustre:MGS`                 |

 

 

All keys for Lustre use the `user` type for keys and are attached to the user’s keyring. This is not configurable. Below is an example showing how to list the user’s keyring, load a key file, read the key, and clear the key from the kernel keyring.

```
client# keyctl show
Session Keyring
  17053352 --alswrv      0     0  keyring: _ses
 773000099 --alswrv      0 65534   \_ keyring: _uid.0

client# lgss_sk -l /secure_directory/testfs.client.key

client# keyctl show
Session Keyring
  17053352 --alswrv      0     0  keyring: _ses
 773000099 --alswrv      0 65534   \_ keyring: _uid.0
1028795127 --alswrv      0     0       \_ user: lustre:testfs

client# keyctl pipe 1028795127 | lgss_sk -r -
Version:        1
Type:           client
HMAC alg:       SHA256
Crypt alg:      AES-256-CTR
Ctx Expiration: 604800 seconds
Shared keylen:  256 bits
Prime length:   2048 bits
File system:    testfs
MGS NIDs:
Nodemap name:   default
Shared key:
  0000: faaf 85da 93d0 6ffc f38c a5c6 f3a6 0408  ......o.........
  0010: 1e94 9b69 cf82 d0b9 880b f173 c3ea 787a  ...i.......s..xz
Prime (p) :
  0000: 9c12 ed95 7b9d 275a 229e 8083 9280 94a0  ....{.'Z".......
  0010: 8593 16b2 a537 aa6f 8b16 5210 3dd5 4c0c  .....7.o..R.=.L.
  0020: 6fae 2729 fcea 4979 9435 f989 5b6e 1b8a  o.')..Iy.5..[n..
  0030: 5039 8db2 3a23 31f0 540c 33cb 3b8e 6136  P9..:#1.T.3.;.a6
  0040: ac18 1eba f79f c8dd 883d b4d2 056c 0501  .........=...l..
  0050: ac17 a4ab 9027 4930 1d19 7850 2401 7ac4  .....'I0..xP$.z.
  0060: 92b4 2151 8837 ba23 94cf 22af 72b3 e567  ..!Q.7.#..".r..g
  0070: 30eb 0cd4 3525 8128 b0ff 935d 0ba3 0fc0  0...5%.(...]....
  0080: 9afa 5da7 0329 3ce9 e636 8a7d c782 6203  ..]..)<..6.}..b.
  0090: bb88 012e 61e7 5594 4512 4e37 e01d bdfc  ....a.U.E.N7....
  00a0: cb1d 6bd2 6159 4c3a 1f4f 1167 0e26 9e5e  ..k.aYL:.O.g.&.^
  00b0: 3cdc 4a93 63f6 24b1 e0f1 ed77 930b 9490  <.J.c.$....w....
  00c0: 25ef 4718 bff5 033e 11ba e769 4969 8a73  %.G....>...iIi.s
  00d0: 9f5f b7bb 9fa0 7671 79a4 0d28 8a80 1ea1  ._....vqy..(....
  00e0: a4df 98d6 e20e fe10 8190 5680 0d95 7c83  ..........V...|.
  00f0: 6e21 abb3 a303 ff55 0aa8 ad89 b8bf 7723  n!.....U......w#

client# keyctl clear @u

client# keyctl show
Session Keyring
  17053352 --alswrv      0     0  keyring: _ses
 773000099 --alswrv      0 65534   \_ keyring: _uid.0
```

### Debugging GSS Keyring

Lustre client and server support several debug levels, which can be seen below.

Debug levels:

- 0 - Error
- 1 - Warn
- 2 - Info
- 3 - Debug
- 4 - Trace

To set the debug level on the client use the Lustre parameter:

`sptlrpc.gss.lgss_keyring.debug_level`

For example to set the debug level to trace, enter:

```
client# lctl set_param sptlrpc.gss.lgss_keyring.debug_level=4
```

Server-side verbosity is increased by adding additional verbose flags (`-v`) to the command line arguments for the daemon. The following command runs the `lsvcgssd` daemon in the foreground with debug verbosity supporting gssnull and SSK

```
server# lsvcgssd -f -vvv -z -s
```

`lgss_keyring` is called as part of the `request-key` upcall which has no standard output; therefore logging is done through syslog. The server-side logging with `lsvcgssd` is written to standard output when executing in the foreground and to syslog in daemon mode.

### Revoking Keys

The keys discussed above with `lgss_sk` and the `skpath` mount options are not revoked. They are only used to create valid contexts for client connections. Instead of revoking them they can be invalidated in one of two ways.

- Unloading the key from the user keyring on the server will cause new client connections to fail. If no longer necessary it can be deleted.
- Changing the nodemap name for the clients on the servers. Since the nodemap is an integral part of the shared key context instantiation, renaming the nodemap a group of NIDs belongs to will prevent any new contexts.

There currently does not exist a mechanism to flush contexts from Lustre. Targets could be unmounted from the servers to purge contexts. Alternatively shorter context expiration could be used when the key is created so that contexts need to be refreshed more frequently than the default. 3600 seconds could be reasonable depending on the use case so that contexts will have to be renegotiated every hour.

## Role of Nodemap in SSK

SSK uses Nodemap (See *Mapping UIDs and GIDs with Nodemap*) policy group names and their associated NID range(s) as a mechanism to prevent key file forgery, and to control the range of NIDs on which a given key file can be used.

Clients assume they are in the nodemap specified in the key file they use. When clients instantiate security contexts an upcall is triggered that specifies information about the context that triggers it. From this context information `request-key` calls `lgss_keyring`, which in turn looks up the key with description lustre:*fsname* or lustre:*target_name* for the MGC. Using the key found in the user keyring matching the description, the nodemap name is read from the key, hashed with SHA256, and sent to the server.

Servers look up the client’s NID to determine which nodemap the NID is associated with and sends the nodemap name to `lsvcgssd`. The `lsvcgssd` daemon verifies whether the HMAC equals the nodemap value sent by the client. This prevents forgery and invalidates the key when a client’s NID is not associated with the nodemap name defined on the servers.

It is not required to activate the Nodemap feature in order for SSK to perform client NID to nodemap name lookups.

## SSK Examples

The examples in this section use 1 MGS/MDS (NID 172.16.0.1@tcp), 1 OSS (NID 172.16.0.3@tcp), and 2 clients. The Lustre file system name is *testfs*.

### Securing Client to Server Communications

This example illustrates how to configure SSK to apply Privacy and Integrity protections to client-to-server PtlRPC traffic on the `tcp` network. Rules that specify a direction, specifically `cli2mdt` and `cli2ost`, are used. This permits server-to-server communications to continue using `null` which is the *default* flavor for all Lustre connections. This arrangement provides no server-to-server protections, see *the section called “Securing Server to Server Communications”*.

1. Create secure directory for storing SSK key files.

   ```
   mds# mkdir /secure_directory
   mds# chmod 600 /secure_directory
   oss# mkdir /secure_directory
   oss# chmod 600 /secure_directory
   cli1# mkdir /secure_directory
   cli1# chmod 600 /secure_directory
   cli2# mkdir /secure_directory
   cli2# chmod 600 /secure_directory
   ```

2. Generate a key file for the MDS and OSS servers. Run:

   ```
   mds# lgss_sk -t server -f testfs -w \
   /secure_directory/testfs.server.key
   ```

3. Securely copy the /secure_directory/testfs.server.key key file to the OSS.

   ```
   mds# scp /secure_directory/testfs.server.key \
   oss:/secure_directory/
   ```

4. Securely copy the `/secure_directory/testfs.server.key` key file to`/secure_directory/testfs.client.key` on *client1*.

   ```
   mds# scp /secure_directory/testfs.server.key \
   client1:/secure_directory/testfs.client.key
   ```

5. Modify the key file type to `client` on *client1*. This operation also generates a prime number of `Prime length`to populate the `Prime (p)` attribute. Run:

   ```
   client1# lgss_sk -t client \
   -m /secure_directory/testfs.client.key
   ```

6. Create a `/etc/request-key.d/lgssc.conf` file on all nodes that contains this line '`create lgssc * * /usr/sbin/lgss_keyring %o %k %t %d %c %u %g %T %P %S`' without the single quotes. Run:

   ```
   mds# echo create lgssc \* \* /usr/sbin/lgss_keyring %o %k %t %d %c %u %g %T %P %S > /etc/request-key.d/lgssc.conf
   oss# echo create lgssc \* \* /usr/sbin/lgss_keyring %o %k %t %d %c %u %g %T %P %S > /etc/request-key.d/lgssc.conf
   client1# echo create lgssc \* \* /usr/sbin/lgss_keyring %o %k %t %d %c %u %g %T %P %S > /etc/request-key.d/lgssc.conf
   client2# echo create lgssc \* \* /usr/sbin/lgss_keyring %o %k %t %d %c %u %g %T %P %S > /etc/request-key.d/lgssc.conf
   ```

7. Configure the `lsvcgss` daemon on the MDS and OSS. Set the `LSVCGSSDARGS` variable in`/etc/sysconfig/lsvcgss` on the MDS to `‘-s -m’`. On the OSS, set the `LSVCGSSDARGS` variable in`/etc/sysconfig/lsvcgss` to `‘-s -o’`

8. Start the `lsvcgssd` daemon on the MDS and OSS. Run:

   ```
   mds# systemctl start lsvcgss.service
   oss# systemctl start lsvcgss.service
   ```

9. Mount the MDT and OST with the `-o skpath=/secure_directory` mount option. The `skpath` option loads all SSK key files found in the directory into the kernel keyring.

10. Set client to MDT and client to OST security flavor to SSK Privacy and Integrity, `skpi`:

    ```
    mds# lctl conf_param testfs.srpc.flavor.tcp.cli2mdt=skpi
    mds# lctl conf_param testfs.srpc.flavor.tcp.cli2ost=skpi
    ```

11. Mount the testfs file system on client1 and client2:

    ```
    client1# mount -t lustre -o skpath=/secure_directory 172.16.0.1@tcp:/testfs /mnt/testfs
    client2# mount -t lustre -o skpath=/secure_directory 172.16.0.1@tcp:/testfs /mnt/testfs
    mount.lustre: mount 172.16.0.1@tcp:/testfs at /mnt/testfs failed: Connection refused
    ```

12. *client2* failed to authenticate because it does not have a valid key file. Repeat steps 4 and 5, substitute client1 for client2, then mount the testfs file system on client2:

    ```
    client2# mount -t lustre -o skpath=/secure_directory 172.16.0.1@tcp:/testfs /mnt/testfs
    ```

13. Verify that the `mdc` and `osc` connections are using the SSK mechanism and that `rpc` and `bulk` security flavors are `skpi`. See *the section called “Viewing Secure PtlRPC Contexts”*.

    Notice the `mgc` connection to the MGS has no secure PtlRPC security context. This is because `skpi` security was only specified for client-to-MDT and client-to-OST connections in step 10. The following example details the steps necessary to secure the connection to the MGS.

### Securing MGS Communications

This example builds on the previous example.

1. Enable `lsvcgss` MGS service support on MGS. Edit `/etc/sysconfig/lsvcgss` on the MGS and add the (`-g`) parameter to the `LSVCGSSDARGS` variable. Restart the `lsvcgss` service.

2. Add *mgs* key type and *MGS NIDs* to `/secure_directory/testfs.server.key` on MDS.

   ```
   mgs# lgss_sk -t mgs,server -g 172.16.0.1@tcp,172.16.0.2@tcp -m /secure_directory/testfs.server.key
   ```

3. Load the modified key file on the MGS. Run:

   ```
   mgs# lgss_sk -l /secure_directory/testfs.server.key
   ```

4. Add *MGS NIDs* to `/secure_directory/testfs.client.key` on client, client1.

   ```
   client1# lgss_sk -g 172.16.0.1@tcp,172.16.0.2@tcp -m /secure_directory/testfs.client.key
   ```

5. Unmount the testfs file system on client1, then mount with the `mgssec=skpi` mount option:

   ```
   cli1# mount -t lustre -o mgssec=skpi,skpath=/secure_directory 172.16.0.1@tcp:/testfs /mnt/testfs
   ```

6. Verify that client1’s MGC connection is using the SSK mechanism and `skpi` security flavor. See *the section called “Viewing Secure PtlRPC Contexts”*.

### Securing Server to Server Communications

This example illustrates how to configure SSK to apply *Integrity* protection, `ski` flavor, to MDT-to-OST PtlRPC traffic on the `tcp` network.

This example builds on the previous example.

1. Create a Nodemap policy group named `LustreServers` on the MGS for the Lustre Servers, enter:

   ```
   mgs# lctl nodemap_add LustreServers
   ```

2. Add MDS and OSS NIDs to the LustreServers nodemap, enter:

   ```
   mgs# lctl nodemap_add_range --name LustreServers --range 172.16.0.[1-3]@tcp
   ```

3. Create key file of type `mgs,server` for use with nodes in the *LustreServers* Nodemap range.

   ```
   mds# lgss_sk -t mgs,server -f testfs -g \
   172.16.0.1@tcp,172.16.0.2@tcp -n LustreServers -w \
   /secure_directory/testfs.LustreServers.key
   ```

4. Securely copy the `/secure_directory/testfs.LustreServers.key` key file to the OSS.

   ```
   mds# scp /secure_directory/testfs.LustreServers.key oss:/secure_directory/
   ```

5. On the MDS and OSS, copy `/secure_directory/testfs.LustreServers.key` to`/secure_directory/testfs.LustreServers.client.key`.

6. On each server modify the key file type of `/secure_directory/testfs.LustreServers.client.key` to be of type client. This operation also generates a prime number of *Prime length* to populate the *Prime (p)* attribute. Run:

   ```
   mds# lgss_sk -t client -m \
   /secure_directory/testfs.LustreServers.client.key
   oss# lgss_sk -t client -m \
   /secure_directory/testfs.LustreServers.client.key
   ```

7. Load the `/secure_directory/testfs.LustreServers.key` and`/secure_directory/testfs.LustreServers.client.key` key files into the keyring on the MDS and OSS, enter:

   ```
   mds# lgss_sk -l /secure_directory/testfs.LustreServers.key
   mds# lgss_sk -l /secure_directory/testfs.LustreServers.client.key
   oss# lgss_sk -l /secure_directory/testfs.LustreServers.key
   oss# lgss_sk -l /secure_directory/testfs.LustreServers.client.key
   ```

8. Set MDT to OST security flavor to SSK Integrity, `ski`:

   ```
   mds# lctl conf_param testfs.srpc.flavor.tcp.mdt2ost=ski
   ```

9. Verify that the `osc` and `osp` connections to the OST have a secure `ski` security context. See *the section called “Viewing Secure PtlRPC Contexts”*.

## Viewing Secure PtlRPC Contexts

From the client (or servers which have mgc, osc, mdc contexts) you can view info regarding all users’ contexts and the flavor in use for an import. For user’s contexts (srpc_context), SSK and gssnull only support a single root UID so there should only be one context. The other file in the import (srpc_info) has additional sptlrpc details. The `rpc`and `bulk` flavors allow you to verify which security flavor is in use.

```
client1# lctl get_param *.*.srpc_*
mdc.testfs-MDT0000-mdc-ffff8800da9f0800.srpc_contexts=
ffff8800da9600c0: uid 0, ref 2, expire 1478531769(+604695), fl uptodate,cached,, seq 7, win 2048, key 27a24430(ref 1), hdl 0xf2020f47cbffa93d:0xc23f4df4bcfb7be7, mech: sk
mdc.testfs-MDT0000-mdc-ffff8800da9f0800.srpc_info=
rpc flavor:     skpi
bulk flavor:    skpi
flags:          rootonly,udesc,
id:             3
refcount:       3
nctx:   1
gc internal     3600
gc next 3505
mgc.MGC172.16.0.1@tcp.srpc_contexts=
ffff8800dbb09b40: uid 0, ref 2, expire 1478531769(+604695), fl uptodate,cached,, seq 18, win 2048, key 3e3f709f(ref 1), hdl 0xf2020f47cbffa93b:0xc23f4df4bcfb7be6, mech: sk
mgc.MGC172.16.0.1@tcp.srpc_info=
rpc flavor:     skpi
bulk flavor:    skpi
flags:          -,
id:             2
refcount:       3
nctx:   1
gc internal     3600
gc next 3505
osc.testfs-OST0000-osc-ffff8800da9f0800.srpc_contexts=
ffff8800db9e5600: uid 0, ref 2, expire 1478531770(+604696), fl uptodate,cached,, seq 3, win 2048, key 3f7c1d70(ref 1), hdl 0xf93e61c64b6b415d:0xc23f4df4bcfb7bea, mech: sk
osc.testfs-OST0000-osc-ffff8800da9f0800.srpc_info=
rpc flavor:     skpi
bulk flavor:    skpi
flags:          rootonly,bulk,
id:             6
refcount:       3
nctx:   1
gc internal     3600
gc next 3505
```

 

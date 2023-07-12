# Preface

The Lustre*Software Release 2.x Operations Manual provides detailed information and procedures to install, configure and tune a Lustre file system. The manual covers topics such as failover, quotas, striping, and bonding. This manual also contains troubleshooting information and tips to improve the operation and performance of a Lustre file system.

- [Preface](#preface)
  * [About this Document](#about-this-document)
    + [UNIX* Commands](#unix-commands)
    + [Shell Prompts](#shell-prompts)
    + [Related Documentation](#related-documentation)
    + [Documentation and Support](#documentation-and-support)
  * [Revisions](#revisions)
    + [which version?](#which-version)



## About this Document

This document is maintained by Whamcloud in Docbook format. The canonical version is available at<http://wiki.whamcloud.com/display/PUB/Documentation>.



### UNIX* Commands

This document may not contain information about basic UNIX* operating system commands and procedures such as shutting down the system, booting the system, and configuring devices. Refer to the following for this information:

- Software documentation that you received with your system

- Red Hat* Enterprise Linux* documentation, which is at: <http://docs.redhat.com/docs/en-US/index.html>

  ### Note

  The Lustre client module is available for many different Linux* versions and distributions. The Red Hat Enterprise Linux distribution is the best supported and tested platform for Lustre servers.



### Shell Prompts

The shell prompt used in the example text indicates whether a command can or should be executed by a regular user, or whether it requires superuser permission to run. Also, the machine type is often included in the prompt to indicate whether the command should be run on a client node, on an MDS node, an OSS node, or the MGS node.

Some examples are listed below, but other prompt combinations are also used as needed for the example.

| **Shell**                  | **Prompt** |
| -------------------------- | ---------- |
| Regular user               | `machine$` |
| Superuser (root)           | `machine#` |
| Regular user on the client | `client$`  |
| Superuser on the MDS       | `mds#`     |
| Superuser on the OSS       | `oss#`     |
| Superuser on the MGS       | `mgs#`     |





### Related Documentation

| **Application**    | **Title**                                       | **Format** | **Location**                                                 |
| ------------------ | ----------------------------------------------- | ---------- | ------------------------------------------------------------ |
| Latest information | *Lustre Software Release 2.x Change Logs*       | Wiki page  | Online at <http://wiki.whamcloud.com/display/PUB/Documentation> |
| Service            | *Lustre Software Release 2.x Operations Manual* | PDFHTML    | Online at <http://wiki.whamcloud.com/display/PUB/Documentation> |

 

### Documentation and Support

These web sites provide additional resources:

- Documentation <http://wiki.whamcloud.com/display/PUB/Documentation> <http://www.lustre.org>
- Support <https://jira.whamcloud.com/>



## Revisions

The Lustre* File System Release 2.x Operations Manual is a community maintained work. Versions of the manual are continually built as suggestions for changes and improvements arrive. Suggestions for improvements can be submitted through the ticketing system maintained at <https://jira.whamcloud.com/browse/LUDOC>. Instructions for providing a patch to the existing manual are available at: <http://wiki.lustre.org/Lustre_Manual_Changes>.

This manual currently covers all the 2.x Lustre software releases. Features that are specific to individual releases are identified within the table of contents using a short hand notation (i.e. 'L24' is a Lustre software release 2.4 specific feature), and within the text using a distinct box. For example:

Introduced in Lustre 2.4Lustre software release version 2.4 includes support for multiple metadata servers.

### which version?

The current version of Lustre that is in use on the client can be found using the command `lctl get_param version`, for example:

```
$ lctl get_param version
version=
lustre: 2.7.59
kernel: patchless_client
build:  v2_7_59_0-g703195a-CHANGED-3.10.0.lustreopa
```

Only the latest revision of this document is made readily available because changes are continually arriving. The current and latest revision of this manual is available from links maintained at: <http://lustre.opensfs.org/documentation/>.

| **Revision History**        |                                     |      |
| --------------------------- | ----------------------------------- | ---- |
| Revision 0                  | Built on 11 May 2019 23:15:21-04:00 |      |
| Continuous build of Manual. |                                     |      |
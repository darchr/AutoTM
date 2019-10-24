#!/bin/bash

# Unmount any existing devices
echo "Unmounting Any Existing Devices"
umount -f /mnt

# Destroy and recreate namespace
echo "Destroying Old Namespaces"
ndctl destroy-namespace --force namespace1.0

# Create a new namespace
echo "Creating New Namespaces"
sudo ndctl create-namespace --type=pmem --mode=fsdax --region=region1

### Create a partition on the devise, beginning at a 2MiB boundary

echo "Creating Device partition table"
TGTDEV=/dev/pmem1
wipefs ${TGTDEV}

# to create the partitions programatically (rather than manually)
# we're going to simulate the manual input to fdisk
# The sed script strips off all the comments so that we can 
# document what we're doing in-line with the actual commands
# Note that a blank line (commented as "default" will send a empty
# line terminated with a newline to take the fdisk default.
sed -e 's/\s*\([\+0-9a-zA-Z]*\).*/\1/' << EOF | fdisk ${TGTDEV}
  g # create a new empty GPT partition table
  n # add a new partition
  1 # partition number 1
  4096 # Begin first sector at 4096 blocks (4096 * 512 = 2MiB)
    # default - end of disk
  w # write the partition table
  q # we're done
EOF

# Make a file system
sudo mkfs.ext4 -b 4096 -E stride=512 -F /dev/pmem1

# # Mount the NVDIMM
echo "Mounting Device"
mount -o dax /dev/pmem1 /mnt
mkdir /mnt/public
chmod 777 /mnt/public

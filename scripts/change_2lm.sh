#!/bin/bash

# Unmount any existing devices
echo "Unmounting Any Existing Devices"
umount -f /mnt

# Destroy and recreate namespace
echo "Destroying Old Namespaces"
ndctl destroy-namespace --force namespace1.0

echo "Setting Goal"
ipmctl create -goal MemoryMode=100

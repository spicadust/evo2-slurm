#!/bin/bash
key=~/.ssh/erda.ed25519
user=nxw782@alumni.ku.dk
erdadir=llm_matrix
mnt=/mnt/llm_matrix
if [ -f "$key" ]
then
    mkdir -p ${mnt}
    sshfs ${user}@io.erda.dk:${erdadir} ${mnt} -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 -o IdentityFile=${key}
else
    echo "'${key}' is not an ssh key"
fi

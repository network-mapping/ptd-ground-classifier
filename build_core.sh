#!/bin/sh
# Rebuilds the compiled core of the fast engine from ptdcore.c.
# Only needed on a machine the shipped libptdcore.so does not suit
# (anything other than 64-bit x86 Linux), or after editing ptdcore.c.
# Needs gcc or clang; nothing else. The result has no dependencies.
set -e
cd "$(dirname "$0")"
CC="${CC:-gcc}"
"$CC" -O2 -Wall -Wextra -Wframe-larger-than=3800 -shared -fPIC -ffreestanding -nostdlib -o libptdcore.so ptdcore.c
echo "built libptdcore.so with $CC"
# Windows DLL, cross-built from Linux with clang + lld (this is how ptdcore.dll was made):
#   clang --target=x86_64-pc-windows-msvc -O2 -ffreestanding -fno-builtin -fno-stack-protector -c ptdcore.c -o ptdcore.obj
#   lld-link /dll /noentry /nodefaultlib /out:ptdcore.dll ptdcore.obj
# On Windows itself, with LLVM installed:
#   clang-cl /O2 /LD ptdcore.c /link /noentry /nodefaultlib /out:ptdcore.dll

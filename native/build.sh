#!/bin/sh
# Builds the compiled core (native/ptdcore.c) into build/.
#   macOS  -> build/libptdcore.dylib     Linux -> build/libptdcore.so
# Needs clang or gcc (CC=...); the result has no dependencies.
# -ffp-contract=off: no fused multiply-add, so results match across CPUs (Apple Silicon fuses by default).
# -fno-stack-protector: the core links no C library, so the guard symbols would be unresolved.
set -e
cd "$(dirname "$0")/.."
mkdir -p build
case "$(uname -s)" in
    Darwin) OUT=build/libptdcore.dylib; CC="${CC:-clang}"; LINK="" ;;   # macOS links libSystem implicitly (-nostdlib is refused); the core calls none of it
    *)      OUT=build/libptdcore.so;    CC="${CC:-gcc}";   LINK="-nostdlib" ;;
esac
"$CC" -O2 -Wall -Wextra -Wframe-larger-than=3800 -ffp-contract=off -fno-stack-protector \
    -shared -fPIC -ffreestanding $LINK -o "$OUT" native/ptdcore.c
echo "built $OUT with $CC"
# Windows DLL, cross-built with clang + lld:
#   clang --target=x86_64-pc-windows-msvc -O2 -ffp-contract=off -ffreestanding -fno-builtin -fno-stack-protector -c native/ptdcore.c -o build/ptdcore.obj
#   lld-link /dll /noentry /nodefaultlib /out:build/ptdcore.dll build/ptdcore.obj
# On Windows itself, with LLVM installed:
#   clang-cl /O2 /LD native\ptdcore.c /link /noentry /nodefaultlib /out:build\ptdcore.dll

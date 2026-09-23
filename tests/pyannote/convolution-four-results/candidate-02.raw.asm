; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:Kernel512Four(ptr,ptr,ptr,int,int,int,int,int,int,int) (Instrumented Tier0)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Instrumented Tier0 code
; rbp based frame
; fully interruptible
; compiling with minopt

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       lea      r11, [rsp-0x2240]
       call     CORINFO_HELP_STACK_PROBE
       mov      rsp, r11
       lea      rbp, [rsp+0x2240]
       vxorps   xmm8, xmm8, xmm8
       vmovdqa  xmmword ptr [rbp-0x1270], xmm8
       vmovdqa  xmmword ptr [rbp-0x1260], xmm8
       mov      rax, -0x1200
       vmovdqa  xmmword ptr [rbp+rax-0x50], xmm8
       vmovdqa  xmmword ptr [rbp+rax-0x40], xmm8
       vmovdqa  xmmword ptr [rbp+rax-0x30], xmm8
       add      rax, 48
       jne      SHORT  -5 instr
       mov      dword ptr [rbp-0x50], eax
       mov      qword ptr [rbp-0x30], rdi
       mov      qword ptr [rbp-0x38], rsi
       mov      qword ptr [rbp-0x40], rdx
       mov      dword ptr [rbp-0x44], ecx
       mov      dword ptr [rbp-0x48], r8d
       mov      dword ptr [rbp-0x4C], r9d
 
G_M000_IG02:                ;; offset=0x006A
       mov      dword ptr [rbp-0x2238], 0x3E8
       mov      eax, dword ptr [rbp-0x4C]
       add      eax, 2
       mov      dword ptr [rbp-0x50], eax
       mov      eax, dword ptr [rbp+0x10]
       add      eax, 2
       mov      dword ptr [rbp-0x54], eax
       mov      eax, dword ptr [rbp+0x20]
       imul     eax, dword ptr [rbp+0x28]
       mov      dword ptr [rbp-0x58], eax
       mov      eax, dword ptr [rbp-0x58]
       sar      eax, 31
       and      eax, 7
       add      eax, dword ptr [rbp-0x58]
       sar      eax, 3
       shl      eax, 3
       mov      dword ptr [rbp-0x5C], eax
       mov      eax, dword ptr [rbp-0x54]
       shl      eax, 4
       mov      dword ptr [rbp-0x60], eax
       mov      eax, dword ptr [rbp-0x50]
       imul     eax, dword ptr [rbp-0x60]
       mov      dword ptr [rbp-0x64], eax
       mov      eax, dword ptr [rbp+0x18]
       shl      eax, 4
       mov      dword ptr [rbp-0x68], eax
       xor      eax, eax
       mov      dword ptr [rbp-0x6C], eax
       jmp      G_M000_IG67
 
G_M000_IG03:                ;; offset=0x00CB
       xor      eax, eax
       mov      dword ptr [rbp-0x70], eax
       jmp      G_M000_IG64
 
G_M000_IG04:                ;; offset=0x00D5
       xor      eax, eax
       mov      dword ptr [rbp-0x74], eax
       jmp      G_M000_IG41
 
G_M000_IG05:                ;; offset=0x00DF
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0xF0], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x130], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x170], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x1B0], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x1F0], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x230], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x270], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x2B0], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x2F0], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x330], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x370], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x3B0], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x3F0], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x430], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x470], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x4B0], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x4F0], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x530], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x570], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x5B0], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x5F0], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x630], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x670], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x6B0], zmm0
       mov      eax, dword ptr [rbp-0x6C]
       imul     eax, dword ptr [rbp-0x44]
       lea      eax, [rax+8*rax]
       cdqe     
       mov      rcx, qword ptr [rbp-0x38]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x6B8], rax
       mov      eax, dword ptr [rbp-0x44]
       shl      eax, 4
       lea      eax, [rax+8*rax]
       cdqe     
 
G_M000_IG06:                ;; offset=0x0255
       mov      rcx, qword ptr [rbp-0x6B8]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x6C0], rax
       mov      eax, dword ptr [rbp-0x44]
       shl      eax, 4
       lea      eax, [rax+8*rax]
       cdqe     
       mov      rcx, qword ptr [rbp-0x6C0]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x6C8], rax
       mov      eax, dword ptr [rbp-0x44]
       shl      eax, 4
       lea      eax, [rax+8*rax]
       cdqe     
       mov      rcx, qword ptr [rbp-0x6C8]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x6D0], rax
       mov      eax, dword ptr [rbp-0x70]
       imul     eax, dword ptr [rbp+0x18]
       imul     eax, dword ptr [rbp-0x54]
       mov      ecx, dword ptr [rbp-0x74]
       imul     ecx, dword ptr [rbp+0x18]
       add      eax, ecx
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x30]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x6D8], rax
       xor      eax, eax
       mov      dword ptr [rbp-0x6DC], eax
       jmp      G_M000_IG32
 
G_M000_IG07:                ;; offset=0x02D6
       mov      rdi, 0x72A8A2D4EF80
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x6DC]
       sar      eax, 4
       imul     eax, dword ptr [rbp-0x64]
       cdqe     
       shl      rax, 2
       add      rax, qword ptr [rbp-0x6D8]
       mov      ecx, dword ptr [rbp-0x6DC]
       and      ecx, 15
       movsxd   rcx, ecx
       lea      rax, [rax+4*rcx]
       mov      qword ptr [rbp-0x1278], rax
       movsxd   rax, dword ptr [rbp-0x60]
       mov      rcx, qword ptr [rbp-0x1278]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x6E8], rax
       movsxd   rax, dword ptr [rbp-0x60]
       mov      rcx, qword ptr [rbp-0x6E8]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x6F0], rax
       mov      rax, qword ptr [rbp-0x6B8]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0x730], zmm0
       mov      rax, qword ptr [rbp-0x6C0]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0x770], zmm0
       mov      rax, qword ptr [rbp-0x6C8]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0x7B0], zmm0
       mov      rax, qword ptr [rbp-0x6D0]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0x7F0], zmm0
       mov      rax, qword ptr [rbp-0x1278]
       vbroadcastss zmm0, dword ptr [rax]
       vmovups  zmmword ptr [rbp-0x12F0], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x12F0]
       vmovups  zmm1, zmmword ptr [rbp-0xF0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x730]
       vmovups  zmmword ptr [rbp-0xF0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x12F0]
       vmovups  zmm1, zmmword ptr [rbp-0x130]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x770]
       vmovups  zmmword ptr [rbp-0x130], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x12F0]
       vmovups  zmm1, zmmword ptr [rbp-0x170]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x7B0]
       vmovups  zmmword ptr [rbp-0x170], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x12F0]
       vmovups  zmm1, zmmword ptr [rbp-0x1B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x7F0]
       vmovups  zmmword ptr [rbp-0x1B0], zmm1
       movsxd   rax, dword ptr [rbp-0x68]
       mov      rcx, qword ptr [rbp-0x1278]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x1330], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x1330]
       vmovups  zmm1, zmmword ptr [rbp-0x1F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x730]
       vmovups  zmmword ptr [rbp-0x1F0], zmm1
 
G_M000_IG08:                ;; offset=0x0499
       vmovups  zmm0, zmmword ptr [rbp-0x1330]
       vmovups  zmm1, zmmword ptr [rbp-0x230]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x770]
       vmovups  zmmword ptr [rbp-0x230], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1330]
       vmovups  zmm1, zmmword ptr [rbp-0x270]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x7B0]
       vmovups  zmmword ptr [rbp-0x270], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1330]
       vmovups  zmm1, zmmword ptr [rbp-0x2B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x7F0]
       vmovups  zmmword ptr [rbp-0x2B0], zmm1
       mov      eax, dword ptr [rbp-0x68]
       add      eax, eax
       cdqe     
       mov      rcx, qword ptr [rbp-0x1278]
       vmovss   xmm0, dword ptr [rcx+4*rax]
       vmovss   dword ptr [rbp-0x1334], xmm0
       vbroadcastss zmm0, dword ptr [rbp-0x1334]
       vmovups  zmmword ptr [rbp-0x13B0], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x13B0]
       vmovups  zmm1, zmmword ptr [rbp-0x2F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x730]
       vmovups  zmmword ptr [rbp-0x2F0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x13B0]
       vmovups  zmm1, zmmword ptr [rbp-0x330]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x770]
       vmovups  zmmword ptr [rbp-0x330], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x13B0]
       vmovups  zmm1, zmmword ptr [rbp-0x370]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x7B0]
       vmovups  zmmword ptr [rbp-0x370], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x13B0]
       vmovups  zmm1, zmmword ptr [rbp-0x3B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x7F0]
       vmovups  zmmword ptr [rbp-0x3B0], zmm1
       mov      eax, dword ptr [rbp-0x68]
       lea      eax, [rax+2*rax]
       cdqe     
       mov      rcx, qword ptr [rbp-0x1278]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x13F0], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x13F0]
       vmovups  zmm1, zmmword ptr [rbp-0x3F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x730]
       vmovups  zmmword ptr [rbp-0x3F0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x13F0]
       vmovups  zmm1, zmmword ptr [rbp-0x430]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x770]
       vmovups  zmmword ptr [rbp-0x430], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x13F0]
       vmovups  zmm1, zmmword ptr [rbp-0x470]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x7B0]
       vmovups  zmmword ptr [rbp-0x470], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x13F0]
       vmovups  zmm1, zmmword ptr [rbp-0x4B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x7F0]
       vmovups  zmmword ptr [rbp-0x4B0], zmm1
       mov      eax, dword ptr [rbp-0x68]
 
G_M000_IG09:                ;; offset=0x06A3
       shl      eax, 2
       cdqe     
       mov      rcx, qword ptr [rbp-0x1278]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x1430], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x1430]
       vmovups  zmm1, zmmword ptr [rbp-0x4F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x730]
       vmovups  zmmword ptr [rbp-0x4F0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1430]
       vmovups  zmm1, zmmword ptr [rbp-0x530]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x770]
       vmovups  zmmword ptr [rbp-0x530], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1430]
       vmovups  zmm1, zmmword ptr [rbp-0x570]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x7B0]
       vmovups  zmmword ptr [rbp-0x570], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1430]
       vmovups  zmm1, zmmword ptr [rbp-0x5B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x7F0]
       vmovups  zmmword ptr [rbp-0x5B0], zmm1
       mov      eax, dword ptr [rbp-0x68]
       lea      eax, [rax+4*rax]
       cdqe     
       mov      rcx, qword ptr [rbp-0x1278]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x1470], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x1470]
       vmovups  zmm1, zmmword ptr [rbp-0x5F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x730]
       vmovups  zmmword ptr [rbp-0x5F0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1470]
       vmovups  zmm1, zmmword ptr [rbp-0x630]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x770]
       vmovups  zmmword ptr [rbp-0x630], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1470]
       vmovups  zmm1, zmmword ptr [rbp-0x670]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x7B0]
       vmovups  zmmword ptr [rbp-0x670], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1470]
       vmovups  zmm1, zmmword ptr [rbp-0x6B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x7F0]
       vmovups  zmmword ptr [rbp-0x6B0], zmm1
       mov      rax, qword ptr [rbp-0x6B8]
       add      rax, 64
       mov      qword ptr [rbp-0x6B8], rax
       mov      rax, qword ptr [rbp-0x6C0]
       add      rax, 64
       mov      qword ptr [rbp-0x6C0], rax
       mov      rax, qword ptr [rbp-0x6C8]
       add      rax, 64
       mov      qword ptr [rbp-0x6C8], rax
       mov      rax, qword ptr [rbp-0x6D0]
       add      rax, 64
       mov      qword ptr [rbp-0x6D0], rax
       mov      rax, qword ptr [rbp-0x6B8]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0x830], zmm0
 
G_M000_IG10:                ;; offset=0x087F
       mov      rax, qword ptr [rbp-0x6C0]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0x870], zmm0
       mov      rax, qword ptr [rbp-0x6C8]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0x8B0], zmm0
       mov      rax, qword ptr [rbp-0x6D0]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0x8F0], zmm0
       mov      rax, qword ptr [rbp-0x1278]
       add      rax, 64
       mov      qword ptr [rbp-0x1478], rax
       mov      rax, qword ptr [rbp-0x1478]
       vbroadcastss zmm0, dword ptr [rax]
       vmovups  zmmword ptr [rbp-0x14F0], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x14F0]
       vmovups  zmm1, zmmword ptr [rbp-0xF0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x830]
       vmovups  zmmword ptr [rbp-0xF0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x14F0]
       vmovups  zmm1, zmmword ptr [rbp-0x130]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x870]
       vmovups  zmmword ptr [rbp-0x130], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x14F0]
       vmovups  zmm1, zmmword ptr [rbp-0x170]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x8B0]
       vmovups  zmmword ptr [rbp-0x170], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x14F0]
       vmovups  zmm1, zmmword ptr [rbp-0x1B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x8F0]
       vmovups  zmmword ptr [rbp-0x1B0], zmm1
       movsxd   rax, dword ptr [rbp-0x68]
       mov      rcx, qword ptr [rbp-0x1478]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x1530], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x1530]
       vmovups  zmm1, zmmword ptr [rbp-0x1F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x830]
       vmovups  zmmword ptr [rbp-0x1F0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1530]
       vmovups  zmm1, zmmword ptr [rbp-0x230]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x870]
       vmovups  zmmword ptr [rbp-0x230], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1530]
       vmovups  zmm1, zmmword ptr [rbp-0x270]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x8B0]
       vmovups  zmmword ptr [rbp-0x270], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1530]
       vmovups  zmm1, zmmword ptr [rbp-0x2B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x8F0]
       vmovups  zmmword ptr [rbp-0x2B0], zmm1
       mov      eax, dword ptr [rbp-0x68]
       add      eax, eax
       cdqe     
       mov      rcx, qword ptr [rbp-0x1478]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x1570], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x1570]
       vmovups  zmm1, zmmword ptr [rbp-0x2F0]
 
G_M000_IG11:                ;; offset=0x0A7C
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x830]
       vmovups  zmmword ptr [rbp-0x2F0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1570]
       vmovups  zmm1, zmmword ptr [rbp-0x330]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x870]
       vmovups  zmmword ptr [rbp-0x330], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1570]
       vmovups  zmm1, zmmword ptr [rbp-0x370]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x8B0]
       vmovups  zmmword ptr [rbp-0x370], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1570]
       vmovups  zmm1, zmmword ptr [rbp-0x3B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x8F0]
       vmovups  zmmword ptr [rbp-0x3B0], zmm1
       mov      eax, dword ptr [rbp-0x68]
       lea      eax, [rax+2*rax]
       cdqe     
       mov      rcx, qword ptr [rbp-0x1478]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x15B0], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x15B0]
       vmovups  zmm1, zmmword ptr [rbp-0x3F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x830]
       vmovups  zmmword ptr [rbp-0x3F0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x15B0]
       vmovups  zmm1, zmmword ptr [rbp-0x430]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x870]
       vmovups  zmmword ptr [rbp-0x430], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x15B0]
       vmovups  zmm1, zmmword ptr [rbp-0x470]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x8B0]
       vmovups  zmmword ptr [rbp-0x470], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x15B0]
       vmovups  zmm1, zmmword ptr [rbp-0x4B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x8F0]
       vmovups  zmmword ptr [rbp-0x4B0], zmm1
       mov      eax, dword ptr [rbp-0x68]
       shl      eax, 2
       cdqe     
       mov      rcx, qword ptr [rbp-0x1478]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x15F0], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x15F0]
       vmovups  zmm1, zmmword ptr [rbp-0x4F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x830]
       vmovups  zmmword ptr [rbp-0x4F0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x15F0]
       vmovups  zmm1, zmmword ptr [rbp-0x530]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x870]
       vmovups  zmmword ptr [rbp-0x530], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x15F0]
       vmovups  zmm1, zmmword ptr [rbp-0x570]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x8B0]
       vmovups  zmmword ptr [rbp-0x570], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x15F0]
       vmovups  zmm1, zmmword ptr [rbp-0x5B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x8F0]
       vmovups  zmmword ptr [rbp-0x5B0], zmm1
       mov      eax, dword ptr [rbp-0x68]
 
G_M000_IG12:                ;; offset=0x0C8B
       lea      eax, [rax+4*rax]
       cdqe     
       mov      rcx, qword ptr [rbp-0x1478]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x1630], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x1630]
       vmovups  zmm1, zmmword ptr [rbp-0x5F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x830]
       vmovups  zmmword ptr [rbp-0x5F0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1630]
       vmovups  zmm1, zmmword ptr [rbp-0x630]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x870]
       vmovups  zmmword ptr [rbp-0x630], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1630]
       vmovups  zmm1, zmmword ptr [rbp-0x670]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x8B0]
       vmovups  zmmword ptr [rbp-0x670], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1630]
       vmovups  zmm1, zmmword ptr [rbp-0x6B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x8F0]
       vmovups  zmmword ptr [rbp-0x6B0], zmm1
       mov      rax, qword ptr [rbp-0x6B8]
       add      rax, 64
       mov      qword ptr [rbp-0x6B8], rax
       mov      rax, qword ptr [rbp-0x6C0]
       add      rax, 64
       mov      qword ptr [rbp-0x6C0], rax
       mov      rax, qword ptr [rbp-0x6C8]
       add      rax, 64
       mov      qword ptr [rbp-0x6C8], rax
       mov      rax, qword ptr [rbp-0x6D0]
       add      rax, 64
       mov      qword ptr [rbp-0x6D0], rax
       mov      rax, qword ptr [rbp-0x6B8]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0x930], zmm0
       mov      rax, qword ptr [rbp-0x6C0]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0x970], zmm0
       mov      rax, qword ptr [rbp-0x6C8]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0x9B0], zmm0
       mov      rax, qword ptr [rbp-0x6D0]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0x9F0], zmm0
       mov      rax, qword ptr [rbp-0x1278]
       add      rax, 128
       mov      qword ptr [rbp-0x1638], rax
       mov      rax, qword ptr [rbp-0x1638]
       vbroadcastss zmm0, dword ptr [rax]
       vmovups  zmmword ptr [rbp-0x16B0], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x16B0]
       vmovups  zmm1, zmmword ptr [rbp-0xF0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x930]
       vmovups  zmmword ptr [rbp-0xF0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x16B0]
       vmovups  zmm1, zmmword ptr [rbp-0x130]
 
G_M000_IG13:                ;; offset=0x0E53
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x970]
       vmovups  zmmword ptr [rbp-0x130], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x16B0]
       vmovups  zmm1, zmmword ptr [rbp-0x170]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x9B0]
       vmovups  zmmword ptr [rbp-0x170], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x16B0]
       vmovups  zmm1, zmmword ptr [rbp-0x1B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x9F0]
       vmovups  zmmword ptr [rbp-0x1B0], zmm1
       movsxd   rax, dword ptr [rbp-0x68]
       mov      rcx, qword ptr [rbp-0x1638]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x16F0], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x16F0]
       vmovups  zmm1, zmmword ptr [rbp-0x1F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x930]
       vmovups  zmmword ptr [rbp-0x1F0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x16F0]
       vmovups  zmm1, zmmword ptr [rbp-0x230]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x970]
       vmovups  zmmword ptr [rbp-0x230], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x16F0]
       vmovups  zmm1, zmmword ptr [rbp-0x270]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x9B0]
       vmovups  zmmword ptr [rbp-0x270], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x16F0]
       vmovups  zmm1, zmmword ptr [rbp-0x2B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x9F0]
       vmovups  zmmword ptr [rbp-0x2B0], zmm1
       mov      eax, dword ptr [rbp-0x68]
       add      eax, eax
       cdqe     
       mov      rcx, qword ptr [rbp-0x1638]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x1730], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x1730]
       vmovups  zmm1, zmmword ptr [rbp-0x2F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x930]
       vmovups  zmmword ptr [rbp-0x2F0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1730]
       vmovups  zmm1, zmmword ptr [rbp-0x330]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x970]
       vmovups  zmmword ptr [rbp-0x330], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1730]
       vmovups  zmm1, zmmword ptr [rbp-0x370]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x9B0]
       vmovups  zmmword ptr [rbp-0x370], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1730]
       vmovups  zmm1, zmmword ptr [rbp-0x3B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x9F0]
       vmovups  zmmword ptr [rbp-0x3B0], zmm1
       mov      eax, dword ptr [rbp-0x68]
       lea      eax, [rax+2*rax]
       cdqe     
       mov      rcx, qword ptr [rbp-0x1638]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x1770], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x1770]
 
G_M000_IG14:                ;; offset=0x105C
       vmovups  zmm1, zmmword ptr [rbp-0x3F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x930]
       vmovups  zmmword ptr [rbp-0x3F0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1770]
       vmovups  zmm1, zmmword ptr [rbp-0x430]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x970]
       vmovups  zmmword ptr [rbp-0x430], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1770]
       vmovups  zmm1, zmmword ptr [rbp-0x470]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x9B0]
       vmovups  zmmword ptr [rbp-0x470], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1770]
       vmovups  zmm1, zmmword ptr [rbp-0x4B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x9F0]
       vmovups  zmmword ptr [rbp-0x4B0], zmm1
       mov      eax, dword ptr [rbp-0x68]
       shl      eax, 2
       cdqe     
       mov      rcx, qword ptr [rbp-0x1638]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x1778], rax
       mov      rax, qword ptr [rbp-0x1778]
       vbroadcastss zmm0, dword ptr [rax]
       vmovups  zmmword ptr [rbp-0x17F0], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x17F0]
       vmovups  zmm1, zmmword ptr [rbp-0x4F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x930]
       vmovups  zmmword ptr [rbp-0x4F0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x17F0]
       vmovups  zmm1, zmmword ptr [rbp-0x530]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x970]
       vmovups  zmmword ptr [rbp-0x530], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x17F0]
       vmovups  zmm1, zmmword ptr [rbp-0x570]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x9B0]
       vmovups  zmmword ptr [rbp-0x570], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x17F0]
       vmovups  zmm1, zmmword ptr [rbp-0x5B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x9F0]
       vmovups  zmmword ptr [rbp-0x5B0], zmm1
       mov      eax, dword ptr [rbp-0x68]
       lea      eax, [rax+4*rax]
       cdqe     
       mov      rcx, qword ptr [rbp-0x1638]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x1830], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x1830]
       vmovups  zmm1, zmmword ptr [rbp-0x5F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x930]
       vmovups  zmmword ptr [rbp-0x5F0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1830]
       vmovups  zmm1, zmmword ptr [rbp-0x630]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x970]
       vmovups  zmmword ptr [rbp-0x630], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1830]
       vmovups  zmm1, zmmword ptr [rbp-0x670]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x9B0]
       vmovups  zmmword ptr [rbp-0x670], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1830]
 
G_M000_IG15:                ;; offset=0x1265
       vmovups  zmm1, zmmword ptr [rbp-0x6B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x9F0]
       vmovups  zmmword ptr [rbp-0x6B0], zmm1
       mov      rax, qword ptr [rbp-0x6B8]
       add      rax, 64
       mov      qword ptr [rbp-0x6B8], rax
       mov      rax, qword ptr [rbp-0x6C0]
       add      rax, 64
       mov      qword ptr [rbp-0x6C0], rax
       mov      rax, qword ptr [rbp-0x6C8]
       add      rax, 64
       mov      qword ptr [rbp-0x6C8], rax
       mov      rax, qword ptr [rbp-0x6D0]
       add      rax, 64
       mov      qword ptr [rbp-0x6D0], rax
       mov      rax, qword ptr [rbp-0x6B8]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0xA30], zmm0
       mov      rax, qword ptr [rbp-0x6C0]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0xA70], zmm0
       mov      rax, qword ptr [rbp-0x6C8]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0xAB0], zmm0
       mov      rax, qword ptr [rbp-0x6D0]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0xAF0], zmm0
       mov      rax, qword ptr [rbp-0x6E8]
       vbroadcastss zmm0, dword ptr [rax]
       vmovups  zmmword ptr [rbp-0x1870], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x1870]
       vmovups  zmm1, zmmword ptr [rbp-0xF0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xA30]
       vmovups  zmmword ptr [rbp-0xF0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1870]
       vmovups  zmm1, zmmword ptr [rbp-0x130]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xA70]
       vmovups  zmmword ptr [rbp-0x130], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1870]
       vmovups  zmm1, zmmword ptr [rbp-0x170]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xAB0]
       vmovups  zmmword ptr [rbp-0x170], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1870]
       vmovups  zmm1, zmmword ptr [rbp-0x1B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xAF0]
       vmovups  zmmword ptr [rbp-0x1B0], zmm1
       movsxd   rax, dword ptr [rbp-0x68]
       mov      rcx, qword ptr [rbp-0x6E8]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x18B0], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x18B0]
       vmovups  zmm1, zmmword ptr [rbp-0x1F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xA30]
       vmovups  zmmword ptr [rbp-0x1F0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x18B0]
       vmovups  zmm1, zmmword ptr [rbp-0x230]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xA70]
 
G_M000_IG16:                ;; offset=0x1440
       vmovups  zmmword ptr [rbp-0x230], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x18B0]
       vmovups  zmm1, zmmword ptr [rbp-0x270]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xAB0]
       vmovups  zmmword ptr [rbp-0x270], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x18B0]
       vmovups  zmm1, zmmword ptr [rbp-0x2B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xAF0]
       vmovups  zmmword ptr [rbp-0x2B0], zmm1
       mov      eax, dword ptr [rbp-0x68]
       add      eax, eax
       cdqe     
       mov      rcx, qword ptr [rbp-0x6E8]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x18F0], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x18F0]
       vmovups  zmm1, zmmword ptr [rbp-0x2F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xA30]
       vmovups  zmmword ptr [rbp-0x2F0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x18F0]
       vmovups  zmm1, zmmword ptr [rbp-0x330]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xA70]
       vmovups  zmmword ptr [rbp-0x330], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x18F0]
       vmovups  zmm1, zmmword ptr [rbp-0x370]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xAB0]
       vmovups  zmmword ptr [rbp-0x370], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x18F0]
       vmovups  zmm1, zmmword ptr [rbp-0x3B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xAF0]
       vmovups  zmmword ptr [rbp-0x3B0], zmm1
       mov      eax, dword ptr [rbp-0x68]
       lea      eax, [rax+2*rax]
       cdqe     
       mov      rcx, qword ptr [rbp-0x6E8]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x1930], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x1930]
       vmovups  zmm1, zmmword ptr [rbp-0x3F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xA30]
       vmovups  zmmword ptr [rbp-0x3F0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1930]
       vmovups  zmm1, zmmword ptr [rbp-0x430]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xA70]
       vmovups  zmmword ptr [rbp-0x430], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1930]
       vmovups  zmm1, zmmword ptr [rbp-0x470]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xAB0]
       vmovups  zmmword ptr [rbp-0x470], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1930]
       vmovups  zmm1, zmmword ptr [rbp-0x4B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xAF0]
       vmovups  zmmword ptr [rbp-0x4B0], zmm1
       mov      eax, dword ptr [rbp-0x68]
       shl      eax, 2
       cdqe     
       mov      rcx, qword ptr [rbp-0x6E8]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x1970], zmm0
 
G_M000_IG17:                ;; offset=0x1639
       vmovups  zmm0, zmmword ptr [rbp-0x1970]
       vmovups  zmm1, zmmword ptr [rbp-0x4F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xA30]
       vmovups  zmmword ptr [rbp-0x4F0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1970]
       vmovups  zmm1, zmmword ptr [rbp-0x530]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xA70]
       vmovups  zmmword ptr [rbp-0x530], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1970]
       vmovups  zmm1, zmmword ptr [rbp-0x570]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xAB0]
       vmovups  zmmword ptr [rbp-0x570], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1970]
       vmovups  zmm1, zmmword ptr [rbp-0x5B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xAF0]
       vmovups  zmmword ptr [rbp-0x5B0], zmm1
       mov      eax, dword ptr [rbp-0x68]
       lea      eax, [rax+4*rax]
       cdqe     
       mov      rcx, qword ptr [rbp-0x6E8]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x19B0], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x19B0]
       vmovups  zmm1, zmmword ptr [rbp-0x5F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xA30]
       vmovups  zmmword ptr [rbp-0x5F0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x19B0]
       vmovups  zmm1, zmmword ptr [rbp-0x630]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xA70]
       vmovups  zmmword ptr [rbp-0x630], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x19B0]
       vmovups  zmm1, zmmword ptr [rbp-0x670]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xAB0]
       vmovups  zmmword ptr [rbp-0x670], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x19B0]
       vmovups  zmm1, zmmword ptr [rbp-0x6B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xAF0]
       vmovups  zmmword ptr [rbp-0x6B0], zmm1
       mov      rax, qword ptr [rbp-0x6B8]
       add      rax, 64
       mov      qword ptr [rbp-0x6B8], rax
       mov      rax, qword ptr [rbp-0x6C0]
       add      rax, 64
       mov      qword ptr [rbp-0x6C0], rax
       mov      rax, qword ptr [rbp-0x6C8]
       add      rax, 64
       mov      qword ptr [rbp-0x6C8], rax
       mov      rax, qword ptr [rbp-0x6D0]
       add      rax, 64
       mov      qword ptr [rbp-0x6D0], rax
       mov      rax, qword ptr [rbp-0x6B8]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0xB30], zmm0
       mov      rax, qword ptr [rbp-0x6C0]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0xB70], zmm0
       mov      rax, qword ptr [rbp-0x6C8]
 
G_M000_IG18:                ;; offset=0x1816
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0xBB0], zmm0
       mov      rax, qword ptr [rbp-0x6D0]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0xBF0], zmm0
       mov      rax, qword ptr [rbp-0x6E8]
       add      rax, 64
       mov      qword ptr [rbp-0x19B8], rax
       mov      rax, qword ptr [rbp-0x19B8]
       vbroadcastss zmm0, dword ptr [rax]
       vmovups  zmmword ptr [rbp-0x1A30], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x1A30]
       vmovups  zmm1, zmmword ptr [rbp-0xF0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xB30]
       vmovups  zmmword ptr [rbp-0xF0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1A30]
       vmovups  zmm1, zmmword ptr [rbp-0x130]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xB70]
       vmovups  zmmword ptr [rbp-0x130], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1A30]
       vmovups  zmm1, zmmword ptr [rbp-0x170]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xBB0]
       vmovups  zmmword ptr [rbp-0x170], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1A30]
       vmovups  zmm1, zmmword ptr [rbp-0x1B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xBF0]
       vmovups  zmmword ptr [rbp-0x1B0], zmm1
       movsxd   rax, dword ptr [rbp-0x68]
       mov      rcx, qword ptr [rbp-0x19B8]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x1A70], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x1A70]
       vmovups  zmm1, zmmword ptr [rbp-0x1F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xB30]
       vmovups  zmmword ptr [rbp-0x1F0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1A70]
       vmovups  zmm1, zmmword ptr [rbp-0x230]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xB70]
       vmovups  zmmword ptr [rbp-0x230], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1A70]
       vmovups  zmm1, zmmword ptr [rbp-0x270]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xBB0]
       vmovups  zmmword ptr [rbp-0x270], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1A70]
       vmovups  zmm1, zmmword ptr [rbp-0x2B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xBF0]
       vmovups  zmmword ptr [rbp-0x2B0], zmm1
       mov      eax, dword ptr [rbp-0x68]
       add      eax, eax
       cdqe     
       mov      rcx, qword ptr [rbp-0x19B8]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x1AB0], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x1AB0]
       vmovups  zmm1, zmmword ptr [rbp-0x2F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xB30]
       vmovups  zmmword ptr [rbp-0x2F0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1AB0]
       vmovups  zmm1, zmmword ptr [rbp-0x330]
 
G_M000_IG19:                ;; offset=0x1A1D
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xB70]
       vmovups  zmmword ptr [rbp-0x330], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1AB0]
       vmovups  zmm1, zmmword ptr [rbp-0x370]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xBB0]
       vmovups  zmmword ptr [rbp-0x370], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1AB0]
       vmovups  zmm1, zmmword ptr [rbp-0x3B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xBF0]
       vmovups  zmmword ptr [rbp-0x3B0], zmm1
       mov      eax, dword ptr [rbp-0x68]
       lea      eax, [rax+2*rax]
       cdqe     
       mov      rcx, qword ptr [rbp-0x19B8]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x1AF0], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x1AF0]
       vmovups  zmm1, zmmword ptr [rbp-0x3F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xB30]
       vmovups  zmmword ptr [rbp-0x3F0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1AF0]
       vmovups  zmm1, zmmword ptr [rbp-0x430]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xB70]
       vmovups  zmmword ptr [rbp-0x430], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1AF0]
       vmovups  zmm1, zmmword ptr [rbp-0x470]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xBB0]
       vmovups  zmmword ptr [rbp-0x470], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1AF0]
       vmovups  zmm1, zmmword ptr [rbp-0x4B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xBF0]
       vmovups  zmmword ptr [rbp-0x4B0], zmm1
       mov      eax, dword ptr [rbp-0x68]
       shl      eax, 2
       cdqe     
       mov      rcx, qword ptr [rbp-0x19B8]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x1B30], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x1B30]
       vmovups  zmm1, zmmword ptr [rbp-0x4F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xB30]
       vmovups  zmmword ptr [rbp-0x4F0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1B30]
       vmovups  zmm1, zmmword ptr [rbp-0x530]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xB70]
       vmovups  zmmword ptr [rbp-0x530], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1B30]
       vmovups  zmm1, zmmword ptr [rbp-0x570]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xBB0]
       vmovups  zmmword ptr [rbp-0x570], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1B30]
       vmovups  zmm1, zmmword ptr [rbp-0x5B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xBF0]
       vmovups  zmmword ptr [rbp-0x5B0], zmm1
       mov      eax, dword ptr [rbp-0x68]
       lea      eax, [rax+4*rax]
       cdqe     
       mov      rcx, qword ptr [rbp-0x19B8]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
 
G_M000_IG20:                ;; offset=0x1C17
       vmovups  zmmword ptr [rbp-0x1B70], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x1B70]
       vmovups  zmm1, zmmword ptr [rbp-0x5F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xB30]
       vmovups  zmmword ptr [rbp-0x5F0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1B70]
       vmovups  zmm1, zmmword ptr [rbp-0x630]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xB70]
       vmovups  zmmword ptr [rbp-0x630], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1B70]
       vmovups  zmm1, zmmword ptr [rbp-0x670]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xBB0]
       vmovups  zmmword ptr [rbp-0x670], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1B70]
       vmovups  zmm1, zmmword ptr [rbp-0x6B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xBF0]
       vmovups  zmmword ptr [rbp-0x6B0], zmm1
       mov      rax, qword ptr [rbp-0x6B8]
       add      rax, 64
       mov      qword ptr [rbp-0x6B8], rax
       mov      rax, qword ptr [rbp-0x6C0]
       add      rax, 64
       mov      qword ptr [rbp-0x6C0], rax
       mov      rax, qword ptr [rbp-0x6C8]
       add      rax, 64
       mov      qword ptr [rbp-0x6C8], rax
       mov      rax, qword ptr [rbp-0x6D0]
       add      rax, 64
       mov      qword ptr [rbp-0x6D0], rax
       mov      rax, qword ptr [rbp-0x6B8]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0xC30], zmm0
       mov      rax, qword ptr [rbp-0x6C0]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0xC70], zmm0
       mov      rax, qword ptr [rbp-0x6C8]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0xCB0], zmm0
       mov      rax, qword ptr [rbp-0x6D0]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0xCF0], zmm0
       mov      rax, qword ptr [rbp-0x6E8]
       add      rax, 128
       mov      qword ptr [rbp-0x1B78], rax
       mov      rax, qword ptr [rbp-0x1B78]
       vbroadcastss zmm0, dword ptr [rax]
       vmovups  zmmword ptr [rbp-0x1BF0], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x1BF0]
       vmovups  zmm1, zmmword ptr [rbp-0xF0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xC30]
       vmovups  zmmword ptr [rbp-0xF0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1BF0]
       vmovups  zmm1, zmmword ptr [rbp-0x130]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xC70]
       vmovups  zmmword ptr [rbp-0x130], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1BF0]
       vmovups  zmm1, zmmword ptr [rbp-0x170]
 
G_M000_IG21:                ;; offset=0x1DF4
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xCB0]
       vmovups  zmmword ptr [rbp-0x170], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1BF0]
       vmovups  zmm1, zmmword ptr [rbp-0x1B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xCF0]
       vmovups  zmmword ptr [rbp-0x1B0], zmm1
       movsxd   rax, dword ptr [rbp-0x68]
       mov      rcx, qword ptr [rbp-0x1B78]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x1C30], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x1C30]
       vmovups  zmm1, zmmword ptr [rbp-0x1F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xC30]
       vmovups  zmmword ptr [rbp-0x1F0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1C30]
       vmovups  zmm1, zmmword ptr [rbp-0x230]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xC70]
       vmovups  zmmword ptr [rbp-0x230], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1C30]
       vmovups  zmm1, zmmword ptr [rbp-0x270]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xCB0]
       vmovups  zmmword ptr [rbp-0x270], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1C30]
       vmovups  zmm1, zmmword ptr [rbp-0x2B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xCF0]
       vmovups  zmmword ptr [rbp-0x2B0], zmm1
       mov      eax, dword ptr [rbp-0x68]
       add      eax, eax
       cdqe     
       mov      rcx, qword ptr [rbp-0x1B78]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x1C70], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x1C70]
       vmovups  zmm1, zmmword ptr [rbp-0x2F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xC30]
       vmovups  zmmword ptr [rbp-0x2F0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1C70]
       vmovups  zmm1, zmmword ptr [rbp-0x330]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xC70]
       vmovups  zmmword ptr [rbp-0x330], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1C70]
       vmovups  zmm1, zmmword ptr [rbp-0x370]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xCB0]
       vmovups  zmmword ptr [rbp-0x370], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1C70]
       vmovups  zmm1, zmmword ptr [rbp-0x3B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xCF0]
       vmovups  zmmword ptr [rbp-0x3B0], zmm1
       mov      eax, dword ptr [rbp-0x68]
       lea      eax, [rax+2*rax]
       cdqe     
       mov      rcx, qword ptr [rbp-0x1B78]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x1CB0], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x1CB0]
       vmovups  zmm1, zmmword ptr [rbp-0x3F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xC30]
       vmovups  zmmword ptr [rbp-0x3F0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1CB0]
 
G_M000_IG22:                ;; offset=0x1FFD
       vmovups  zmm1, zmmword ptr [rbp-0x430]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xC70]
       vmovups  zmmword ptr [rbp-0x430], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1CB0]
       vmovups  zmm1, zmmword ptr [rbp-0x470]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xCB0]
       vmovups  zmmword ptr [rbp-0x470], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1CB0]
       vmovups  zmm1, zmmword ptr [rbp-0x4B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xCF0]
       vmovups  zmmword ptr [rbp-0x4B0], zmm1
       mov      eax, dword ptr [rbp-0x68]
       shl      eax, 2
       cdqe     
       mov      rcx, qword ptr [rbp-0x1B78]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x1CF0], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x1CF0]
       vmovups  zmm1, zmmword ptr [rbp-0x4F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xC30]
       vmovups  zmmword ptr [rbp-0x4F0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1CF0]
       vmovups  zmm1, zmmword ptr [rbp-0x530]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xC70]
       vmovups  zmmword ptr [rbp-0x530], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1CF0]
       vmovups  zmm1, zmmword ptr [rbp-0x570]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xCB0]
       vmovups  zmmword ptr [rbp-0x570], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1CF0]
       vmovups  zmm1, zmmword ptr [rbp-0x5B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xCF0]
       vmovups  zmmword ptr [rbp-0x5B0], zmm1
       mov      eax, dword ptr [rbp-0x68]
       lea      eax, [rax+4*rax]
       cdqe     
       mov      rcx, qword ptr [rbp-0x1B78]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x1D30], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x1D30]
       vmovups  zmm1, zmmword ptr [rbp-0x5F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xC30]
       vmovups  zmmword ptr [rbp-0x5F0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1D30]
       vmovups  zmm1, zmmword ptr [rbp-0x630]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xC70]
       vmovups  zmmword ptr [rbp-0x630], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1D30]
       vmovups  zmm1, zmmword ptr [rbp-0x670]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xCB0]
       vmovups  zmmword ptr [rbp-0x670], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1D30]
       vmovups  zmm1, zmmword ptr [rbp-0x6B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xCF0]
       vmovups  zmmword ptr [rbp-0x6B0], zmm1
       mov      rax, qword ptr [rbp-0x6B8]
       add      rax, 64
       mov      qword ptr [rbp-0x6B8], rax
       mov      rax, qword ptr [rbp-0x6C0]
 
G_M000_IG23:                ;; offset=0x2204
       add      rax, 64
       mov      qword ptr [rbp-0x6C0], rax
       mov      rax, qword ptr [rbp-0x6C8]
       add      rax, 64
       mov      qword ptr [rbp-0x6C8], rax
       mov      rax, qword ptr [rbp-0x6D0]
       add      rax, 64
       mov      qword ptr [rbp-0x6D0], rax
       mov      rax, qword ptr [rbp-0x6B8]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0xD30], zmm0
       mov      rax, qword ptr [rbp-0x6C0]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0xD70], zmm0
       mov      rax, qword ptr [rbp-0x6C8]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0xDB0], zmm0
       mov      rax, qword ptr [rbp-0x6D0]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0xDF0], zmm0
       mov      rax, qword ptr [rbp-0x6F0]
       vbroadcastss zmm0, dword ptr [rax]
       vmovups  zmmword ptr [rbp-0x1D70], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x1D70]
       vmovups  zmm1, zmmword ptr [rbp-0xF0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xD30]
       vmovups  zmmword ptr [rbp-0xF0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1D70]
       vmovups  zmm1, zmmword ptr [rbp-0x130]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xD70]
       vmovups  zmmword ptr [rbp-0x130], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1D70]
       vmovups  zmm1, zmmword ptr [rbp-0x170]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xDB0]
       vmovups  zmmword ptr [rbp-0x170], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1D70]
       vmovups  zmm1, zmmword ptr [rbp-0x1B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xDF0]
       vmovups  zmmword ptr [rbp-0x1B0], zmm1
       movsxd   rax, dword ptr [rbp-0x68]
       mov      rcx, qword ptr [rbp-0x6F0]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x1DB0], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x1DB0]
       vmovups  zmm1, zmmword ptr [rbp-0x1F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xD30]
       vmovups  zmmword ptr [rbp-0x1F0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1DB0]
       vmovups  zmm1, zmmword ptr [rbp-0x230]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xD70]
       vmovups  zmmword ptr [rbp-0x230], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1DB0]
       vmovups  zmm1, zmmword ptr [rbp-0x270]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xDB0]
       vmovups  zmmword ptr [rbp-0x270], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1DB0]
       vmovups  zmm1, zmmword ptr [rbp-0x2B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xDF0]
 
G_M000_IG24:                ;; offset=0x23F8
       vmovups  zmmword ptr [rbp-0x2B0], zmm1
       mov      eax, dword ptr [rbp-0x68]
       add      eax, eax
       cdqe     
       mov      rcx, qword ptr [rbp-0x6F0]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x1DF0], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x1DF0]
       vmovups  zmm1, zmmword ptr [rbp-0x2F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xD30]
       vmovups  zmmword ptr [rbp-0x2F0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1DF0]
       vmovups  zmm1, zmmword ptr [rbp-0x330]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xD70]
       vmovups  zmmword ptr [rbp-0x330], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1DF0]
       vmovups  zmm1, zmmword ptr [rbp-0x370]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xDB0]
       vmovups  zmmword ptr [rbp-0x370], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1DF0]
       vmovups  zmm1, zmmword ptr [rbp-0x3B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xDF0]
       vmovups  zmmword ptr [rbp-0x3B0], zmm1
       mov      eax, dword ptr [rbp-0x68]
       lea      eax, [rax+2*rax]
       cdqe     
       mov      rcx, qword ptr [rbp-0x6F0]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x1E30], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x1E30]
       vmovups  zmm1, zmmword ptr [rbp-0x3F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xD30]
       vmovups  zmmword ptr [rbp-0x3F0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1E30]
       vmovups  zmm1, zmmword ptr [rbp-0x430]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xD70]
       vmovups  zmmword ptr [rbp-0x430], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1E30]
       vmovups  zmm1, zmmword ptr [rbp-0x470]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xDB0]
       vmovups  zmmword ptr [rbp-0x470], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1E30]
       vmovups  zmm1, zmmword ptr [rbp-0x4B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xDF0]
       vmovups  zmmword ptr [rbp-0x4B0], zmm1
       mov      eax, dword ptr [rbp-0x68]
       shl      eax, 2
       cdqe     
       mov      rcx, qword ptr [rbp-0x6F0]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x1E70], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x1E70]
       vmovups  zmm1, zmmword ptr [rbp-0x4F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xD30]
       vmovups  zmmword ptr [rbp-0x4F0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1E70]
       vmovups  zmm1, zmmword ptr [rbp-0x530]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xD70]
       vmovups  zmmword ptr [rbp-0x530], zmm1
 
G_M000_IG25:                ;; offset=0x25F1
       vmovups  zmm0, zmmword ptr [rbp-0x1E70]
       vmovups  zmm1, zmmword ptr [rbp-0x570]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xDB0]
       vmovups  zmmword ptr [rbp-0x570], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1E70]
       vmovups  zmm1, zmmword ptr [rbp-0x5B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xDF0]
       vmovups  zmmword ptr [rbp-0x5B0], zmm1
       mov      eax, dword ptr [rbp-0x68]
       lea      eax, [rax+4*rax]
       cdqe     
       mov      rcx, qword ptr [rbp-0x6F0]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x1EB0], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x1EB0]
       vmovups  zmm1, zmmword ptr [rbp-0x5F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xD30]
       vmovups  zmmword ptr [rbp-0x5F0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1EB0]
       vmovups  zmm1, zmmword ptr [rbp-0x630]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xD70]
       vmovups  zmmword ptr [rbp-0x630], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1EB0]
       vmovups  zmm1, zmmword ptr [rbp-0x670]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xDB0]
       vmovups  zmmword ptr [rbp-0x670], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1EB0]
       vmovups  zmm1, zmmword ptr [rbp-0x6B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xDF0]
       vmovups  zmmword ptr [rbp-0x6B0], zmm1
       mov      rax, qword ptr [rbp-0x6B8]
       add      rax, 64
       mov      qword ptr [rbp-0x6B8], rax
       mov      rax, qword ptr [rbp-0x6C0]
       add      rax, 64
       mov      qword ptr [rbp-0x6C0], rax
       mov      rax, qword ptr [rbp-0x6C8]
       add      rax, 64
       mov      qword ptr [rbp-0x6C8], rax
       mov      rax, qword ptr [rbp-0x6D0]
       add      rax, 64
       mov      qword ptr [rbp-0x6D0], rax
       mov      rax, qword ptr [rbp-0x6B8]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0xE30], zmm0
       mov      rax, qword ptr [rbp-0x6C0]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0xE70], zmm0
       mov      rax, qword ptr [rbp-0x6C8]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0xEB0], zmm0
       mov      rax, qword ptr [rbp-0x6D0]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0xEF0], zmm0
       mov      rax, qword ptr [rbp-0x6F0]
       add      rax, 64
       mov      qword ptr [rbp-0x1EB8], rax
 
G_M000_IG26:                ;; offset=0x27B7
       mov      rax, qword ptr [rbp-0x1EB8]
       vbroadcastss zmm0, dword ptr [rax]
       vmovups  zmmword ptr [rbp-0x1F30], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x1F30]
       vmovups  zmm1, zmmword ptr [rbp-0xF0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xE30]
       vmovups  zmmword ptr [rbp-0xF0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1F30]
       vmovups  zmm1, zmmword ptr [rbp-0x130]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xE70]
       vmovups  zmmword ptr [rbp-0x130], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1F30]
       vmovups  zmm1, zmmword ptr [rbp-0x170]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xEB0]
       vmovups  zmmword ptr [rbp-0x170], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1F30]
       vmovups  zmm1, zmmword ptr [rbp-0x1B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xEF0]
       vmovups  zmmword ptr [rbp-0x1B0], zmm1
       movsxd   rax, dword ptr [rbp-0x68]
       mov      rcx, qword ptr [rbp-0x1EB8]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x1F70], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x1F70]
       vmovups  zmm1, zmmword ptr [rbp-0x1F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xE30]
       vmovups  zmmword ptr [rbp-0x1F0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1F70]
       vmovups  zmm1, zmmword ptr [rbp-0x230]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xE70]
       vmovups  zmmword ptr [rbp-0x230], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1F70]
       vmovups  zmm1, zmmword ptr [rbp-0x270]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xEB0]
       vmovups  zmmword ptr [rbp-0x270], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1F70]
       vmovups  zmm1, zmmword ptr [rbp-0x2B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xEF0]
       vmovups  zmmword ptr [rbp-0x2B0], zmm1
       mov      eax, dword ptr [rbp-0x68]
       add      eax, eax
       cdqe     
       mov      rcx, qword ptr [rbp-0x1EB8]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x1FB0], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x1FB0]
       vmovups  zmm1, zmmword ptr [rbp-0x2F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xE30]
       vmovups  zmmword ptr [rbp-0x2F0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1FB0]
       vmovups  zmm1, zmmword ptr [rbp-0x330]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xE70]
       vmovups  zmmword ptr [rbp-0x330], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1FB0]
       vmovups  zmm1, zmmword ptr [rbp-0x370]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xEB0]
       vmovups  zmmword ptr [rbp-0x370], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1FB0]
       vmovups  zmm1, zmmword ptr [rbp-0x3B0]
 
G_M000_IG27:                ;; offset=0x29D5
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xEF0]
       vmovups  zmmword ptr [rbp-0x3B0], zmm1
       mov      eax, dword ptr [rbp-0x68]
       lea      eax, [rax+2*rax]
       cdqe     
       mov      rcx, qword ptr [rbp-0x1EB8]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x1FF0], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x1FF0]
       vmovups  zmm1, zmmword ptr [rbp-0x3F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xE30]
       vmovups  zmmword ptr [rbp-0x3F0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1FF0]
       vmovups  zmm1, zmmword ptr [rbp-0x430]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xE70]
       vmovups  zmmword ptr [rbp-0x430], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1FF0]
       vmovups  zmm1, zmmword ptr [rbp-0x470]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xEB0]
       vmovups  zmmword ptr [rbp-0x470], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1FF0]
       vmovups  zmm1, zmmword ptr [rbp-0x4B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xEF0]
       vmovups  zmmword ptr [rbp-0x4B0], zmm1
       mov      eax, dword ptr [rbp-0x68]
       shl      eax, 2
       cdqe     
       mov      rcx, qword ptr [rbp-0x1EB8]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x2030], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x2030]
       vmovups  zmm1, zmmword ptr [rbp-0x4F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xE30]
       vmovups  zmmword ptr [rbp-0x4F0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x2030]
       vmovups  zmm1, zmmword ptr [rbp-0x530]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xE70]
       vmovups  zmmword ptr [rbp-0x530], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x2030]
       vmovups  zmm1, zmmword ptr [rbp-0x570]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xEB0]
       vmovups  zmmword ptr [rbp-0x570], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x2030]
       vmovups  zmm1, zmmword ptr [rbp-0x5B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xEF0]
       vmovups  zmmword ptr [rbp-0x5B0], zmm1
       mov      eax, dword ptr [rbp-0x68]
       lea      eax, [rax+4*rax]
       cdqe     
       mov      rcx, qword ptr [rbp-0x1EB8]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x2070], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x2070]
       vmovups  zmm1, zmmword ptr [rbp-0x5F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xE30]
       vmovups  zmmword ptr [rbp-0x5F0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x2070]
       vmovups  zmm1, zmmword ptr [rbp-0x630]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xE70]
 
G_M000_IG28:                ;; offset=0x2BCF
       vmovups  zmmword ptr [rbp-0x630], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x2070]
       vmovups  zmm1, zmmword ptr [rbp-0x670]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xEB0]
       vmovups  zmmword ptr [rbp-0x670], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x2070]
       vmovups  zmm1, zmmword ptr [rbp-0x6B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xEF0]
       vmovups  zmmword ptr [rbp-0x6B0], zmm1
       mov      rax, qword ptr [rbp-0x6B8]
       add      rax, 64
       mov      qword ptr [rbp-0x6B8], rax
       mov      rax, qword ptr [rbp-0x6C0]
       add      rax, 64
       mov      qword ptr [rbp-0x6C0], rax
       mov      rax, qword ptr [rbp-0x6C8]
       add      rax, 64
       mov      qword ptr [rbp-0x6C8], rax
       mov      rax, qword ptr [rbp-0x6D0]
       add      rax, 64
       mov      qword ptr [rbp-0x6D0], rax
       mov      rax, qword ptr [rbp-0x6B8]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0xF30], zmm0
       mov      rax, qword ptr [rbp-0x6C0]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0xF70], zmm0
       mov      rax, qword ptr [rbp-0x6C8]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0xFB0], zmm0
       mov      rax, qword ptr [rbp-0x6D0]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0xFF0], zmm0
       mov      rax, qword ptr [rbp-0x6F0]
       add      rax, 128
       mov      qword ptr [rbp-0x2078], rax
       mov      rax, qword ptr [rbp-0x2078]
       vbroadcastss zmm0, dword ptr [rax]
       vmovups  zmmword ptr [rbp-0x20F0], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x20F0]
       vmovups  zmm1, zmmword ptr [rbp-0xF0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xF30]
       vmovups  zmmword ptr [rbp-0xF0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x20F0]
       vmovups  zmm1, zmmword ptr [rbp-0x130]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xF70]
       vmovups  zmmword ptr [rbp-0x130], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x20F0]
       vmovups  zmm1, zmmword ptr [rbp-0x170]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xFB0]
       vmovups  zmmword ptr [rbp-0x170], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x20F0]
       vmovups  zmm1, zmmword ptr [rbp-0x1B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xFF0]
       vmovups  zmmword ptr [rbp-0x1B0], zmm1
       movsxd   rax, dword ptr [rbp-0x68]
       mov      rcx, qword ptr [rbp-0x2078]
 
G_M000_IG29:                ;; offset=0x2DA3
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x2130], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x2130]
       vmovups  zmm1, zmmword ptr [rbp-0x1F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xF30]
       vmovups  zmmword ptr [rbp-0x1F0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x2130]
       vmovups  zmm1, zmmword ptr [rbp-0x230]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xF70]
       vmovups  zmmword ptr [rbp-0x230], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x2130]
       vmovups  zmm1, zmmword ptr [rbp-0x270]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xFB0]
       vmovups  zmmword ptr [rbp-0x270], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x2130]
       vmovups  zmm1, zmmword ptr [rbp-0x2B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xFF0]
       vmovups  zmmword ptr [rbp-0x2B0], zmm1
       mov      eax, dword ptr [rbp-0x68]
       add      eax, eax
       cdqe     
       mov      rcx, qword ptr [rbp-0x2078]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x2170], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x2170]
       vmovups  zmm1, zmmword ptr [rbp-0x2F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xF30]
       vmovups  zmmword ptr [rbp-0x2F0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x2170]
       vmovups  zmm1, zmmword ptr [rbp-0x330]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xF70]
       vmovups  zmmword ptr [rbp-0x330], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x2170]
       vmovups  zmm1, zmmword ptr [rbp-0x370]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xFB0]
       vmovups  zmmword ptr [rbp-0x370], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x2170]
       vmovups  zmm1, zmmword ptr [rbp-0x3B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xFF0]
       vmovups  zmmword ptr [rbp-0x3B0], zmm1
       mov      eax, dword ptr [rbp-0x68]
       lea      eax, [rax+2*rax]
       cdqe     
       mov      rcx, qword ptr [rbp-0x2078]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x21B0], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x21B0]
       vmovups  zmm1, zmmword ptr [rbp-0x3F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xF30]
       vmovups  zmmword ptr [rbp-0x3F0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x21B0]
       vmovups  zmm1, zmmword ptr [rbp-0x430]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xF70]
       vmovups  zmmword ptr [rbp-0x430], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x21B0]
       vmovups  zmm1, zmmword ptr [rbp-0x470]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xFB0]
       vmovups  zmmword ptr [rbp-0x470], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x21B0]
 
G_M000_IG30:                ;; offset=0x2FB5
       vmovups  zmm1, zmmword ptr [rbp-0x4B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xFF0]
       vmovups  zmmword ptr [rbp-0x4B0], zmm1
       mov      eax, dword ptr [rbp-0x68]
       shl      eax, 2
       cdqe     
       mov      rcx, qword ptr [rbp-0x2078]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x21F0], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x21F0]
       vmovups  zmm1, zmmword ptr [rbp-0x4F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xF30]
       vmovups  zmmword ptr [rbp-0x4F0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x21F0]
       vmovups  zmm1, zmmword ptr [rbp-0x530]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xF70]
       vmovups  zmmword ptr [rbp-0x530], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x21F0]
       vmovups  zmm1, zmmword ptr [rbp-0x570]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xFB0]
       vmovups  zmmword ptr [rbp-0x570], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x21F0]
       vmovups  zmm1, zmmword ptr [rbp-0x5B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xFF0]
       vmovups  zmmword ptr [rbp-0x5B0], zmm1
       mov      eax, dword ptr [rbp-0x68]
       lea      eax, [rax+4*rax]
       cdqe     
       mov      rcx, qword ptr [rbp-0x2078]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x2230], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x2230]
       vmovups  zmm1, zmmword ptr [rbp-0x5F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xF30]
       vmovups  zmmword ptr [rbp-0x5F0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x2230]
       vmovups  zmm1, zmmword ptr [rbp-0x630]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xF70]
       vmovups  zmmword ptr [rbp-0x630], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x2230]
       vmovups  zmm1, zmmword ptr [rbp-0x670]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xFB0]
       vmovups  zmmword ptr [rbp-0x670], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x2230]
       vmovups  zmm1, zmmword ptr [rbp-0x6B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0xFF0]
       vmovups  zmmword ptr [rbp-0x6B0], zmm1
       mov      rax, qword ptr [rbp-0x6B8]
       add      rax, 64
       mov      qword ptr [rbp-0x6B8], rax
       mov      rax, qword ptr [rbp-0x6C0]
       add      rax, 64
       mov      qword ptr [rbp-0x6C0], rax
       mov      rax, qword ptr [rbp-0x6C8]
       add      rax, 64
       mov      qword ptr [rbp-0x6C8], rax
       mov      rax, qword ptr [rbp-0x6D0]
       add      rax, 64
 
G_M000_IG31:                ;; offset=0x3194
       mov      qword ptr [rbp-0x6D0], rax
       mov      eax, dword ptr [rbp-0x6DC]
       inc      eax
       mov      dword ptr [rbp-0x6DC], eax
 
G_M000_IG32:                ;; offset=0x31A9
       mov      eax, dword ptr [rbp-0x2238]
       dec      eax
       mov      dword ptr [rbp-0x2238], eax
       cmp      dword ptr [rbp-0x2238], 0
       jg       SHORT G_M000_IG34
 
G_M000_IG33:                ;; offset=0x31C0
       lea      rdi, [rbp-0x2238]
       mov      esi, 0x1101
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG34:                ;; offset=0x31D1
       mov      eax, dword ptr [rbp-0x6DC]
       cmp      eax, dword ptr [rbp-0x44]
       jl       G_M000_IG07
       mov      rdi, 0x72A8A2D4EF84
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x6C]
       sar      eax, 31
       and      eax, 15
       add      eax, dword ptr [rbp-0x6C]
       sar      eax, 4
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x70]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       add      eax, dword ptr [rbp-0x74]
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  zmm0, zmmword ptr [rbp-0xF0]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
       mov      eax, dword ptr [rbp-0x6C]
       sar      eax, 31
       and      eax, 15
       add      eax, dword ptr [rbp-0x6C]
       sar      eax, 4
       inc      eax
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x70]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       add      eax, dword ptr [rbp-0x74]
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  zmm0, zmmword ptr [rbp-0x130]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
       mov      eax, dword ptr [rbp-0x6C]
       sar      eax, 31
       and      eax, 15
       add      eax, dword ptr [rbp-0x6C]
       sar      eax, 4
       add      eax, 2
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x70]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       add      eax, dword ptr [rbp-0x74]
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  zmm0, zmmword ptr [rbp-0x170]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
       mov      eax, dword ptr [rbp-0x6C]
       sar      eax, 31
       and      eax, 15
       add      eax, dword ptr [rbp-0x6C]
       sar      eax, 4
       add      eax, 3
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x70]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
 
G_M000_IG35:                ;; offset=0x32BE
       add      eax, dword ptr [rbp-0x74]
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  zmm0, zmmword ptr [rbp-0x1B0]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
       mov      eax, dword ptr [rbp-0x6C]
       sar      eax, 31
       and      eax, 15
       add      eax, dword ptr [rbp-0x6C]
       sar      eax, 4
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x70]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       mov      ecx, dword ptr [rbp-0x74]
       lea      eax, [rax+rcx+0x01]
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  zmm0, zmmword ptr [rbp-0x1F0]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
       mov      eax, dword ptr [rbp-0x6C]
       sar      eax, 31
       and      eax, 15
       add      eax, dword ptr [rbp-0x6C]
       sar      eax, 4
       inc      eax
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x70]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       mov      ecx, dword ptr [rbp-0x74]
       lea      eax, [rax+rcx+0x01]
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  zmm0, zmmword ptr [rbp-0x230]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
       mov      eax, dword ptr [rbp-0x6C]
       sar      eax, 31
       and      eax, 15
       add      eax, dword ptr [rbp-0x6C]
       sar      eax, 4
       add      eax, 2
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x70]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       mov      ecx, dword ptr [rbp-0x74]
       lea      eax, [rax+rcx+0x01]
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  zmm0, zmmword ptr [rbp-0x270]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
       mov      eax, dword ptr [rbp-0x6C]
       sar      eax, 31
       and      eax, 15
       add      eax, dword ptr [rbp-0x6C]
       sar      eax, 4
       add      eax, 3
       imul     eax, dword ptr [rbp-0x58]
 
G_M000_IG36:                ;; offset=0x33AD
       mov      ecx, dword ptr [rbp-0x70]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       mov      ecx, dword ptr [rbp-0x74]
       lea      eax, [rax+rcx+0x01]
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  zmm0, zmmword ptr [rbp-0x2B0]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
       mov      eax, dword ptr [rbp-0x6C]
       sar      eax, 31
       and      eax, 15
       add      eax, dword ptr [rbp-0x6C]
       sar      eax, 4
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x70]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       mov      ecx, dword ptr [rbp-0x74]
       lea      eax, [rax+rcx+0x02]
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  zmm0, zmmword ptr [rbp-0x2F0]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
       mov      eax, dword ptr [rbp-0x6C]
       sar      eax, 31
       and      eax, 15
       add      eax, dword ptr [rbp-0x6C]
       sar      eax, 4
       inc      eax
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x70]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       mov      ecx, dword ptr [rbp-0x74]
       lea      eax, [rax+rcx+0x02]
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  zmm0, zmmword ptr [rbp-0x330]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
       mov      eax, dword ptr [rbp-0x6C]
       sar      eax, 31
       and      eax, 15
       add      eax, dword ptr [rbp-0x6C]
       sar      eax, 4
       add      eax, 2
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x70]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       mov      ecx, dword ptr [rbp-0x74]
       lea      eax, [rax+rcx+0x02]
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  zmm0, zmmword ptr [rbp-0x370]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
       mov      eax, dword ptr [rbp-0x6C]
       sar      eax, 31
       and      eax, 15
 
G_M000_IG37:                ;; offset=0x349C
       add      eax, dword ptr [rbp-0x6C]
       sar      eax, 4
       add      eax, 3
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x70]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       mov      ecx, dword ptr [rbp-0x74]
       lea      eax, [rax+rcx+0x02]
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  zmm0, zmmword ptr [rbp-0x3B0]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
       mov      eax, dword ptr [rbp-0x6C]
       sar      eax, 31
       and      eax, 15
       add      eax, dword ptr [rbp-0x6C]
       sar      eax, 4
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x70]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       mov      ecx, dword ptr [rbp-0x74]
       lea      eax, [rax+rcx+0x03]
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  zmm0, zmmword ptr [rbp-0x3F0]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
       mov      eax, dword ptr [rbp-0x6C]
       sar      eax, 31
       and      eax, 15
       add      eax, dword ptr [rbp-0x6C]
       sar      eax, 4
       inc      eax
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x70]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       mov      ecx, dword ptr [rbp-0x74]
       lea      eax, [rax+rcx+0x03]
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  zmm0, zmmword ptr [rbp-0x430]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
       mov      eax, dword ptr [rbp-0x6C]
       sar      eax, 31
       and      eax, 15
       add      eax, dword ptr [rbp-0x6C]
       sar      eax, 4
       add      eax, 2
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x70]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       mov      ecx, dword ptr [rbp-0x74]
       lea      eax, [rax+rcx+0x03]
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  zmm0, zmmword ptr [rbp-0x470]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
 
G_M000_IG38:                ;; offset=0x358F
       mov      eax, dword ptr [rbp-0x6C]
       sar      eax, 31
       and      eax, 15
       add      eax, dword ptr [rbp-0x6C]
       sar      eax, 4
       add      eax, 3
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x70]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       mov      ecx, dword ptr [rbp-0x74]
       lea      eax, [rax+rcx+0x03]
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  zmm0, zmmword ptr [rbp-0x4B0]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
       mov      eax, dword ptr [rbp-0x6C]
       sar      eax, 31
       and      eax, 15
       add      eax, dword ptr [rbp-0x6C]
       sar      eax, 4
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x70]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       mov      ecx, dword ptr [rbp-0x74]
       lea      eax, [rax+rcx+0x04]
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  zmm0, zmmword ptr [rbp-0x4F0]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
       mov      eax, dword ptr [rbp-0x6C]
       sar      eax, 31
       and      eax, 15
       add      eax, dword ptr [rbp-0x6C]
       sar      eax, 4
       inc      eax
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x70]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       mov      ecx, dword ptr [rbp-0x74]
       lea      eax, [rax+rcx+0x04]
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  zmm0, zmmword ptr [rbp-0x530]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
       mov      eax, dword ptr [rbp-0x6C]
       sar      eax, 31
       and      eax, 15
       add      eax, dword ptr [rbp-0x6C]
       sar      eax, 4
       add      eax, 2
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x70]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       mov      ecx, dword ptr [rbp-0x74]
       lea      eax, [rax+rcx+0x04]
       shl      eax, 4
       cdqe     
 
G_M000_IG39:                ;; offset=0x3676
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  zmm0, zmmword ptr [rbp-0x570]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
       mov      eax, dword ptr [rbp-0x6C]
       sar      eax, 31
       and      eax, 15
       add      eax, dword ptr [rbp-0x6C]
       sar      eax, 4
       add      eax, 3
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x70]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       mov      ecx, dword ptr [rbp-0x74]
       lea      eax, [rax+rcx+0x04]
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  zmm0, zmmword ptr [rbp-0x5B0]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
       mov      eax, dword ptr [rbp-0x6C]
       sar      eax, 31
       and      eax, 15
       add      eax, dword ptr [rbp-0x6C]
       sar      eax, 4
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x70]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       mov      ecx, dword ptr [rbp-0x74]
       lea      eax, [rax+rcx+0x05]
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  zmm0, zmmword ptr [rbp-0x5F0]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
       mov      eax, dword ptr [rbp-0x6C]
       sar      eax, 31
       and      eax, 15
       add      eax, dword ptr [rbp-0x6C]
       sar      eax, 4
       inc      eax
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x70]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       mov      ecx, dword ptr [rbp-0x74]
       lea      eax, [rax+rcx+0x05]
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  zmm0, zmmword ptr [rbp-0x630]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
       mov      eax, dword ptr [rbp-0x6C]
       sar      eax, 31
       and      eax, 15
       add      eax, dword ptr [rbp-0x6C]
       sar      eax, 4
       add      eax, 2
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x70]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
 
G_M000_IG40:                ;; offset=0x3766
       mov      ecx, dword ptr [rbp-0x74]
       lea      eax, [rax+rcx+0x05]
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  zmm0, zmmword ptr [rbp-0x670]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
       mov      eax, dword ptr [rbp-0x6C]
       sar      eax, 31
       and      eax, 15
       add      eax, dword ptr [rbp-0x6C]
       sar      eax, 4
       add      eax, 3
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x70]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       mov      ecx, dword ptr [rbp-0x74]
       lea      eax, [rax+rcx+0x05]
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  zmm0, zmmword ptr [rbp-0x6B0]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
       mov      eax, dword ptr [rbp-0x74]
       add      eax, 6
       mov      dword ptr [rbp-0x74], eax
 
G_M000_IG41:                ;; offset=0x37D0
       mov      eax, dword ptr [rbp-0x74]
       add      eax, 6
       cmp      eax, dword ptr [rbp+0x28]
       jg       G_M000_IG61
       mov      eax, dword ptr [rbp-0x2238]
       dec      eax
       mov      dword ptr [rbp-0x2238], eax
       cmp      dword ptr [rbp-0x2238], 0
       jg       SHORT G_M000_IG43
 
G_M000_IG42:                ;; offset=0x37F6
       lea      rdi, [rbp-0x2238]
       mov      esi, 0x144E
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG43:                ;; offset=0x3807
       mov      eax, dword ptr [rbp-0x70]
       imul     eax, dword ptr [rbp+0x28]
       mov      ecx, dword ptr [rbp-0x74]
       lea      eax, [rax+rcx+0x06]
       cmp      eax, dword ptr [rbp-0x5C]
       jle      G_M000_IG05
       mov      rdi, 0x72A8A2D4EF88
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG61
 
G_M000_IG44:                ;; offset=0x3832
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x1030], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x1070], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x10B0], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x10F0], zmm0
       mov      eax, dword ptr [rbp-0x6C]
       imul     eax, dword ptr [rbp-0x44]
       lea      eax, [rax+8*rax]
       cdqe     
       mov      rcx, qword ptr [rbp-0x38]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x10F8], rax
       mov      eax, dword ptr [rbp-0x44]
       shl      eax, 4
       lea      eax, [rax+8*rax]
       cdqe     
       mov      rcx, qword ptr [rbp-0x10F8]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x1100], rax
       mov      eax, dword ptr [rbp-0x44]
       shl      eax, 4
       lea      eax, [rax+8*rax]
       cdqe     
       mov      rcx, qword ptr [rbp-0x1100]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x1108], rax
       mov      eax, dword ptr [rbp-0x44]
       shl      eax, 4
       lea      eax, [rax+8*rax]
       cdqe     
       mov      rcx, qword ptr [rbp-0x1108]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x1110], rax
       mov      eax, dword ptr [rbp-0x70]
       imul     eax, dword ptr [rbp+0x28]
       add      eax, dword ptr [rbp-0x74]
       cmp      eax, dword ptr [rbp-0x5C]
       setl     al
       movzx    rax, al
       mov      dword ptr [rbp-0x1114], eax
       xor      eax, eax
       mov      dword ptr [rbp-0x1118], eax
       jmp      G_M000_IG57
 
G_M000_IG45:                ;; offset=0x3902
       xor      eax, eax
       mov      dword ptr [rbp-0x111C], eax
       jmp      G_M000_IG54
 
G_M000_IG46:                ;; offset=0x390F
       xor      eax, eax
       mov      dword ptr [rbp-0x1120], eax
       jmp      G_M000_IG51
 
G_M000_IG47:                ;; offset=0x391C
       mov      eax, dword ptr [rbp-0x1118]
       sar      eax, 31
       and      eax, 15
       add      eax, dword ptr [rbp-0x1118]
       sar      eax, 4
       imul     eax, dword ptr [rbp-0x50]
       mov      ecx, dword ptr [rbp-0x70]
       imul     ecx, dword ptr [rbp+0x18]
       add      eax, ecx
       add      eax, dword ptr [rbp-0x111C]
       imul     eax, dword ptr [rbp-0x54]
       mov      ecx, dword ptr [rbp-0x74]
       imul     ecx, dword ptr [rbp+0x18]
       add      eax, ecx
       add      eax, dword ptr [rbp-0x1120]
       shl      eax, 4
       mov      ecx, dword ptr [rbp-0x1118]
       mov      edx, dword ptr [rbp-0x1118]
       sar      edx, 31
       and      edx, 15
       add      edx, dword ptr [rbp-0x1118]
       and      edx, -16
       sub      ecx, edx
       add      eax, ecx
       cdqe     
       mov      rcx, qword ptr [rbp-0x30]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x1170], zmm0
       mov      rax, qword ptr [rbp-0x10F8]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0x11B0], zmm0
       mov      rax, qword ptr [rbp-0x1100]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0x11F0], zmm0
       mov      rax, qword ptr [rbp-0x1108]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0x1230], zmm0
       mov      rax, qword ptr [rbp-0x1110]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0x1270], zmm0
       cmp      dword ptr [rbp-0x1114], 0
       je       G_M000_IG49
       vmovups  zmm0, zmmword ptr [rbp-0x1170]
       vmovups  zmm1, zmmword ptr [rbp-0x1030]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x11B0]
       vmovups  zmmword ptr [rbp-0x1030], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1170]
       vmovups  zmm1, zmmword ptr [rbp-0x1070]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x11F0]
       vmovups  zmmword ptr [rbp-0x1070], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1170]
       vmovups  zmm1, zmmword ptr [rbp-0x10B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x1230]
       vmovups  zmmword ptr [rbp-0x10B0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x1170]
       vmovups  zmm1, zmmword ptr [rbp-0x10F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x1270]
       vmovups  zmmword ptr [rbp-0x10F0], zmm1
 
G_M000_IG48:                ;; offset=0x3A99
       jmp      G_M000_IG50
 
G_M000_IG49:                ;; offset=0x3A9E
       mov      rdi, 0x72A8A2D4EF8C
       call     CORINFO_HELP_COUNTPROFILE32
       vmovups  zmm0, zmmword ptr [rbp-0x1170]
       vmulps   zmm0, zmm0, zmmword ptr [rbp-0x11B0]
       vaddps   zmm0, zmm0, zmmword ptr [rbp-0x1030]
       vmovups  zmmword ptr [rbp-0x1030], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x1170]
       vmulps   zmm0, zmm0, zmmword ptr [rbp-0x11F0]
       vaddps   zmm0, zmm0, zmmword ptr [rbp-0x1070]
       vmovups  zmmword ptr [rbp-0x1070], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x1170]
       vmulps   zmm0, zmm0, zmmword ptr [rbp-0x1230]
       vaddps   zmm0, zmm0, zmmword ptr [rbp-0x10B0]
       vmovups  zmmword ptr [rbp-0x10B0], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x1170]
       vmulps   zmm0, zmm0, zmmword ptr [rbp-0x1270]
       vaddps   zmm0, zmm0, zmmword ptr [rbp-0x10F0]
       vmovups  zmmword ptr [rbp-0x10F0], zmm0
 
G_M000_IG50:                ;; offset=0x3B4D
       mov      rdi, 0x72A8A2D4EF90
       call     CORINFO_HELP_COUNTPROFILE32
       mov      rax, qword ptr [rbp-0x10F8]
       add      rax, 64
       mov      qword ptr [rbp-0x10F8], rax
       mov      rax, qword ptr [rbp-0x1100]
       add      rax, 64
       mov      qword ptr [rbp-0x1100], rax
       mov      rax, qword ptr [rbp-0x1108]
       add      rax, 64
       mov      qword ptr [rbp-0x1108], rax
       mov      rax, qword ptr [rbp-0x1110]
       add      rax, 64
       mov      qword ptr [rbp-0x1110], rax
       mov      eax, dword ptr [rbp-0x1120]
       inc      eax
       mov      dword ptr [rbp-0x1120], eax
 
G_M000_IG51:                ;; offset=0x3BB2
       mov      eax, dword ptr [rbp-0x2238]
       dec      eax
       mov      dword ptr [rbp-0x2238], eax
       cmp      dword ptr [rbp-0x2238], 0
       jg       SHORT G_M000_IG53
 
G_M000_IG52:                ;; offset=0x3BC9
       lea      rdi, [rbp-0x2238]
       mov      esi, 0x15E4
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG53:                ;; offset=0x3BDA
       cmp      dword ptr [rbp-0x1120], 3
       jl       G_M000_IG47
       mov      rdi, 0x72A8A2D4EF94
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x111C]
       inc      eax
       mov      dword ptr [rbp-0x111C], eax
 
G_M000_IG54:                ;; offset=0x3C04
       mov      eax, dword ptr [rbp-0x2238]
       dec      eax
       mov      dword ptr [rbp-0x2238], eax
       cmp      dword ptr [rbp-0x2238], 0
       jg       SHORT G_M000_IG56
 
G_M000_IG55:                ;; offset=0x3C1B
       lea      rdi, [rbp-0x2238]
       mov      esi, 0x15F2
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG56:                ;; offset=0x3C2C
       cmp      dword ptr [rbp-0x111C], 3
       jl       G_M000_IG46
       mov      rdi, 0x72A8A2D4EF98
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x1118]
       inc      eax
       mov      dword ptr [rbp-0x1118], eax
 
G_M000_IG57:                ;; offset=0x3C56
       mov      eax, dword ptr [rbp-0x2238]
       dec      eax
       mov      dword ptr [rbp-0x2238], eax
       cmp      dword ptr [rbp-0x2238], 0
       jg       SHORT G_M000_IG59
 
G_M000_IG58:                ;; offset=0x3C6D
       lea      rdi, [rbp-0x2238]
       mov      esi, 0x1600
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG59:                ;; offset=0x3C7E
       mov      eax, dword ptr [rbp-0x1118]
       cmp      eax, dword ptr [rbp-0x44]
       jl       G_M000_IG45
       mov      rdi, 0x72A8A2D4EF9C
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x6C]
       sar      eax, 31
       and      eax, 15
       add      eax, dword ptr [rbp-0x6C]
       sar      eax, 4
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x70]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       add      eax, dword ptr [rbp-0x74]
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  zmm0, zmmword ptr [rbp-0x1030]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
       mov      eax, dword ptr [rbp-0x6C]
       sar      eax, 31
       and      eax, 15
       add      eax, dword ptr [rbp-0x6C]
       sar      eax, 4
       inc      eax
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x70]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       add      eax, dword ptr [rbp-0x74]
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  zmm0, zmmword ptr [rbp-0x1070]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
       mov      eax, dword ptr [rbp-0x6C]
       sar      eax, 31
       and      eax, 15
       add      eax, dword ptr [rbp-0x6C]
       sar      eax, 4
       add      eax, 2
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x70]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       add      eax, dword ptr [rbp-0x74]
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  zmm0, zmmword ptr [rbp-0x10B0]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
       mov      eax, dword ptr [rbp-0x6C]
       sar      eax, 31
       and      eax, 15
       add      eax, dword ptr [rbp-0x6C]
       sar      eax, 4
       add      eax, 3
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x70]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
 
G_M000_IG60:                ;; offset=0x3D6B
       add      eax, dword ptr [rbp-0x74]
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  zmm0, zmmword ptr [rbp-0x10F0]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
       mov      eax, dword ptr [rbp-0x74]
       inc      eax
       mov      dword ptr [rbp-0x74], eax
 
G_M000_IG61:                ;; offset=0x3D90
       mov      eax, dword ptr [rbp-0x2238]
       dec      eax
       mov      dword ptr [rbp-0x2238], eax
       cmp      dword ptr [rbp-0x2238], 0
       jg       SHORT G_M000_IG63
 
G_M000_IG62:                ;; offset=0x3DA7
       lea      rdi, [rbp-0x2238]
       mov      esi, 0x1690
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG63:                ;; offset=0x3DB8
       mov      eax, dword ptr [rbp-0x74]
       cmp      eax, dword ptr [rbp+0x28]
       jl       G_M000_IG44
       mov      rdi, 0x72A8A2D4EFA0
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x70]
       inc      eax
       mov      dword ptr [rbp-0x70], eax
 
G_M000_IG64:                ;; offset=0x3DDB
       mov      eax, dword ptr [rbp-0x2238]
       dec      eax
       mov      dword ptr [rbp-0x2238], eax
       cmp      dword ptr [rbp-0x2238], 0
       jg       SHORT G_M000_IG66
 
G_M000_IG65:                ;; offset=0x3DF2
       lea      rdi, [rbp-0x2238]
       mov      esi, 0x169F
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG66:                ;; offset=0x3E03
       mov      eax, dword ptr [rbp-0x70]
       cmp      eax, dword ptr [rbp+0x20]
       jl       G_M000_IG04
       mov      rdi, 0x72A8A2D4EFA4
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x6C]
       add      eax, 64
       mov      dword ptr [rbp-0x6C], eax
 
G_M000_IG67:                ;; offset=0x3E27
       mov      eax, dword ptr [rbp-0x2238]
       dec      eax
       mov      dword ptr [rbp-0x2238], eax
       cmp      dword ptr [rbp-0x2238], 0
       jg       SHORT G_M000_IG69
 
G_M000_IG68:                ;; offset=0x3E3E
       lea      rdi, [rbp-0x2238]
       mov      esi, 0x16AF
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG69:                ;; offset=0x3E4F
       mov      eax, dword ptr [rbp-0x6C]
       cmp      eax, dword ptr [rbp-0x48]
       jl       G_M000_IG03
       mov      rdi, 0x72A8A2D4EFA8
       call     CORINFO_HELP_COUNTPROFILE32
       nop      
 
G_M000_IG70:                ;; offset=0x3E6B
       vzeroupper 
       add      rsp, 0x2240
       pop      rbp
       ret      
 
; Total bytes of code 15991


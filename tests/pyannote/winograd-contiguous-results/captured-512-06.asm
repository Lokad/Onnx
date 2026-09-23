; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:MultiplyWinograd512(ptr,ptr,ptr,int,int) (Instrumented Tier0)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Instrumented Tier0 code
; rbp based frame
; fully interruptible
; compiling with minopt

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       sub      rsp, 768
       lea      rbp, [rsp+0x300]
       vxorps   xmm8, xmm8, xmm8
       mov      rax, -672
       vmovdqa  xmmword ptr [rbp+rax-0x50], xmm8
       vmovdqa  xmmword ptr [rbp+rax-0x40], xmm8
       vmovdqa  xmmword ptr [rbp+rax-0x30], xmm8
       add      rax, 48
       jne      SHORT  -5 instr
       mov      qword ptr [rbp-0x50], rax
       mov      qword ptr [rbp-0x30], rdi
       mov      qword ptr [rbp-0x38], rsi
       mov      qword ptr [rbp-0x40], rdx
       mov      dword ptr [rbp-0x44], ecx
       mov      dword ptr [rbp-0x48], r8d
 
G_M000_IG02:                ;; offset=0x004E
       mov      dword ptr [rbp-0x300], 0x3E8
       xor      eax, eax
       mov      dword ptr [rbp-0x4C], eax
       jmp      G_M000_IG12
 
G_M000_IG03:                ;; offset=0x0062
       xor      eax, eax
       mov      dword ptr [rbp-0x50], eax
       jmp      G_M000_IG09
 
G_M000_IG04:                ;; offset=0x006C
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0xB0], zmm0
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
       mov      eax, dword ptr [rbp-0x4C]
       imul     eax, dword ptr [rbp-0x44]
       imul     eax, dword ptr [rbp-0x48]
       cdqe     
       shl      rax, 2
       add      rax, qword ptr [rbp-0x38]
       movsxd   rcx, dword ptr [rbp-0x50]
       lea      rax, [rax+4*rcx]
       mov      qword ptr [rbp-0x278], rax
       mov      eax, dword ptr [rbp-0x4C]
       imul     eax, dword ptr [rbp-0x44]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x30]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x280], rax
       xor      eax, eax
       mov      dword ptr [rbp-0x284], eax
       jmp      G_M000_IG06
 
G_M000_IG05:                ;; offset=0x0128
       mov      rdi, 0x781E6C70B080
       call     CORINFO_HELP_COUNTPROFILE32
       mov      rax, qword ptr [rbp-0x278]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0x2F0], zmm0
       mov      rax, qword ptr [rbp-0x280]
       vbroadcastss zmm0, dword ptr [rax]
       vmovups  zmm1, zmmword ptr [rbp-0xB0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x2F0]
       vmovups  zmmword ptr [rbp-0xB0], zmm1
       mov      rax, qword ptr [rbp-0x280]
       vbroadcastss zmm0, dword ptr [rax+0x04]
       vmovups  zmm1, zmmword ptr [rbp-0xF0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x2F0]
       vmovups  zmmword ptr [rbp-0xF0], zmm1
       mov      rax, qword ptr [rbp-0x280]
       vbroadcastss zmm0, dword ptr [rax+0x08]
       vmovups  zmm1, zmmword ptr [rbp-0x130]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x2F0]
       vmovups  zmmword ptr [rbp-0x130], zmm1
       mov      rax, qword ptr [rbp-0x280]
       vbroadcastss zmm0, dword ptr [rax+0x0C]
       vmovups  zmm1, zmmword ptr [rbp-0x170]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x2F0]
       vmovups  zmmword ptr [rbp-0x170], zmm1
       mov      rax, qword ptr [rbp-0x280]
       vbroadcastss zmm0, dword ptr [rax+0x10]
       vmovups  zmm1, zmmword ptr [rbp-0x1B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x2F0]
       vmovups  zmmword ptr [rbp-0x1B0], zmm1
       mov      rax, qword ptr [rbp-0x280]
       vbroadcastss zmm0, dword ptr [rax+0x14]
       vmovups  zmm1, zmmword ptr [rbp-0x1F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x2F0]
       vmovups  zmmword ptr [rbp-0x1F0], zmm1
       mov      rax, qword ptr [rbp-0x280]
       vbroadcastss zmm0, dword ptr [rax+0x18]
       vmovups  zmm1, zmmword ptr [rbp-0x230]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x2F0]
       vmovups  zmmword ptr [rbp-0x230], zmm1
       mov      rax, qword ptr [rbp-0x280]
       vbroadcastss zmm0, dword ptr [rax+0x1C]
       vmovups  zmm1, zmmword ptr [rbp-0x270]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x2F0]
       vmovups  zmmword ptr [rbp-0x270], zmm1
       movsxd   rax, dword ptr [rbp-0x48]
       mov      rcx, qword ptr [rbp-0x278]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x278], rax
       mov      rax, qword ptr [rbp-0x280]
       add      rax, 32
       mov      qword ptr [rbp-0x280], rax
       mov      eax, dword ptr [rbp-0x284]
       inc      eax
       mov      dword ptr [rbp-0x284], eax
 
G_M000_IG06:                ;; offset=0x02E3
       mov      eax, dword ptr [rbp-0x300]
       dec      eax
       mov      dword ptr [rbp-0x300], eax
       cmp      dword ptr [rbp-0x300], 0
       jg       SHORT G_M000_IG08
 
G_M000_IG07:                ;; offset=0x02FA
       lea      rdi, [rbp-0x300]
       mov      esi, 320
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG08:                ;; offset=0x030B
       mov      eax, dword ptr [rbp-0x284]
       cmp      eax, dword ptr [rbp-0x44]
       jl       G_M000_IG05
       mov      rdi, 0x781E6C70B084
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x4C]
       imul     eax, dword ptr [rbp-0x48]
       add      eax, dword ptr [rbp-0x50]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x2F8], rax
       mov      rax, qword ptr [rbp-0x2F8]
       vmovups  zmm0, zmmword ptr [rbp-0xB0]
       vmovups  zmmword ptr [rax], zmm0
       mov      rax, qword ptr [rbp-0x2F8]
       vmovups  zmm0, zmmword ptr [rbp-0xF0]
       vmovups  zmmword ptr [rax+0x40], zmm0
       mov      rax, qword ptr [rbp-0x2F8]
       vmovups  zmm0, zmmword ptr [rbp-0x130]
       vmovups  zmmword ptr [rax+0x80], zmm0
       mov      rax, qword ptr [rbp-0x2F8]
       vmovups  zmm0, zmmword ptr [rbp-0x170]
       vmovups  zmmword ptr [rax+0xC0], zmm0
       mov      rax, qword ptr [rbp-0x2F8]
       vmovups  zmm0, zmmword ptr [rbp-0x1B0]
       vmovups  zmmword ptr [rax+0x100], zmm0
       mov      rax, qword ptr [rbp-0x2F8]
       vmovups  zmm0, zmmword ptr [rbp-0x1F0]
       vmovups  zmmword ptr [rax+0x140], zmm0
       mov      rax, qword ptr [rbp-0x2F8]
       vmovups  zmm0, zmmword ptr [rbp-0x230]
       vmovups  zmmword ptr [rax+0x180], zmm0
       mov      rax, qword ptr [rbp-0x2F8]
       vmovups  zmm0, zmmword ptr [rbp-0x270]
       vmovups  zmmword ptr [rax+0x1C0], zmm0
       mov      eax, dword ptr [rbp-0x50]
       add      eax, 16
       mov      dword ptr [rbp-0x50], eax
 
G_M000_IG09:                ;; offset=0x040F
       mov      eax, dword ptr [rbp-0x300]
       dec      eax
       mov      dword ptr [rbp-0x300], eax
       cmp      dword ptr [rbp-0x300], 0
       jg       SHORT G_M000_IG11
 
G_M000_IG10:                ;; offset=0x0426
       lea      rdi, [rbp-0x300]
       mov      esi, 449
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG11:                ;; offset=0x0437
       mov      eax, dword ptr [rbp-0x50]
       cmp      eax, dword ptr [rbp-0x48]
       jl       G_M000_IG04
       mov      rdi, 0x781E6C70B088
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x4C]
       inc      eax
       mov      dword ptr [rbp-0x4C], eax
 
G_M000_IG12:                ;; offset=0x045A
       mov      eax, dword ptr [rbp-0x300]
       dec      eax
       mov      dword ptr [rbp-0x300], eax
       cmp      dword ptr [rbp-0x300], 0
       jg       SHORT G_M000_IG14
 
G_M000_IG13:                ;; offset=0x0471
       lea      rdi, [rbp-0x300]
       mov      esi, 461
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG14:                ;; offset=0x0482
       cmp      dword ptr [rbp-0x4C], 16
       jl       G_M000_IG03
       mov      rdi, 0x781E6C70B08C
       call     CORINFO_HELP_COUNTPROFILE32
       nop      
 
G_M000_IG15:                ;; offset=0x049C
       vzeroupper 
       add      rsp, 768
       pop      rbp
       ret      
 
; Total bytes of code 1192


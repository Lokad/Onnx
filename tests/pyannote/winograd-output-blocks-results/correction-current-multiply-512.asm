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
       mov      rdi, 0x7F34183418F8
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
       mov      rdi, 0x7F34183418FC
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
       mov      rdi, 0x7F3418341900
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
       mov      rdi, 0x7F3418341904
       call     CORINFO_HELP_COUNTPROFILE32
       nop      
 
G_M000_IG15:                ;; offset=0x049C
       vzeroupper 
       add      rsp, 768
       pop      rbp
       ret      
 
; Total bytes of code 1192

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:MultiplyWinograd512(ptr,ptr,ptr,int,int) (Tier1-OSR)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Tier1-OSR code
; OSR variant for entry point 0x140
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 10

G_M000_IG01:                ;; offset=0x0000
       mov      rax, qword ptr [rbp]
       push     rax
       sub      rsp, 16
       mov      qword ptr [rsp+0x318], r15
       mov      qword ptr [rsp+0x310], rbx
       lea      rbp, [rsp+0x10]
       mov      rdi, qword ptr [rbp+0x2E0]
       mov      rsi, qword ptr [rbp+0x2D8]
       mov      rdx, qword ptr [rbp+0x2D0]
       mov      ecx, dword ptr [rbp+0x2CC]
       mov      eax, dword ptr [rbp+0x2C8]
       mov      ebx, dword ptr [rbp+0x2C4]
       mov      r11d, dword ptr [rbp+0x2C0]
       vmovups  zmm0, zmmword ptr [rbp+0x260]
       vmovups  zmm1, zmmword ptr [rbp+0x220]
       vmovups  zmm2, zmmword ptr [rbp+0x1E0]
       vmovups  zmm3, zmmword ptr [rbp+0x1A0]
       vmovups  zmm4, zmmword ptr [rbp+0x160]
       vmovups  zmm5, zmmword ptr [rbp+0x120]
       vmovups  zmm6, zmmword ptr [rbp+0xE0]
       vmovups  zmm7, zmmword ptr [rbp+0xA0]
       mov      r10, qword ptr [rbp+0x98]
       mov      r8, qword ptr [rbp+0x90]
       mov      r9d, dword ptr [rbp+0x8C]
 
G_M000_IG02:                ;; offset=0x00B1
       jmp      G_M000_IG05
       align    [15 bytes for IG07]
 
G_M000_IG03:                ;; offset=0x00C5
       vxorps   ymm0, ymm0, ymm0
       vxorps   ymm1, ymm1, ymm1
       vxorps   ymm2, ymm2, ymm2
       vxorps   ymm3, ymm3, ymm3
       vxorps   ymm4, ymm4, ymm4
       vxorps   ymm5, ymm5, ymm5
       vxorps   ymm6, ymm6, ymm6
       vxorps   ymm7, ymm7, ymm7
       mov      r8d, ebx
       imul     r8d, ecx
       mov      r9d, r8d
       imul     r9d, eax
       movsxd   r9, r9d
       shl      r9, 2
       add      r9, rsi
       movsxd   r10, r11d
       lea      r10, [r9+4*r10]
       shl      r8d, 3
       movsxd   r8, r8d
       lea      r8, [rdi+4*r8]
       xor      r9d, r9d
       cmp      r9d, ecx
       jge      SHORT G_M000_IG06
 
G_M000_IG04:                ;; offset=0x0117
       vmovups  zmm8, zmmword ptr [r10]
       vfmadd231ps zmm0, zmm8, dword ptr [r8] {1to16}
       vfmadd231ps zmm1, zmm8, dword ptr [r8+0x04] {1to16}
       vfmadd231ps zmm2, zmm8, dword ptr [r8+0x08] {1to16}
       vfmadd231ps zmm3, zmm8, dword ptr [r8+0x0C] {1to16}
       vfmadd231ps zmm4, zmm8, dword ptr [r8+0x10] {1to16}
       vfmadd231ps zmm5, zmm8, dword ptr [r8+0x14] {1to16}
       vfmadd231ps zmm6, zmm8, dword ptr [r8+0x18] {1to16}
       vfmadd231ps zmm7, zmm8, dword ptr [r8+0x1C] {1to16}
       movsxd   r15, eax
       lea      r10, [r10+4*r15]
       add      r8, 32
       inc      r9d
 
G_M000_IG05:                ;; offset=0x0162
       cmp      r9d, ecx
       jl       SHORT G_M000_IG04
 
G_M000_IG06:                ;; offset=0x0167
       mov      r10d, ebx
       imul     r10d, eax
       add      r10d, r11d
       shl      r10d, 3
       movsxd   r8, r10d
       lea      r9, [rdx+4*r8]
       vmovups  zmmword ptr [r9], zmm0
       vmovups  zmmword ptr [r9+0x40], zmm1
       vmovups  zmmword ptr [r9+0x80], zmm2
       vmovups  zmmword ptr [r9+0xC0], zmm3
       vmovups  zmmword ptr [r9+0x100], zmm4
       vmovups  zmmword ptr [r9+0x140], zmm5
       vmovups  zmmword ptr [r9+0x180], zmm6
       vmovups  zmmword ptr [r9+0x1C0], zmm7
       add      r11d, 16
       cmp      r11d, eax
       jl       G_M000_IG03
 
G_M000_IG07:                ;; offset=0x01C0
       inc      ebx
       cmp      ebx, 16
       jge      SHORT G_M000_IG09
 
G_M000_IG08:                ;; offset=0x01C7
       xor      r11d, r11d
       test     eax, eax
       jg       G_M000_IG03
       jmp      SHORT G_M000_IG07
 
G_M000_IG09:                ;; offset=0x01D4
       vzeroupper 
       add      rsp, 784
       pop      rbx
       pop      r15
       pop      rbp
       ret      
 
; Total bytes of code 483

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
       mov      rdi, 0x7F34183418F8
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
       mov      rdi, 0x7F34183418FC
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
       mov      rdi, 0x7F3418341900
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
       mov      rdi, 0x7F3418341904
       call     CORINFO_HELP_COUNTPROFILE32
       nop      
 
G_M000_IG15:                ;; offset=0x049C
       vzeroupper 
       add      rsp, 768
       pop      rbp
       ret      
 
; Total bytes of code 1192

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:MultiplyWinograd512(ptr,ptr,ptr,int,int) (Tier1-OSR)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Tier1-OSR code
; OSR variant for entry point 0x140
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 12

G_M000_IG01:                ;; offset=0x0000
       mov      rax, qword ptr [rbp]
       push     rax
       sub      rsp, 16
       mov      qword ptr [rsp+0x318], r15
       mov      qword ptr [rsp+0x310], rbx
       lea      rbp, [rsp+0x10]
       mov      rdi, qword ptr [rbp+0x2E0]
       mov      rsi, qword ptr [rbp+0x2D8]
       mov      rdx, qword ptr [rbp+0x2D0]
       mov      ecx, dword ptr [rbp+0x2CC]
       mov      eax, dword ptr [rbp+0x2C8]
       mov      ebx, dword ptr [rbp+0x2C4]
       mov      r11d, dword ptr [rbp+0x2C0]
       vmovups  zmm0, zmmword ptr [rbp+0x260]
       vmovups  zmm1, zmmword ptr [rbp+0x220]
       vmovups  zmm2, zmmword ptr [rbp+0x1E0]
       vmovups  zmm3, zmmword ptr [rbp+0x1A0]
       vmovups  zmm4, zmmword ptr [rbp+0x160]
       vmovups  zmm5, zmmword ptr [rbp+0x120]
       vmovups  zmm6, zmmword ptr [rbp+0xE0]
       vmovups  zmm7, zmmword ptr [rbp+0xA0]
       mov      r10, qword ptr [rbp+0x98]
       mov      r8, qword ptr [rbp+0x90]
       mov      r9d, dword ptr [rbp+0x8C]
 
G_M000_IG02:                ;; offset=0x00B1
       jmp      G_M000_IG05
       align    [15 bytes for IG07]
 
G_M000_IG03:                ;; offset=0x00C5
       vxorps   ymm0, ymm0, ymm0
       vxorps   ymm1, ymm1, ymm1
       vxorps   ymm2, ymm2, ymm2
       vxorps   ymm3, ymm3, ymm3
       vxorps   ymm4, ymm4, ymm4
       vxorps   ymm5, ymm5, ymm5
       vxorps   ymm6, ymm6, ymm6
       vxorps   ymm7, ymm7, ymm7
       mov      r8d, ebx
       imul     r8d, ecx
       mov      r9d, r8d
       imul     r9d, eax
       movsxd   r9, r9d
       shl      r9, 2
       add      r9, rsi
       movsxd   r10, r11d
       lea      r10, [r9+4*r10]
       shl      r8d, 3
       movsxd   r8, r8d
       lea      r8, [rdi+4*r8]
       xor      r9d, r9d
       cmp      r9d, ecx
       jge      SHORT G_M000_IG06
 
G_M000_IG04:                ;; offset=0x0117
       vmovups  zmm8, zmmword ptr [r10]
       vfmadd231ps zmm0, zmm8, dword ptr [r8] {1to16}
       vfmadd231ps zmm1, zmm8, dword ptr [r8+0x04] {1to16}
       vfmadd231ps zmm2, zmm8, dword ptr [r8+0x08] {1to16}
       vfmadd231ps zmm3, zmm8, dword ptr [r8+0x0C] {1to16}
       vfmadd231ps zmm4, zmm8, dword ptr [r8+0x10] {1to16}
       vfmadd231ps zmm5, zmm8, dword ptr [r8+0x14] {1to16}
       vfmadd231ps zmm6, zmm8, dword ptr [r8+0x18] {1to16}
       vfmadd231ps zmm7, zmm8, dword ptr [r8+0x1C] {1to16}
       movsxd   r15, eax
       lea      r10, [r10+4*r15]
       add      r8, 32
       inc      r9d
 
G_M000_IG05:                ;; offset=0x0162
       cmp      r9d, ecx
       jl       SHORT G_M000_IG04
 
G_M000_IG06:                ;; offset=0x0167
       mov      r10d, ebx
       imul     r10d, eax
       add      r10d, r11d
       shl      r10d, 3
       movsxd   r8, r10d
       lea      r9, [rdx+4*r8]
       vmovups  zmmword ptr [r9], zmm0
       vmovups  zmmword ptr [r9+0x40], zmm1
       vmovups  zmmword ptr [r9+0x80], zmm2
       vmovups  zmmword ptr [r9+0xC0], zmm3
       vmovups  zmmword ptr [r9+0x100], zmm4
       vmovups  zmmword ptr [r9+0x140], zmm5
       vmovups  zmmword ptr [r9+0x180], zmm6
       vmovups  zmmword ptr [r9+0x1C0], zmm7
       add      r11d, 16
       cmp      r11d, eax
       jl       G_M000_IG03
 
G_M000_IG07:                ;; offset=0x01C0
       inc      ebx
       cmp      ebx, 16
       jge      SHORT G_M000_IG09
 
G_M000_IG08:                ;; offset=0x01C7
       xor      r11d, r11d
       test     eax, eax
       jg       G_M000_IG03
       jmp      SHORT G_M000_IG07
 
G_M000_IG09:                ;; offset=0x01D4
       vzeroupper 
       add      rsp, 784
       pop      rbx
       pop      r15
       pop      rbp
       ret      
 
; Total bytes of code 483

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:MultiplyWinograd512(ptr,ptr,ptr,int,int) (Tier1)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Tier1 code
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 11

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       push     r15
       push     rbx
       lea      rbp, [rsp+0x10]
 
G_M000_IG02:                ;; offset=0x0009
       xor      eax, eax
       movsxd   r9, r8d
       shl      r9, 2
       jmp      SHORT G_M000_IG04
       align    [0 bytes for IG07]
 
G_M000_IG03:                ;; offset=0x0014
       inc      eax
       cmp      eax, 16
       jge      G_M000_IG09
 
G_M000_IG04:                ;; offset=0x001F
       xor      r10d, r10d
       cmp      r10d, r8d
       jge      SHORT G_M000_IG03
 
G_M000_IG05:                ;; offset=0x0027
       vxorps   ymm0, ymm0, ymm0
       vxorps   ymm1, ymm1, ymm1
       vxorps   ymm2, ymm2, ymm2
       vxorps   ymm3, ymm3, ymm3
       vxorps   ymm4, ymm4, ymm4
       vxorps   ymm5, ymm5, ymm5
       vxorps   ymm6, ymm6, ymm6
       vxorps   ymm7, ymm7, ymm7
       mov      r11d, eax
       imul     r11d, ecx
       mov      ebx, r11d
       imul     ebx, r8d
       movsxd   rbx, ebx
       shl      rbx, 2
       add      rbx, rsi
       movsxd   r15, r10d
       lea      rbx, [rbx+4*r15]
       shl      r11d, 3
       movsxd   r11, r11d
       lea      r11, [rdi+4*r11]
       test     ecx, ecx
       jle      SHORT G_M000_IG08
 
G_M000_IG06:                ;; offset=0x0075
       mov      r15d, ecx
 
G_M000_IG07:                ;; offset=0x0078
       vmovups  zmm8, zmmword ptr [rbx]
       vfmadd231ps zmm0, zmm8, dword ptr [r11] {1to16}
       vfmadd231ps zmm1, zmm8, dword ptr [r11+0x04] {1to16}
       vfmadd231ps zmm2, zmm8, dword ptr [r11+0x08] {1to16}
       vfmadd231ps zmm3, zmm8, dword ptr [r11+0x0C] {1to16}
       vfmadd231ps zmm4, zmm8, dword ptr [r11+0x10] {1to16}
       vfmadd231ps zmm5, zmm8, dword ptr [r11+0x14] {1to16}
       vfmadd231ps zmm6, zmm8, dword ptr [r11+0x18] {1to16}
       vfmadd231ps zmm7, zmm8, dword ptr [r11+0x1C] {1to16}
       add      rbx, r9
       add      r11, 32
       dec      r15d
       jne      SHORT G_M000_IG07
 
G_M000_IG08:                ;; offset=0x00C1
       mov      r11d, eax
       imul     r11d, r8d
       add      r11d, r10d
       shl      r11d, 3
       movsxd   r11, r11d
       lea      r11, [rdx+4*r11]
       vmovups  zmmword ptr [r11], zmm0
       vmovups  zmmword ptr [r11+0x40], zmm1
       vmovups  zmmword ptr [r11+0x80], zmm2
       vmovups  zmmword ptr [r11+0xC0], zmm3
       vmovups  zmmword ptr [r11+0x100], zmm4
       vmovups  zmmword ptr [r11+0x140], zmm5
       vmovups  zmmword ptr [r11+0x180], zmm6
       vmovups  zmmword ptr [r11+0x1C0], zmm7
       add      r10d, 16
       cmp      r10d, r8d
       jl       G_M000_IG05
       jmp      G_M000_IG03
 
G_M000_IG09:                ;; offset=0x011F
       vzeroupper 
       pop      rbx
       pop      r15
       pop      rbp
       ret      
 
; Total bytes of code 295


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


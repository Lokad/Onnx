; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:MultiplyWinograd256(ptr,ptr,ptr,int,int) (Tier1-OSR)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
; Tier1-OSR code
; OSR variant for entry point 0x140
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 5

G_M000_IG01:                ;; offset=0x0000
       mov      rax, qword ptr [rbp]
       push     rax
       sub      rsp, 16
       mov      qword ptr [rsp+0x1B8], r15
       mov      qword ptr [rsp+0x1B0], rbx
       lea      rbp, [rsp+0x10]
       mov      rdi, qword ptr [rbp+0x180]
       mov      rsi, qword ptr [rbp+0x178]
       mov      rdx, qword ptr [rbp+0x170]
       mov      ecx, dword ptr [rbp+0x16C]
       mov      eax, dword ptr [rbp+0x168]
       mov      ebx, dword ptr [rbp+0x164]
       mov      r11d, dword ptr [rbp+0x160]
       vmovups  ymm0, ymmword ptr [rbp+0x140]
       vmovups  ymm1, ymmword ptr [rbp+0x120]
       vmovups  ymm2, ymmword ptr [rbp+0x100]
       vmovups  ymm3, ymmword ptr [rbp+0xE0]
       vmovups  ymm4, ymmword ptr [rbp+0xC0]
       vmovups  ymm5, ymmword ptr [rbp+0xA0]
       vmovups  ymm6, ymmword ptr [rbp+0x80]
       vmovups  ymm7, ymmword ptr [rbp+0x60]
       mov      r10, qword ptr [rbp+0x58]
       mov      r8, qword ptr [rbp+0x50]
       mov      r9d, dword ptr [rbp+0x4C]
 
G_M000_IG02:                ;; offset=0x0095
       jmp      G_M000_IG05
       align    [8 bytes for IG07]
 
G_M000_IG03:                ;; offset=0x00A2
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
 
G_M000_IG04:                ;; offset=0x00F4
       vmovups  ymm8, ymmword ptr [r10]
       vbroadcastss ymm9, dword ptr [r8]
       vfmadd231ps ymm0, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r8+0x04]
       vfmadd231ps ymm1, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r8+0x08]
       vfmadd231ps ymm2, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r8+0x0C]
       vfmadd231ps ymm3, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r8+0x10]
       vfmadd231ps ymm4, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r8+0x14]
       vfmadd231ps ymm5, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r8+0x18]
       vfmadd231ps ymm6, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r8+0x1C]
       vfmadd231ps ymm7, ymm8, ymm9
       movsxd   r15, eax
       lea      r10, [r10+4*r15]
       add      r8, 32
       inc      r9d
 
G_M000_IG05:                ;; offset=0x015E
       cmp      r9d, ecx
       jl       SHORT G_M000_IG04
 
G_M000_IG06:                ;; offset=0x0163
       mov      r10d, ebx
       imul     r10d, eax
       add      r10d, r11d
       shl      r10d, 3
       movsxd   r8, r10d
       lea      r9, [rdx+4*r8]
       vmovups  ymmword ptr [r9], ymm0
       vmovups  ymmword ptr [r9+0x20], ymm1
       vmovups  ymmword ptr [r9+0x40], ymm2
       vmovups  ymmword ptr [r9+0x60], ymm3
       vmovups  ymmword ptr [r9+0x80], ymm4
       vmovups  ymmword ptr [r9+0xA0], ymm5
       vmovups  ymmword ptr [r9+0xC0], ymm6
       vmovups  ymmword ptr [r9+0xE0], ymm7
       add      r11d, 8
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
       add      rsp, 432
       pop      rbx
       pop      r15
       pop      rbp
       ret      
 
; Total bytes of code 483


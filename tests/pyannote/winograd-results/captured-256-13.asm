; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:MultiplyWinograd256(ptr,ptr,ptr,int,int) (Tier1)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
; Tier1 code
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 5

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       push     r15
       push     rbx
       lea      rbp, [rsp+0x10]
 
G_M000_IG02:                ;; offset=0x0009
       xor      eax, eax
       movsxd   r9, r8d
       shl      r9, 2
       jmp      G_M000_IG08
       align    [0 bytes for IG05]
 
G_M000_IG03:                ;; offset=0x0017
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
       jle      SHORT G_M000_IG06
 
G_M000_IG04:                ;; offset=0x0065
       mov      r15d, ecx
 
G_M000_IG05:                ;; offset=0x0068
       vmovups  ymm8, ymmword ptr [rbx]
       vbroadcastss ymm9, dword ptr [r11]
       vfmadd231ps ymm0, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r11+0x04]
       vfmadd231ps ymm1, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r11+0x08]
       vfmadd231ps ymm2, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r11+0x0C]
       vfmadd231ps ymm3, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r11+0x10]
       vfmadd231ps ymm4, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r11+0x14]
       vfmadd231ps ymm5, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r11+0x18]
       vfmadd231ps ymm6, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r11+0x1C]
       vfmadd231ps ymm7, ymm8, ymm9
       add      rbx, r9
       add      r11, 32
       dec      r15d
       jne      SHORT G_M000_IG05
 
G_M000_IG06:                ;; offset=0x00CF
       mov      r11d, eax
       imul     r11d, r8d
       add      r11d, r10d
       shl      r11d, 3
       movsxd   r11, r11d
       lea      r11, [rdx+4*r11]
       vmovups  ymmword ptr [r11], ymm0
       vmovups  ymmword ptr [r11+0x20], ymm1
       vmovups  ymmword ptr [r11+0x40], ymm2
       vmovups  ymmword ptr [r11+0x60], ymm3
       vmovups  ymmword ptr [r11+0x80], ymm4
       vmovups  ymmword ptr [r11+0xA0], ymm5
       vmovups  ymmword ptr [r11+0xC0], ymm6
       vmovups  ymmword ptr [r11+0xE0], ymm7
       add      r10d, 8
       cmp      r10d, r8d
       jl       G_M000_IG03
 
G_M000_IG07:                ;; offset=0x012C
       inc      eax
       cmp      eax, 16
       jge      SHORT G_M000_IG09
 
G_M000_IG08:                ;; offset=0x0133
       xor      r10d, r10d
       cmp      r10d, r8d
       jl       G_M000_IG03
       jmp      SHORT G_M000_IG07
 
G_M000_IG09:                ;; offset=0x0141
       vzeroupper 
       pop      rbx
       pop      r15
       pop      rbp
       ret      
 
; Total bytes of code 329


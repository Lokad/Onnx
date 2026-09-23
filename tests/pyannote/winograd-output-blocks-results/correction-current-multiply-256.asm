; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:MultiplyWinograd256(ptr,ptr,ptr,int,int) (Instrumented Tier0)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
; Instrumented Tier0 code
; rbp based frame
; fully interruptible
; compiling with minopt

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       sub      rsp, 416
       lea      rbp, [rsp+0x1A0]
       vxorps   xmm8, xmm8, xmm8
       vmovdqa  xmmword ptr [rbp-0x190], xmm8
       vmovdqa  xmmword ptr [rbp-0x180], xmm8
       mov      rax, -288
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
 
G_M000_IG02:                ;; offset=0x005E
       mov      dword ptr [rbp-0x1A0], 0x3E8
       xor      eax, eax
       mov      dword ptr [rbp-0x4C], eax
       jmp      G_M000_IG12
 
G_M000_IG03:                ;; offset=0x0072
       xor      eax, eax
       mov      dword ptr [rbp-0x50], eax
       jmp      G_M000_IG09
 
G_M000_IG04:                ;; offset=0x007C
       vxorps   ymm0, ymm0, ymm0
       vmovups  ymmword ptr [rbp-0x70], ymm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  ymmword ptr [rbp-0x90], ymm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  ymmword ptr [rbp-0xB0], ymm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  ymmword ptr [rbp-0xD0], ymm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  ymmword ptr [rbp-0xF0], ymm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  ymmword ptr [rbp-0x110], ymm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  ymmword ptr [rbp-0x130], ymm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  ymmword ptr [rbp-0x150], ymm0
       mov      eax, dword ptr [rbp-0x4C]
       imul     eax, dword ptr [rbp-0x44]
       imul     eax, dword ptr [rbp-0x48]
       cdqe     
       shl      rax, 2
       add      rax, qword ptr [rbp-0x38]
       movsxd   rcx, dword ptr [rbp-0x50]
       lea      rax, [rax+4*rcx]
       mov      qword ptr [rbp-0x158], rax
       mov      eax, dword ptr [rbp-0x4C]
       imul     eax, dword ptr [rbp-0x44]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x30]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x160], rax
       xor      eax, eax
       mov      dword ptr [rbp-0x164], eax
       jmp      G_M000_IG06
 
G_M000_IG05:                ;; offset=0x0125
       mov      rdi, 0x77BDFC56AFC8
       call     CORINFO_HELP_COUNTPROFILE32
       mov      rax, qword ptr [rbp-0x158]
       vmovups  ymm0, ymmword ptr [rax]
       vmovups  ymmword ptr [rbp-0x190], ymm0
       mov      rax, qword ptr [rbp-0x160]
       vbroadcastss ymm0, dword ptr [rax]
       vmovups  ymm1, ymmword ptr [rbp-0x70]
       vfmadd231ps ymm1, ymm0, ymmword ptr [rbp-0x190]
       vmovups  ymmword ptr [rbp-0x70], ymm1
       mov      rax, qword ptr [rbp-0x160]
       vbroadcastss ymm0, dword ptr [rax+0x04]
       vmovups  ymm1, ymmword ptr [rbp-0x90]
       vfmadd231ps ymm1, ymm0, ymmword ptr [rbp-0x190]
       vmovups  ymmword ptr [rbp-0x90], ymm1
       mov      rax, qword ptr [rbp-0x160]
       vbroadcastss ymm0, dword ptr [rax+0x08]
       vmovups  ymm1, ymmword ptr [rbp-0xB0]
       vfmadd231ps ymm1, ymm0, ymmword ptr [rbp-0x190]
       vmovups  ymmword ptr [rbp-0xB0], ymm1
       mov      rax, qword ptr [rbp-0x160]
       vbroadcastss ymm0, dword ptr [rax+0x0C]
       vmovups  ymm1, ymmword ptr [rbp-0xD0]
       vfmadd231ps ymm1, ymm0, ymmword ptr [rbp-0x190]
       vmovups  ymmword ptr [rbp-0xD0], ymm1
       mov      rax, qword ptr [rbp-0x160]
       vbroadcastss ymm0, dword ptr [rax+0x10]
       vmovups  ymm1, ymmword ptr [rbp-0xF0]
       vfmadd231ps ymm1, ymm0, ymmword ptr [rbp-0x190]
       vmovups  ymmword ptr [rbp-0xF0], ymm1
       mov      rax, qword ptr [rbp-0x160]
       vbroadcastss ymm0, dword ptr [rax+0x14]
       vmovups  ymm1, ymmword ptr [rbp-0x110]
       vfmadd231ps ymm1, ymm0, ymmword ptr [rbp-0x190]
       vmovups  ymmword ptr [rbp-0x110], ymm1
       mov      rax, qword ptr [rbp-0x160]
       vbroadcastss ymm0, dword ptr [rax+0x18]
       vmovups  ymm1, ymmword ptr [rbp-0x130]
       vfmadd231ps ymm1, ymm0, ymmword ptr [rbp-0x190]
       vmovups  ymmword ptr [rbp-0x130], ymm1
       mov      rax, qword ptr [rbp-0x160]
       vbroadcastss ymm0, dword ptr [rax+0x1C]
       vmovups  ymm1, ymmword ptr [rbp-0x150]
       vfmadd231ps ymm1, ymm0, ymmword ptr [rbp-0x190]
       vmovups  ymmword ptr [rbp-0x150], ymm1
       movsxd   rax, dword ptr [rbp-0x48]
       mov      rcx, qword ptr [rbp-0x158]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x158], rax
       mov      rax, qword ptr [rbp-0x160]
       add      rax, 32
       mov      qword ptr [rbp-0x160], rax
       mov      eax, dword ptr [rbp-0x164]
       inc      eax
       mov      dword ptr [rbp-0x164], eax
 
G_M000_IG06:                ;; offset=0x02A6
       mov      eax, dword ptr [rbp-0x1A0]
       dec      eax
       mov      dword ptr [rbp-0x1A0], eax
       cmp      dword ptr [rbp-0x1A0], 0
       jg       SHORT G_M000_IG08
 
G_M000_IG07:                ;; offset=0x02BD
       lea      rdi, [rbp-0x1A0]
       mov      esi, 320
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG08:                ;; offset=0x02CE
       mov      eax, dword ptr [rbp-0x164]
       cmp      eax, dword ptr [rbp-0x44]
       jl       G_M000_IG05
       mov      rdi, 0x77BDFC56AFCC
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x4C]
       imul     eax, dword ptr [rbp-0x48]
       add      eax, dword ptr [rbp-0x50]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x198], rax
       mov      rax, qword ptr [rbp-0x198]
       vmovups  ymm0, ymmword ptr [rbp-0x70]
       vmovups  ymmword ptr [rax], ymm0
       mov      rax, qword ptr [rbp-0x198]
       vmovups  ymm0, ymmword ptr [rbp-0x90]
       vmovups  ymmword ptr [rax+0x20], ymm0
       mov      rax, qword ptr [rbp-0x198]
       vmovups  ymm0, ymmword ptr [rbp-0xB0]
       vmovups  ymmword ptr [rax+0x40], ymm0
       mov      rax, qword ptr [rbp-0x198]
       vmovups  ymm0, ymmword ptr [rbp-0xD0]
       vmovups  ymmword ptr [rax+0x60], ymm0
       mov      rax, qword ptr [rbp-0x198]
       vmovups  ymm0, ymmword ptr [rbp-0xF0]
       vmovups  ymmword ptr [rax+0x80], ymm0
       mov      rax, qword ptr [rbp-0x198]
       vmovups  ymm0, ymmword ptr [rbp-0x110]
       vmovups  ymmword ptr [rax+0xA0], ymm0
       mov      rax, qword ptr [rbp-0x198]
       vmovups  ymm0, ymmword ptr [rbp-0x130]
       vmovups  ymmword ptr [rax+0xC0], ymm0
       mov      rax, qword ptr [rbp-0x198]
       vmovups  ymm0, ymmword ptr [rbp-0x150]
       vmovups  ymmword ptr [rax+0xE0], ymm0
       mov      eax, dword ptr [rbp-0x50]
       add      eax, 8
       mov      dword ptr [rbp-0x50], eax
 
G_M000_IG09:                ;; offset=0x03BB
       mov      eax, dword ptr [rbp-0x1A0]
       dec      eax
       mov      dword ptr [rbp-0x1A0], eax
       cmp      dword ptr [rbp-0x1A0], 0
       jg       SHORT G_M000_IG11
 
G_M000_IG10:                ;; offset=0x03D2
       lea      rdi, [rbp-0x1A0]
       mov      esi, 447
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG11:                ;; offset=0x03E3
       mov      eax, dword ptr [rbp-0x50]
       cmp      eax, dword ptr [rbp-0x48]
       jl       G_M000_IG04
       mov      rdi, 0x77BDFC56AFD0
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x4C]
       inc      eax
       mov      dword ptr [rbp-0x4C], eax
 
G_M000_IG12:                ;; offset=0x0406
       mov      eax, dword ptr [rbp-0x1A0]
       dec      eax
       mov      dword ptr [rbp-0x1A0], eax
       cmp      dword ptr [rbp-0x1A0], 0
       jg       SHORT G_M000_IG14
 
G_M000_IG13:                ;; offset=0x041D
       lea      rdi, [rbp-0x1A0]
       mov      esi, 459
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG14:                ;; offset=0x042E
       cmp      dword ptr [rbp-0x4C], 16
       jl       G_M000_IG03
       mov      rdi, 0x77BDFC56AFD4
       call     CORINFO_HELP_COUNTPROFILE32
       nop      
 
G_M000_IG15:                ;; offset=0x0448
       vzeroupper 
       add      rsp, 416
       pop      rbp
       ret      
 
; Total bytes of code 1108

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

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:MultiplyWinograd256(ptr,ptr,ptr,int,int) (Instrumented Tier0)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
; Instrumented Tier0 code
; rbp based frame
; fully interruptible
; compiling with minopt

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       sub      rsp, 416
       lea      rbp, [rsp+0x1A0]
       vxorps   xmm8, xmm8, xmm8
       vmovdqa  xmmword ptr [rbp-0x190], xmm8
       vmovdqa  xmmword ptr [rbp-0x180], xmm8
       mov      rax, -288
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
 
G_M000_IG02:                ;; offset=0x005E
       mov      dword ptr [rbp-0x1A0], 0x3E8
       xor      eax, eax
       mov      dword ptr [rbp-0x4C], eax
       jmp      G_M000_IG12
 
G_M000_IG03:                ;; offset=0x0072
       xor      eax, eax
       mov      dword ptr [rbp-0x50], eax
       jmp      G_M000_IG09
 
G_M000_IG04:                ;; offset=0x007C
       vxorps   ymm0, ymm0, ymm0
       vmovups  ymmword ptr [rbp-0x70], ymm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  ymmword ptr [rbp-0x90], ymm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  ymmword ptr [rbp-0xB0], ymm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  ymmword ptr [rbp-0xD0], ymm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  ymmword ptr [rbp-0xF0], ymm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  ymmword ptr [rbp-0x110], ymm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  ymmword ptr [rbp-0x130], ymm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  ymmword ptr [rbp-0x150], ymm0
       mov      eax, dword ptr [rbp-0x4C]
       imul     eax, dword ptr [rbp-0x44]
       imul     eax, dword ptr [rbp-0x48]
       cdqe     
       shl      rax, 2
       add      rax, qword ptr [rbp-0x38]
       movsxd   rcx, dword ptr [rbp-0x50]
       lea      rax, [rax+4*rcx]
       mov      qword ptr [rbp-0x158], rax
       mov      eax, dword ptr [rbp-0x4C]
       imul     eax, dword ptr [rbp-0x44]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x30]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x160], rax
       xor      eax, eax
       mov      dword ptr [rbp-0x164], eax
       jmp      G_M000_IG06
 
G_M000_IG05:                ;; offset=0x0125
       mov      rdi, 0x77BDFC56AFC8
       call     CORINFO_HELP_COUNTPROFILE32
       mov      rax, qword ptr [rbp-0x158]
       vmovups  ymm0, ymmword ptr [rax]
       vmovups  ymmword ptr [rbp-0x190], ymm0
       mov      rax, qword ptr [rbp-0x160]
       vbroadcastss ymm0, dword ptr [rax]
       vmovups  ymm1, ymmword ptr [rbp-0x70]
       vfmadd231ps ymm1, ymm0, ymmword ptr [rbp-0x190]
       vmovups  ymmword ptr [rbp-0x70], ymm1
       mov      rax, qword ptr [rbp-0x160]
       vbroadcastss ymm0, dword ptr [rax+0x04]
       vmovups  ymm1, ymmword ptr [rbp-0x90]
       vfmadd231ps ymm1, ymm0, ymmword ptr [rbp-0x190]
       vmovups  ymmword ptr [rbp-0x90], ymm1
       mov      rax, qword ptr [rbp-0x160]
       vbroadcastss ymm0, dword ptr [rax+0x08]
       vmovups  ymm1, ymmword ptr [rbp-0xB0]
       vfmadd231ps ymm1, ymm0, ymmword ptr [rbp-0x190]
       vmovups  ymmword ptr [rbp-0xB0], ymm1
       mov      rax, qword ptr [rbp-0x160]
       vbroadcastss ymm0, dword ptr [rax+0x0C]
       vmovups  ymm1, ymmword ptr [rbp-0xD0]
       vfmadd231ps ymm1, ymm0, ymmword ptr [rbp-0x190]
       vmovups  ymmword ptr [rbp-0xD0], ymm1
       mov      rax, qword ptr [rbp-0x160]
       vbroadcastss ymm0, dword ptr [rax+0x10]
       vmovups  ymm1, ymmword ptr [rbp-0xF0]
       vfmadd231ps ymm1, ymm0, ymmword ptr [rbp-0x190]
       vmovups  ymmword ptr [rbp-0xF0], ymm1
       mov      rax, qword ptr [rbp-0x160]
       vbroadcastss ymm0, dword ptr [rax+0x14]
       vmovups  ymm1, ymmword ptr [rbp-0x110]
       vfmadd231ps ymm1, ymm0, ymmword ptr [rbp-0x190]
       vmovups  ymmword ptr [rbp-0x110], ymm1
       mov      rax, qword ptr [rbp-0x160]
       vbroadcastss ymm0, dword ptr [rax+0x18]
       vmovups  ymm1, ymmword ptr [rbp-0x130]
       vfmadd231ps ymm1, ymm0, ymmword ptr [rbp-0x190]
       vmovups  ymmword ptr [rbp-0x130], ymm1
       mov      rax, qword ptr [rbp-0x160]
       vbroadcastss ymm0, dword ptr [rax+0x1C]
       vmovups  ymm1, ymmword ptr [rbp-0x150]
       vfmadd231ps ymm1, ymm0, ymmword ptr [rbp-0x190]
       vmovups  ymmword ptr [rbp-0x150], ymm1
       movsxd   rax, dword ptr [rbp-0x48]
       mov      rcx, qword ptr [rbp-0x158]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x158], rax
       mov      rax, qword ptr [rbp-0x160]
       add      rax, 32
       mov      qword ptr [rbp-0x160], rax
       mov      eax, dword ptr [rbp-0x164]
       inc      eax
       mov      dword ptr [rbp-0x164], eax
 
G_M000_IG06:                ;; offset=0x02A6
       mov      eax, dword ptr [rbp-0x1A0]
       dec      eax
       mov      dword ptr [rbp-0x1A0], eax
       cmp      dword ptr [rbp-0x1A0], 0
       jg       SHORT G_M000_IG08
 
G_M000_IG07:                ;; offset=0x02BD
       lea      rdi, [rbp-0x1A0]
       mov      esi, 320
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG08:                ;; offset=0x02CE
       mov      eax, dword ptr [rbp-0x164]
       cmp      eax, dword ptr [rbp-0x44]
       jl       G_M000_IG05
       mov      rdi, 0x77BDFC56AFCC
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x4C]
       imul     eax, dword ptr [rbp-0x48]
       add      eax, dword ptr [rbp-0x50]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x198], rax
       mov      rax, qword ptr [rbp-0x198]
       vmovups  ymm0, ymmword ptr [rbp-0x70]
       vmovups  ymmword ptr [rax], ymm0
       mov      rax, qword ptr [rbp-0x198]
       vmovups  ymm0, ymmword ptr [rbp-0x90]
       vmovups  ymmword ptr [rax+0x20], ymm0
       mov      rax, qword ptr [rbp-0x198]
       vmovups  ymm0, ymmword ptr [rbp-0xB0]
       vmovups  ymmword ptr [rax+0x40], ymm0
       mov      rax, qword ptr [rbp-0x198]
       vmovups  ymm0, ymmword ptr [rbp-0xD0]
       vmovups  ymmword ptr [rax+0x60], ymm0
       mov      rax, qword ptr [rbp-0x198]
       vmovups  ymm0, ymmword ptr [rbp-0xF0]
       vmovups  ymmword ptr [rax+0x80], ymm0
       mov      rax, qword ptr [rbp-0x198]
       vmovups  ymm0, ymmword ptr [rbp-0x110]
       vmovups  ymmword ptr [rax+0xA0], ymm0
       mov      rax, qword ptr [rbp-0x198]
       vmovups  ymm0, ymmword ptr [rbp-0x130]
       vmovups  ymmword ptr [rax+0xC0], ymm0
       mov      rax, qword ptr [rbp-0x198]
       vmovups  ymm0, ymmword ptr [rbp-0x150]
       vmovups  ymmword ptr [rax+0xE0], ymm0
       mov      eax, dword ptr [rbp-0x50]
       add      eax, 8
       mov      dword ptr [rbp-0x50], eax
 
G_M000_IG09:                ;; offset=0x03BB
       mov      eax, dword ptr [rbp-0x1A0]
       dec      eax
       mov      dword ptr [rbp-0x1A0], eax
       cmp      dword ptr [rbp-0x1A0], 0
       jg       SHORT G_M000_IG11
 
G_M000_IG10:                ;; offset=0x03D2
       lea      rdi, [rbp-0x1A0]
       mov      esi, 447
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG11:                ;; offset=0x03E3
       mov      eax, dword ptr [rbp-0x50]
       cmp      eax, dword ptr [rbp-0x48]
       jl       G_M000_IG04
       mov      rdi, 0x77BDFC56AFD0
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x4C]
       inc      eax
       mov      dword ptr [rbp-0x4C], eax
 
G_M000_IG12:                ;; offset=0x0406
       mov      eax, dword ptr [rbp-0x1A0]
       dec      eax
       mov      dword ptr [rbp-0x1A0], eax
       cmp      dword ptr [rbp-0x1A0], 0
       jg       SHORT G_M000_IG14
 
G_M000_IG13:                ;; offset=0x041D
       lea      rdi, [rbp-0x1A0]
       mov      esi, 459
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG14:                ;; offset=0x042E
       cmp      dword ptr [rbp-0x4C], 16
       jl       G_M000_IG03
       mov      rdi, 0x77BDFC56AFD4
       call     CORINFO_HELP_COUNTPROFILE32
       nop      
 
G_M000_IG15:                ;; offset=0x0448
       vzeroupper 
       add      rsp, 416
       pop      rbp
       ret      
 
; Total bytes of code 1108

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:MultiplyWinograd256(ptr,ptr,ptr,int,int) (Tier1-OSR)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
; Tier1-OSR code
; OSR variant for entry point 0x140
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 6

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
       jne      SHORT G_M000_IG07
 
G_M000_IG08:                ;; offset=0x00DF
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
       jl       G_M000_IG05
       jmp      G_M000_IG03
 
G_M000_IG09:                ;; offset=0x0141
       vzeroupper 
       pop      rbx
       pop      r15
       pop      rbp
       ret      
 
; Total bytes of code 329


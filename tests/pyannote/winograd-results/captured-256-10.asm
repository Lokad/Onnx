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
       mov      rdi, 0x7F3042114B70
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
       mov      rdi, 0x7F3042114B74
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
       mov      rdi, 0x7F3042114B78
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
       mov      rdi, 0x7F3042114B7C
       call     CORINFO_HELP_COUNTPROFILE32
       nop      
 
G_M000_IG15:                ;; offset=0x0448
       vzeroupper 
       add      rsp, 416
       pop      rbp
       ret      
 
; Total bytes of code 1108


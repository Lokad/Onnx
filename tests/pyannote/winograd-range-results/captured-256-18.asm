; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool (Instrumented Tier0)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
; Instrumented Tier0 code
; rbp based frame
; fully interruptible
; compiling with minopt

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       sub      rsp, 288
       lea      rbp, [rsp+0x120]
       vxorps   xmm8, xmm8, xmm8
       vmovdqa  xmmword ptr [rbp-0x110], xmm8
       mov      rax, -192
       vmovdqa  xmmword ptr [rbp+rax-0x40], xmm8
       vmovdqa  xmmword ptr [rbp+rax-0x30], xmm8
       vmovdqa  xmmword ptr [rbp+rax-0x20], xmm8
       add      rax, 48
       jne      SHORT  -5 instr
       mov      qword ptr [rbp-0x40], rax
       mov      bword ptr [rbp-0x38], rdi
       mov      qword ptr [rbp-0x30], rsi
 
G_M000_IG02:                ;; offset=0x004B
       mov      dword ptr [rbp-0x118], 0x3E8
       xor      eax, eax
       mov      dword ptr [rbp-0x3C], eax
       lea      rdi, [rbp-0x38]
       call     [System.ReadOnlySpan`1[float]:GetPinnableReference():byref:this]
       mov      bword ptr [rbp-0x50], rax
       mov      rax, bword ptr [rbp-0x50]
       mov      qword ptr [rbp-0x120], rax
       mov      rax, qword ptr [rbp-0x120]
       mov      qword ptr [rbp-0x48], rax
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
       vmovups  ymmword ptr [rbp-0xF0], ymm0
       vbroadcastss ymm0, dword ptr [reloc @RWD04]
       vmovups  ymmword ptr [rbp-0x110], ymm0
       jmp      SHORT G_M000_IG06
 
G_M000_IG03:                ;; offset=0x00A2
       movsxd   rax, dword ptr [rbp-0x3C]
       mov      rcx, qword ptr [rbp-0x48]
       vmovups  ymm0, ymmword ptr [rcx+4*rax]
       vpand    ymm0, ymm0, ymmword ptr [rbp-0xF0]
       vcmpgtps ymm0, ymm0, ymmword ptr [rbp-0x110]
       vmovmskps rax, ymm0
       test     eax, eax
       je       SHORT G_M000_IG05
       mov      rdi, 0x761AD7B0A9A8
       call     CORINFO_HELP_COUNTPROFILE32
       xor      eax, eax
 
G_M000_IG04:                ;; offset=0x00D9
       vzeroupper 
       add      rsp, 288
       pop      rbp
       ret      
 
G_M000_IG05:                ;; offset=0x00E5
       mov      rdi, 0x761AD7B0A9AC
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x3C]
       add      eax, 8
       mov      dword ptr [rbp-0x3C], eax
 
G_M000_IG06:                ;; offset=0x00FD
       mov      eax, dword ptr [rbp-0x118]
       dec      eax
       mov      dword ptr [rbp-0x118], eax
       cmp      dword ptr [rbp-0x118], 0
       jg       SHORT G_M000_IG08
 
G_M000_IG07:                ;; offset=0x0114
       lea      rdi, [rbp-0x118]
       mov      esi, 191
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG08:                ;; offset=0x0125
       mov      eax, dword ptr [rbp-0x30]
       add      eax, -8
       cmp      dword ptr [rbp-0x3C], eax
       jle      G_M000_IG03
 
G_M000_IG09:                ;; offset=0x0134
       mov      rdi, 0x761AD7B0A9B0
       call     CORINFO_HELP_COUNTPROFILE32
       xor      eax, eax
       mov      bword ptr [rbp-0x50], rax
       jmp      SHORT G_M000_IG13
 
G_M000_IG10:                ;; offset=0x014B
       mov      eax, dword ptr [rbp-0x30]
       cmp      dword ptr [rbp-0x3C], eax
       jae      G_M000_IG17
       mov      eax, dword ptr [rbp-0x3C]
       mov      rcx, bword ptr [rbp-0x38]
       vmovss   xmm0, dword ptr [rcx+4*rax]
       vandps   xmm0, xmm0, xmmword ptr [reloc @RWD16]
       vucomiss xmm0, dword ptr [reloc @RWD04]
       jbe      SHORT G_M000_IG12
       mov      rdi, 0x761AD7B0A9B4
       call     CORINFO_HELP_COUNTPROFILE32
       xor      eax, eax
 
G_M000_IG11:                ;; offset=0x0186
       vzeroupper 
       add      rsp, 288
       pop      rbp
       ret      
 
G_M000_IG12:                ;; offset=0x0192
       mov      rdi, 0x761AD7B0A9B8
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x3C]
       inc      eax
       mov      dword ptr [rbp-0x3C], eax
 
G_M000_IG13:                ;; offset=0x01A9
       mov      eax, dword ptr [rbp-0x118]
       dec      eax
       mov      dword ptr [rbp-0x118], eax
       cmp      dword ptr [rbp-0x118], 0
       jg       SHORT G_M000_IG15
 
G_M000_IG14:                ;; offset=0x01C0
       lea      rdi, [rbp-0x118]
       mov      esi, 235
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG15:                ;; offset=0x01D1
       mov      eax, dword ptr [rbp-0x3C]
       cmp      eax, dword ptr [rbp-0x30]
       jl       G_M000_IG10
       mov      rdi, 0x761AD7B0A9BC
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, 1
 
G_M000_IG16:                ;; offset=0x01F1
       vzeroupper 
       add      rsp, 288
       pop      rbp
       ret      
 
G_M000_IG17:                ;; offset=0x01FD
       call     CORINFO_HELP_RNGCHKFAIL
       int3     
 
RWD00  	dd	7FFFFFFFh		;       nan
RWD04  	dd	7E7FFFFFh		; 8.50706e+37
RWD08  	dd	00000000h, 00000000h
RWD16  	dq	7FFFFFFF7FFFFFFFh, 7FFFFFFF7FFFFFFFh

; Total bytes of code 515


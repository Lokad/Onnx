; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool (Instrumented Tier0)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Instrumented Tier0 code
; rbp based frame
; fully interruptible
; compiling with minopt

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       sub      rsp, 320
       lea      rbp, [rsp+0x140]
       vxorps   xmm8, xmm8, xmm8
       vmovdqu32 zmmword ptr [rbp-0x130], zmm8
       vmovdqu32 zmmword ptr [rbp-0xF0], zmm8
       vmovdqu32 zmmword ptr [rbp-0xB0], zmm8
       vmovdqu32 zmmword ptr [rbp-0x80], zmm8
       xor      eax, eax
       mov      qword ptr [rbp-0x40], rax
       mov      bword ptr [rbp-0x38], rdi
       mov      qword ptr [rbp-0x30], rsi
 
G_M000_IG02:                ;; offset=0x0048
       mov      dword ptr [rbp-0x138], 0x3E8
       xor      eax, eax
       mov      dword ptr [rbp-0x3C], eax
       lea      rdi, [rbp-0x38]
       call     [System.ReadOnlySpan`1[float]:GetPinnableReference():byref:this]
       mov      bword ptr [rbp-0x50], rax
       mov      rax, bword ptr [rbp-0x50]
       mov      qword ptr [rbp-0x140], rax
       mov      rax, qword ptr [rbp-0x140]
       mov      qword ptr [rbp-0x48], rax
       vbroadcastss zmm0, dword ptr [reloc @RWD00]
       vmovups  zmmword ptr [rbp-0xB0], zmm0
       vbroadcastss zmm0, dword ptr [reloc @RWD04]
       vmovups  zmmword ptr [rbp-0xF0], zmm0
       jmp      SHORT G_M000_IG06
 
G_M000_IG03:                ;; offset=0x00A5
       movsxd   rax, dword ptr [rbp-0x3C]
       mov      rcx, qword ptr [rbp-0x48]
       vmovups  zmm0, zmmword ptr [rcx+4*rax]
       vpandd   zmm0, zmm0, zmmword ptr [rbp-0xB0]
       vcmpgtps k1, zmm0, zmmword ptr [rbp-0xF0]
       kmovw    eax, k1
       test     rax, rax
       je       SHORT G_M000_IG05
       mov      rdi, 0x709EADF1E460
       call     CORINFO_HELP_COUNTPROFILE32
       xor      eax, eax
 
G_M000_IG04:                ;; offset=0x00E3
       vzeroupper 
       add      rsp, 320
       pop      rbp
       ret      
 
G_M000_IG05:                ;; offset=0x00EF
       mov      rdi, 0x709EADF1E464
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x3C]
       add      eax, 16
       mov      dword ptr [rbp-0x3C], eax
 
G_M000_IG06:                ;; offset=0x0107
       mov      eax, dword ptr [rbp-0x138]
       dec      eax
       mov      dword ptr [rbp-0x138], eax
       cmp      dword ptr [rbp-0x138], 0
       jg       SHORT G_M000_IG08
 
G_M000_IG07:                ;; offset=0x011E
       lea      rdi, [rbp-0x138]
       mov      esi, 93
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG08:                ;; offset=0x012F
       mov      eax, dword ptr [rbp-0x30]
       add      eax, -16
       cmp      dword ptr [rbp-0x3C], eax
       jle      G_M000_IG03
       xor      eax, eax
       mov      bword ptr [rbp-0x50], rax
       jmp      SHORT G_M000_IG12
 
G_M000_IG09:                ;; offset=0x0146
       mov      eax, dword ptr [rbp-0x30]
       cmp      dword ptr [rbp-0x3C], eax
       jae      G_M000_IG16
       mov      eax, dword ptr [rbp-0x3C]
       mov      rcx, bword ptr [rbp-0x38]
       vmovss   xmm0, dword ptr [rcx+4*rax]
       vandps   xmm0, xmm0, xmmword ptr [reloc @RWD16]
       vucomiss xmm0, dword ptr [reloc @RWD04]
       jbe      SHORT G_M000_IG11
       mov      rdi, 0x709EADF1E468
       call     CORINFO_HELP_COUNTPROFILE32
       xor      eax, eax
 
G_M000_IG10:                ;; offset=0x0181
       vzeroupper 
       add      rsp, 320
       pop      rbp
       ret      
 
G_M000_IG11:                ;; offset=0x018D
       mov      rdi, 0x709EADF1E46C
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x3C]
       inc      eax
       mov      dword ptr [rbp-0x3C], eax
 
G_M000_IG12:                ;; offset=0x01A4
       mov      eax, dword ptr [rbp-0x138]
       dec      eax
       mov      dword ptr [rbp-0x138], eax
       cmp      dword ptr [rbp-0x138], 0
       jg       SHORT G_M000_IG14
 
G_M000_IG13:                ;; offset=0x01BB
       lea      rdi, [rbp-0x138]
       mov      esi, 235
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG14:                ;; offset=0x01CC
       mov      eax, dword ptr [rbp-0x3C]
       cmp      eax, dword ptr [rbp-0x30]
       jl       G_M000_IG09
       mov      rdi, 0x709EADF1E470
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, 1
 
G_M000_IG15:                ;; offset=0x01EC
       vzeroupper 
       add      rsp, 320
       pop      rbp
       ret      
 
G_M000_IG16:                ;; offset=0x01F8
       call     CORINFO_HELP_RNGCHKFAIL
       int3     
 
RWD00  	dd	7FFFFFFFh		;       nan
RWD04  	dd	7E7FFFFFh		; 8.50706e+37
RWD08  	dd	00000000h, 00000000h
RWD16  	dq	7FFFFFFF7FFFFFFFh, 7FFFFFFF7FFFFFFFh

; Total bytes of code 510


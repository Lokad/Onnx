; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool (Tier1-OSR)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
; Tier1-OSR code
; OSR variant for entry point 0xbf
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 65
; 1 inlinees with PGO data; 0 single block inlinees; 0 inlinees without PGO data

G_M000_IG01:                ;; offset=0x0000
       mov      rax, qword ptr [rbp]
       push     rax
       mov      rbp, rsp
       mov      eax, dword ptr [rbp+0xF4]
       mov      rcx, qword ptr [rbp+0xE8]
       vmovups  ymm0, ymmword ptr [rbp+0x40]
       vmovups  ymm1, ymmword ptr [rbp+0x20]
 
G_M000_IG02:                ;; offset=0x001F
       mov      edx, dword ptr [rbp+0x100]
       lea      edi, [rdx-0x08]
       cmp      eax, edi
       jg       SHORT G_M000_IG05
       align    [4 bytes for IG03]
 
G_M000_IG03:                ;; offset=0x0030
       movsxd   rsi, eax
       vpand    ymm2, ymm0, ymmword ptr [rcx+4*rsi]
       vcmpgtps ymm2, ymm2, ymm1
       vmovmskps rsi, ymm2
       test     esi, esi
       jne      SHORT G_M000_IG11
 
G_M000_IG04:                ;; offset=0x0045
       add      eax, 8
       cmp      eax, edi
       jle      SHORT G_M000_IG03
 
G_M000_IG05:                ;; offset=0x004C
       xor      ecx, ecx
       mov      bword ptr [rbp+0xE0], rcx
       cmp      eax, edx
       jl       SHORT G_M000_IG08
 
G_M000_IG06:                ;; offset=0x0059
       mov      eax, 1
 
G_M000_IG07:                ;; offset=0x005E
       vzeroupper 
       add      rsp, 304
       pop      rbp
       ret      
 
G_M000_IG08:                ;; offset=0x006A
       test     eax, eax
       jl       SHORT G_M000_IG13
 
G_M000_IG09:                ;; offset=0x006E
       mov      rcx, bword ptr [rbp+0xF8]
       mov      edi, eax
       vmovss   xmm0, dword ptr [rcx+4*rdi]
       vandps   xmm0, xmm0, xmmword ptr [reloc @RWD00]
       vucomiss xmm0, dword ptr [reloc @RWD16]
       ja       SHORT G_M000_IG11
 
G_M000_IG10:                ;; offset=0x008E
       inc      eax
       cmp      eax, edx
       jl       SHORT G_M000_IG09
       jmp      SHORT G_M000_IG06
 
G_M000_IG11:                ;; offset=0x0096
       xor      eax, eax
 
G_M000_IG12:                ;; offset=0x0098
       vzeroupper 
       add      rsp, 304
       pop      rbp
       ret      
 
G_M000_IG13:                ;; offset=0x00A4
       cmp      eax, edx
       jae      SHORT G_M000_IG15
       mov      rcx, bword ptr [rbp+0xF8]
       mov      edi, eax
       vmovss   xmm0, dword ptr [rcx+4*rdi]
       vandps   xmm0, xmm0, xmmword ptr [reloc @RWD00]
       vucomiss xmm0, dword ptr [reloc @RWD16]
       ja       SHORT G_M000_IG11
 
G_M000_IG14:                ;; offset=0x00C8
       inc      eax
       cmp      eax, edx
       jl       SHORT G_M000_IG13
       jmp      SHORT G_M000_IG06
 
G_M000_IG15:                ;; offset=0x00D0
       call     CORINFO_HELP_RNGCHKFAIL
       int3     
 
RWD00  	dq	7FFFFFFF7FFFFFFFh, 7FFFFFFF7FFFFFFFh
RWD16  	dd	7E7FFFFFh		; 8.50706e+37

; Total bytes of code 214


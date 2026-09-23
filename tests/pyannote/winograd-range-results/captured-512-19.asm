; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool (Tier1-OSR)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Tier1-OSR code
; OSR variant for entry point 0x5d
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 101
; 1 inlinees with PGO data; 0 single block inlinees; 0 inlinees without PGO data

G_M000_IG01:                ;; offset=0x0000
       mov      rax, qword ptr [rbp]
       push     rax
       mov      rbp, rsp
       mov      eax, dword ptr [rbp+0x114]
       mov      rcx, qword ptr [rbp+0x108]
       vmovups  zmm0, zmmword ptr [rbp+0xA0]
       vmovups  zmm1, zmmword ptr [rbp+0x60]
 
G_M000_IG02:                ;; offset=0x0029
       mov      edx, dword ptr [rbp+0x120]
       lea      edi, [rdx-0x10]
       cmp      eax, edi
       jg       SHORT G_M000_IG05
       align    [0 bytes for IG03]
 
G_M000_IG03:                ;; offset=0x0036
       movsxd   rsi, eax
       vpandd   zmm2, zmm0, zmmword ptr [rcx+4*rsi]
       vcmpgtps k1, zmm2, zmm1
       kmovw    esi, k1
       test     rsi, rsi
       jne      SHORT G_M000_IG11
 
G_M000_IG04:                ;; offset=0x0050
       add      eax, 16
       cmp      eax, edi
       jle      SHORT G_M000_IG03
 
G_M000_IG05:                ;; offset=0x0057
       xor      ecx, ecx
       mov      bword ptr [rbp+0x100], rcx
       cmp      eax, edx
       jl       SHORT G_M000_IG08
 
G_M000_IG06:                ;; offset=0x0064
       mov      eax, 1
 
G_M000_IG07:                ;; offset=0x0069
       vzeroupper 
       add      rsp, 336
       pop      rbp
       ret      
 
G_M000_IG08:                ;; offset=0x0075
       test     eax, eax
       jl       SHORT G_M000_IG13
 
G_M000_IG09:                ;; offset=0x0079
       mov      rcx, bword ptr [rbp+0x118]
       mov      edi, eax
       vmovss   xmm0, dword ptr [rcx+4*rdi]
       vandps   xmm0, xmm0, xmmword ptr [reloc @RWD00]
       vucomiss xmm0, dword ptr [reloc @RWD16]
       ja       SHORT G_M000_IG11
 
G_M000_IG10:                ;; offset=0x0099
       inc      eax
       cmp      eax, edx
       jl       SHORT G_M000_IG09
       jmp      SHORT G_M000_IG06
 
G_M000_IG11:                ;; offset=0x00A1
       xor      eax, eax
 
G_M000_IG12:                ;; offset=0x00A3
       vzeroupper 
       add      rsp, 336
       pop      rbp
       ret      
 
G_M000_IG13:                ;; offset=0x00AF
       cmp      eax, edx
       jae      SHORT G_M000_IG15
       mov      rcx, bword ptr [rbp+0x118]
       mov      edi, eax
       vmovss   xmm0, dword ptr [rcx+4*rdi]
       vandps   xmm0, xmm0, xmmword ptr [reloc @RWD00]
       vucomiss xmm0, dword ptr [reloc @RWD16]
       ja       SHORT G_M000_IG11
 
G_M000_IG14:                ;; offset=0x00D3
       inc      eax
       cmp      eax, edx
       jl       SHORT G_M000_IG13
       jmp      SHORT G_M000_IG06
 
G_M000_IG15:                ;; offset=0x00DB
       call     CORINFO_HELP_RNGCHKFAIL
       int3     
 
RWD00  	dq	7FFFFFFF7FFFFFFFh, 7FFFFFFF7FFFFFFFh
RWD16  	dd	7E7FFFFFh		; 8.50706e+37

; Total bytes of code 225


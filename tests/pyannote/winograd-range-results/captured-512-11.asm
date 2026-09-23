; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool (Tier1-OSR)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Tier1-OSR code
; OSR variant for entry point 0x5d
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 1
; 0 inlinees with PGO data; 0 single block inlinees; 1 inlinees without PGO data

G_M000_IG01:                ;; offset=0x0000
       mov      rax, qword ptr [rbp]
       push     rax
       mov      rbp, rsp
       mov      eax, dword ptr [rbp+0x114]
       mov      rcx, qword ptr [rbp+0x108]
       vmovups  zmm0, zmmword ptr [rbp+0xA0]
       vmovups  zmm1, zmmword ptr [rbp+0x60]
 
G_M000_IG02:                ;; offset=0x0029
       mov      rdx, bword ptr [rbp+0x118]
       mov      edi, dword ptr [rbp+0x120]
       lea      esi, [rdi-0x10]
       cmp      eax, esi
       jg       SHORT G_M000_IG05
       align    [0 bytes for IG03]
 
G_M000_IG03:                ;; offset=0x003D
       movsxd   r8, eax
       vpandd   zmm2, zmm0, zmmword ptr [rcx+4*r8]
       vcmpgtps k1, zmm2, zmm1
       kmovw    r8d, k1
       test     r8, r8
       jne      SHORT G_M000_IG12
 
G_M000_IG04:                ;; offset=0x0057
       add      eax, 16
       cmp      eax, esi
       jle      SHORT G_M000_IG03
 
G_M000_IG05:                ;; offset=0x005E
       xor      ecx, ecx
       mov      bword ptr [rbp+0x100], rcx
       cmp      eax, edi
       jge      SHORT G_M000_IG10
 
G_M000_IG06:                ;; offset=0x006B
       test     eax, eax
       jl       SHORT G_M000_IG14
 
G_M000_IG07:                ;; offset=0x006F
       vmovss   xmm0, dword ptr [reloc @RWD00]
 
G_M000_IG08:                ;; offset=0x0077
       mov      ecx, eax
       vmovss   xmm1, dword ptr [rdx+4*rcx]
       vandps   xmm1, xmm1, xmmword ptr [reloc @RWD16]
       vucomiss xmm1, xmm0
       ja       SHORT G_M000_IG12
 
G_M000_IG09:                ;; offset=0x008C
       inc      eax
       cmp      eax, edi
       jl       SHORT G_M000_IG08
 
G_M000_IG10:                ;; offset=0x0092
       mov      eax, 1
 
G_M000_IG11:                ;; offset=0x0097
       vzeroupper 
       add      rsp, 336
       pop      rbp
       ret      
 
G_M000_IG12:                ;; offset=0x00A3
       xor      eax, eax
 
G_M000_IG13:                ;; offset=0x00A5
       vzeroupper 
       add      rsp, 336
       pop      rbp
       ret      
 
G_M000_IG14:                ;; offset=0x00B1
       cmp      eax, edi
       jae      SHORT G_M000_IG16
       mov      ecx, eax
       vmovss   xmm1, dword ptr [rdx+4*rcx]
       vandps   xmm1, xmm1, xmmword ptr [reloc @RWD16]
       vmovss   xmm0, dword ptr [reloc @RWD00]
       vucomiss xmm1, xmm0
       ja       SHORT G_M000_IG12
 
G_M000_IG15:                ;; offset=0x00D2
       inc      eax
       cmp      eax, edi
       jl       SHORT G_M000_IG14
       jmp      SHORT G_M000_IG10
 
G_M000_IG16:                ;; offset=0x00DA
       call     CORINFO_HELP_RNGCHKFAIL
       int3     
 
RWD00  	dd	7E7FFFFFh		; 8.50706e+37
RWD04  	dd	00000000h, 00000000h, 00000000h
RWD16  	dq	7FFFFFFF7FFFFFFFh, 7FFFFFFF7FFFFFFFh

; Total bytes of code 224


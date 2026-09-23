; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool (Tier1)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
; Tier1 code
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 216
; 1 inlinees with PGO data; 0 single block inlinees; 0 inlinees without PGO data

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       sub      rsp, 16
       lea      rbp, [rsp+0x10]
       xor      eax, eax
       mov      qword ptr [rbp-0x08], rax
 
G_M000_IG02:                ;; offset=0x0010
       xor      eax, eax
       xor      rcx, rcx
       test     esi, esi
       cmovne   rcx, rdi
       mov      bword ptr [rbp-0x08], rcx
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
       vbroadcastss ymm1, dword ptr [reloc @RWD04]
       lea      edx, [rsi-0x08]
       test     edx, edx
       jl       SHORT G_M000_IG05
       align    [9 bytes for IG03]
 
G_M000_IG03:                ;; offset=0x0040
       movsxd   r8, eax
       vpand    ymm2, ymm0, ymmword ptr [rcx+4*r8]
       vcmpgtps ymm2, ymm2, ymm1
       vmovmskps r8, ymm2
       test     r8d, r8d
       jne      SHORT G_M000_IG11
 
G_M000_IG04:                ;; offset=0x0057
       add      eax, 8
       cmp      eax, edx
       jle      SHORT G_M000_IG03
 
G_M000_IG05:                ;; offset=0x005E
       xor      ecx, ecx
       mov      bword ptr [rbp-0x08], rcx
       cmp      eax, esi
       jl       SHORT G_M000_IG08
 
G_M000_IG06:                ;; offset=0x0068
       mov      eax, 1
 
G_M000_IG07:                ;; offset=0x006D
       vzeroupper 
       add      rsp, 16
       pop      rbp
       ret      
 
G_M000_IG08:                ;; offset=0x0076
       test     eax, eax
       jl       SHORT G_M000_IG13
 
G_M000_IG09:                ;; offset=0x007A
       mov      ecx, eax
       vmovss   xmm0, dword ptr [rdi+4*rcx]
       vandps   xmm0, xmm0, xmmword ptr [reloc @RWD16]
       vucomiss xmm0, dword ptr [reloc @RWD04]
       ja       SHORT G_M000_IG11
 
G_M000_IG10:                ;; offset=0x0093
       inc      eax
       cmp      eax, esi
       jl       SHORT G_M000_IG09
       jmp      SHORT G_M000_IG06
 
G_M000_IG11:                ;; offset=0x009B
       xor      eax, eax
 
G_M000_IG12:                ;; offset=0x009D
       vzeroupper 
       add      rsp, 16
       pop      rbp
       ret      
 
G_M000_IG13:                ;; offset=0x00A6
       cmp      eax, esi
       jae      SHORT G_M000_IG15
       mov      ecx, eax
       vmovss   xmm0, dword ptr [rdi+4*rcx]
       vandps   xmm0, xmm0, xmmword ptr [reloc @RWD16]
       vucomiss xmm0, dword ptr [reloc @RWD04]
       ja       SHORT G_M000_IG11
 
G_M000_IG14:                ;; offset=0x00C3
       inc      eax
       cmp      eax, esi
       jl       SHORT G_M000_IG13
       jmp      SHORT G_M000_IG06
 
G_M000_IG15:                ;; offset=0x00CB
       call     CORINFO_HELP_RNGCHKFAIL
       int3     
 
RWD00  	dd	7FFFFFFFh		;       nan
RWD04  	dd	7E7FFFFFh		; 8.50706e+37
RWD08  	dd	00000000h, 00000000h
RWD16  	dq	7FFFFFFF7FFFFFFFh, 7FFFFFFF7FFFFFFFh

; Total bytes of code 209


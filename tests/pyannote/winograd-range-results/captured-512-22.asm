; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool (Tier1)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Tier1 code
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 248
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
       vbroadcastss zmm0, dword ptr [reloc @RWD00]
       vbroadcastss zmm1, dword ptr [reloc @RWD04]
       lea      edx, [rsi-0x10]
       test     edx, edx
       jl       SHORT G_M000_IG05
       align    [0 bytes for IG03]
 
G_M000_IG03:                ;; offset=0x0039
       movsxd   r8, eax
       vpandd   zmm2, zmm0, zmmword ptr [rcx+4*r8]
       vcmpgtps k1, zmm2, zmm1
       kmovw    r8d, k1
       test     r8, r8
       jne      SHORT G_M000_IG11
 
G_M000_IG04:                ;; offset=0x0053
       add      eax, 16
       cmp      eax, edx
       jle      SHORT G_M000_IG03
 
G_M000_IG05:                ;; offset=0x005A
       xor      ecx, ecx
       mov      bword ptr [rbp-0x08], rcx
       cmp      eax, esi
       jl       SHORT G_M000_IG08
 
G_M000_IG06:                ;; offset=0x0064
       mov      eax, 1
 
G_M000_IG07:                ;; offset=0x0069
       vzeroupper 
       add      rsp, 16
       pop      rbp
       ret      
 
G_M000_IG08:                ;; offset=0x0072
       test     eax, eax
       jl       SHORT G_M000_IG13
 
G_M000_IG09:                ;; offset=0x0076
       mov      ecx, eax
       vmovss   xmm0, dword ptr [rdi+4*rcx]
       vandps   xmm0, xmm0, xmmword ptr [reloc @RWD16]
       vucomiss xmm0, dword ptr [reloc @RWD04]
       ja       SHORT G_M000_IG11
 
G_M000_IG10:                ;; offset=0x008F
       inc      eax
       cmp      eax, esi
       jl       SHORT G_M000_IG09
       jmp      SHORT G_M000_IG06
 
G_M000_IG11:                ;; offset=0x0097
       xor      eax, eax
 
G_M000_IG12:                ;; offset=0x0099
       vzeroupper 
       add      rsp, 16
       pop      rbp
       ret      
 
G_M000_IG13:                ;; offset=0x00A2
       cmp      eax, esi
       jae      SHORT G_M000_IG15
       mov      ecx, eax
       vmovss   xmm0, dword ptr [rdi+4*rcx]
       vandps   xmm0, xmm0, xmmword ptr [reloc @RWD16]
       vucomiss xmm0, dword ptr [reloc @RWD04]
       ja       SHORT G_M000_IG11
 
G_M000_IG14:                ;; offset=0x00BF
       inc      eax
       cmp      eax, esi
       jl       SHORT G_M000_IG13
       jmp      SHORT G_M000_IG06
 
G_M000_IG15:                ;; offset=0x00C7
       call     CORINFO_HELP_RNGCHKFAIL
       int3     
 
RWD00  	dd	7FFFFFFFh		;       nan
RWD04  	dd	7E7FFFFFh		; 8.50706e+37
RWD08  	dd	00000000h, 00000000h
RWD16  	dq	7FFFFFFF7FFFFFFFh, 7FFFFFFF7FFFFFFFh

; Total bytes of code 205


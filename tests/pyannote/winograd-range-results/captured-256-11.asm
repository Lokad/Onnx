; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool (Tier1-OSR)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
; Tier1-OSR code
; OSR variant for entry point 0xbf
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
       mov      eax, dword ptr [rbp+0xF4]
       mov      rcx, qword ptr [rbp+0xE8]
       vmovups  ymm0, ymmword ptr [rbp+0x40]
       vmovups  ymm1, ymmword ptr [rbp+0x20]
 
G_M000_IG02:                ;; offset=0x001F
       mov      rdx, bword ptr [rbp+0xF8]
       mov      edi, dword ptr [rbp+0x100]
       lea      esi, [rdi-0x08]
       cmp      eax, esi
       jg       SHORT G_M000_IG05
       align    [13 bytes for IG03]
 
G_M000_IG03:                ;; offset=0x0040
       movsxd   r8, eax
       vpand    ymm2, ymm0, ymmword ptr [rcx+4*r8]
       vcmpgtps ymm2, ymm2, ymm1
       vmovmskps r8, ymm2
       test     r8d, r8d
       jne      SHORT G_M000_IG12
 
G_M000_IG04:                ;; offset=0x0057
       add      eax, 8
       cmp      eax, esi
       jle      SHORT G_M000_IG03
 
G_M000_IG05:                ;; offset=0x005E
       xor      ecx, ecx
       mov      bword ptr [rbp+0xE0], rcx
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
       add      rsp, 304
       pop      rbp
       ret      
 
G_M000_IG12:                ;; offset=0x00A3
       xor      eax, eax
 
G_M000_IG13:                ;; offset=0x00A5
       vzeroupper 
       add      rsp, 304
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


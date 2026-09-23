; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:PrepareWinograd(System.ReadOnlySpan`1[float],int,int,int):float[] (Tier0-FullOpts)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
; Tier-0 switched to FullOpts code
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 100
; No PGO data
; 0 inlinees with PGO data; 2 single block inlinees; 3 inlinees without PGO data

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       push     r15
       push     r14
       push     r13
       push     r12
       push     rbx
       sub      rsp, 24
       lea      rbp, [rsp+0x40]
       mov      qword ptr [rbp-0x38], 0x1DF9575E
       mov      r13, rdi
       mov      r14d, esi
       mov      ebx, edx
       mov      r15d, ecx
 
G_M000_IG02:                ;; offset=0x0026
       mov      edi, ebx
       mov      esi, r15d
       mov      r9d, r8d
       mov      edx, 1
       mov      ecx, 1
       mov      r8d, 1
       call     [Lokad.Onnx.ConvBlockedSpatial:Geometry(int,int,int,int,int,int)]
       mov      edi, ebx
       imul     edi, r15d
       jo       G_M000_IG21
       imul     edi, edi, 9
       jo       G_M000_IG21
       cmp      edi, r14d
       jne      G_M000_IG18
       mov      rdi, r13
       mov      esi, r14d
       call     [Lokad.Onnx.ConvBlockedSpatial:Finite(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       G_M000_IG12
 
G_M000_IG03:                ;; offset=0x0076
       imul     r12d, ebx, 16
       jo       G_M000_IG21
       imul     r12d, r15d
       jo       G_M000_IG21
       movsxd   rsi, r12d
       mov      rdi, 0x761AD7751490
       call     CORINFO_HELP_NEWARR_1_VC
       mov      gword ptr [rbp-0x30], rax
       test     dword ptr [rsp], esp
       sub      rsp, 48
       lea      rcx, [rsp]
       vxorps   ymm0, ymm0, ymm0
       vmovdqu  ymmword ptr [rcx], ymm0
       vmovdqu  xmmword ptr [rcx+0x20], xmm0
       xor      edx, edx
       vmovss   xmm0, dword ptr [reloc @RWD00]
       cmp      edx, r15d
       jge      G_M000_IG11
 
G_M000_IG04:                ;; offset=0x00CB
       xor      edi, edi
       cmp      edi, ebx
       jge      G_M000_IG10
 
G_M000_IG05:                ;; offset=0x00D5
       mov      esi, edx
       imul     esi, ebx
       add      esi, edi
       lea      esi, [rsi+8*rsi]
       mov      r8d, esi
       add      r8, 9
       mov      r9d, r14d
       cmp      r8, r9
       ja       G_M000_IG19
       lea      rsi, bword ptr [r13+4*rsi]
       xor      r8d, r8d
       align    [0 bytes for IG06]
 
G_M000_IG06:                ;; offset=0x00FA
       lea      r9d, [r8+2*r8]
       mov      r10d, r9d
       vmovss   xmm1, dword ptr [rsi+4*r10]
       lea      r10d, [r9+0x01]
       vmovss   xmm2, dword ptr [rsi+4*r10]
       add      r9d, 2
       vmovss   xmm3, dword ptr [rsi+4*r9]
       lea      r9d, [4*r8]
       mov      r10d, r9d
       vmovss   dword ptr [rcx+4*r10], xmm1
       lea      r10d, [r9+0x01]
       vaddss   xmm4, xmm1, xmm2
       vaddss   xmm4, xmm4, xmm3
       vmulss   xmm4, xmm4, xmm0
       vmovss   dword ptr [rcx+4*r10], xmm4
       lea      r10d, [r9+0x02]
       vsubss   xmm1, xmm1, xmm2
       vaddss   xmm1, xmm1, xmm3
       vmulss   xmm1, xmm1, xmm0
       vmovss   dword ptr [rcx+4*r10], xmm1
       add      r9d, 3
       vmovss   dword ptr [rcx+4*r9], xmm3
       inc      r8d
       cmp      r8d, 3
       jl       SHORT G_M000_IG06
 
G_M000_IG07:                ;; offset=0x016B
       xor      esi, esi
       align    [0 bytes for IG08]
 
G_M000_IG08:                ;; offset=0x016D
       vmovss   xmm1, dword ptr [rcx+4*rsi]
       lea      r8d, [rsi+0x04]
       mov      r9d, r8d
       vmovss   xmm2, dword ptr [rcx+4*r9]
       lea      r9d, [rsi+0x08]
       mov      r10d, r9d
       vmovss   xmm3, dword ptr [rcx+4*r10]
       mov      r10d, esi
       imul     r10d, ebx
       add      r10d, edi
       imul     r10d, r15d
       add      r10d, edx
       cmp      r10d, r12d
       jae      G_M000_IG20
       vmovss   dword ptr [rax+4*r10+0x10], xmm1
       imul     r8d, ebx
       add      r8d, edi
       imul     r8d, r15d
       add      r8d, edx
       cmp      r8d, r12d
       jae      G_M000_IG20
       vaddss   xmm4, xmm1, xmm2
       vaddss   xmm4, xmm4, xmm3
       vmulss   xmm4, xmm4, xmm0
       vmovss   dword ptr [rax+4*r8+0x10], xmm4
       imul     r9d, ebx
       add      r9d, edi
       imul     r9d, r15d
       add      r9d, edx
       cmp      r9d, r12d
       jae      G_M000_IG20
       mov      r8d, r9d
       vsubss   xmm1, xmm1, xmm2
       vaddss   xmm1, xmm1, xmm3
       vmulss   xmm1, xmm1, xmm0
       vmovss   dword ptr [rax+4*r8+0x10], xmm1
       lea      r8d, [rsi+0x0C]
       imul     r8d, ebx
       add      r8d, edi
       imul     r8d, r15d
       add      r8d, edx
       cmp      r8d, r12d
       jae      G_M000_IG20
       vmovss   dword ptr [rax+4*r8+0x10], xmm3
       inc      esi
       cmp      esi, 4
       jl       G_M000_IG08
 
G_M000_IG09:                ;; offset=0x0231
       inc      edi
       cmp      edi, ebx
       jl       G_M000_IG05
 
G_M000_IG10:                ;; offset=0x023B
       inc      edx
       cmp      edx, r15d
       jl       G_M000_IG04
 
G_M000_IG11:                ;; offset=0x0246
       lea      rdi, bword ptr [rax+0x10]
       mov      esi, r12d
       call     [Lokad.Onnx.ConvBlockedSpatial:Finite(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      SHORT G_M000_IG15
 
G_M000_IG12:                ;; offset=0x0257
       xor      rax, rax
       cmp      qword ptr [rbp-0x38], 0x1DF9575E
       je       SHORT G_M000_IG13
       call     CORINFO_HELP_FAIL_FAST
 
G_M000_IG13:                ;; offset=0x0268
       nop      
 
G_M000_IG14:                ;; offset=0x0269
       lea      rsp, [rbp-0x28]
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
G_M000_IG15:                ;; offset=0x0278
       mov      rax, gword ptr [rbp-0x30]
       cmp      qword ptr [rbp-0x38], 0x1DF9575E
       je       SHORT G_M000_IG16
       call     CORINFO_HELP_FAIL_FAST
 
G_M000_IG16:                ;; offset=0x028B
       nop      
 
G_M000_IG17:                ;; offset=0x028C
       lea      rsp, [rbp-0x28]
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
G_M000_IG18:                ;; offset=0x029B
       mov      rdi, 0x761AD707CD28
       call     CORINFO_HELP_NEWSFAST
       mov      rbx, rax
       mov      edi, 0xCC8
       mov      rsi, 0x761AD7131A30
       call     [CORINFO_HELP_STRCNS]
       mov      rsi, rax
       mov      rdi, rbx
       call     [System.ArgumentException:.ctor(System.String):this]
       mov      rdi, rbx
       call     CORINFO_HELP_THROW
       int3     
 
G_M000_IG19:                ;; offset=0x02D7
       call     [System.ThrowHelper:ThrowArgumentOutOfRangeException()]
       int3     
 
G_M000_IG20:                ;; offset=0x02DE
       call     CORINFO_HELP_RNGCHKFAIL
       int3     
 
G_M000_IG21:                ;; offset=0x02E4
       call     CORINFO_HELP_OVERFLOW
       int3     
 
RWD00  	dd	3F000000h		;       0.5

; Total bytes of code 746


; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:TransformWinogradInputContiguous(System.ReadOnlySpan`1[float],System.Span`1[float],int,int,int,int,int) (Tier0-FullOpts)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Tier-0 switched to FullOpts code
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 180
; No PGO data
; 0 inlinees with PGO data; 0 single block inlinees; 2 inlinees without PGO data

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       push     r15
       push     r14
       push     r13
       push     r12
       push     rbx
       sub      rsp, 24
       lea      rbp, [rsp+0x40]
       xor      eax, eax
       mov      qword ptr [rbp-0x30], rax
       mov      qword ptr [rbp-0x38], rax
       mov      qword ptr [rbp-0x40], 0x1DE67DC0
       mov      eax, dword ptr [rbp+0x10]
       mov      r10d, dword ptr [rbp+0x18]
       mov      r11d, dword ptr [rbp+0x20]
 
G_M000_IG02:                ;; offset=0x0030
       vmovups  ymm0, ymmword ptr [reloc @RWD00]
       test     dword ptr [rsp], esp
       sub      rsp, 512
       lea      rbx, [rsp]
       vxorps   ymm1, ymm1, ymm1
       vmovdqu32 zmmword ptr [rbx], zmm1
       vmovdqu32 zmmword ptr [rbx+0x40], zmm1
       vmovdqu32 zmmword ptr [rbx+0x80], zmm1
       vmovdqu32 zmmword ptr [rbx+0xC0], zmm1
       vmovdqu32 zmmword ptr [rbx+0x100], zmm1
       vmovdqu32 zmmword ptr [rbx+0x140], zmm1
       vmovdqu32 zmmword ptr [rbx+0x180], zmm1
       vmovdqu32 zmmword ptr [rbx+0x1C0], zmm1
       xor      r15, r15
       test     esi, esi
       cmovne   r15, rdi
       mov      bword ptr [rbp-0x30], r15
       mov      rdi, r15
       xor      rsi, rsi
       test     ecx, ecx
       cmovne   rsi, rdx
       mov      bword ptr [rbp-0x38], rsi
       mov      rcx, rsi
       xor      edx, edx
       cmp      edx, r8d
       jge      G_M000_IG08
 
G_M000_IG03:                ;; offset=0x00AB
       mov      esi, edx
       imul     esi, r9d
       imul     esi, eax
       movsxd   rsi, esi
       lea      rsi, [rdi+4*rsi]
       xor      r15d, r15d
       movsxd   r14, r11d
       shl      r14, 2
       align    [0 bytes for IG04]
 
G_M000_IG04:                ;; offset=0x00C5
       lea      r13d, [r10+r15]
       imul     r13d, eax
       movsxd   r13, r13d
       shl      r13, 2
       add      r13, rsi
       add      r13, r14
       vpermps  ymm1, ymm0, ymmword ptr [r13]
       vpermps  ymm2, ymm0, ymmword ptr [r13+0x20]
       vpermps  ymm3, ymm0, ymmword ptr [r13+0x08]
       vpermps  ymm4, ymm0, ymmword ptr [r13+0x28]
       vperm2f128 ymm5, ymm1, ymm2, 32
       vperm2f128 ymm1, ymm1, ymm2, 49
       vperm2f128 ymm2, ymm3, ymm4, 32
       vperm2f128 ymm3, ymm3, ymm4, 49
       vsubps   ymm4, ymm5, ymm2
       lea      r13d, [4*r15]
       movsxd   r12, r13d
       shl      r12, 5
       vmovups  ymmword ptr [rbx+r12], ymm4
       vaddps   ymm4, ymm1, ymm2
       lea      r12d, [r13+0x01]
       movsxd   r12, r12d
       shl      r12, 5
       vmovups  ymmword ptr [rbx+r12], ymm4
       vsubps   ymm2, ymm2, ymm1
       lea      r12d, [r13+0x02]
       movsxd   r12, r12d
       shl      r12, 5
       vmovups  ymmword ptr [rbx+r12], ymm2
       vsubps   ymm1, ymm1, ymm3
       add      r13d, 3
       movsxd   r13, r13d
       shl      r13, 5
       vmovups  ymmword ptr [rbx+r13], ymm1
       inc      r15d
       cmp      r15d, 4
       jl       G_M000_IG04
 
G_M000_IG05:                ;; offset=0x016F
       xor      esi, esi
       align    [0 bytes for IG06]
 
G_M000_IG06:                ;; offset=0x0171
       movsxd   r15, esi
       shl      r15, 5
       vmovups  ymm1, ymmword ptr [rbx+r15]
       lea      r15d, [rsi+0x04]
       movsxd   r14, r15d
       shl      r14, 5
       vmovups  ymm2, ymmword ptr [rbx+r14]
       lea      r14d, [rsi+0x08]
       movsxd   r13, r14d
       shl      r13, 5
       vmovups  ymm3, ymmword ptr [rbx+r13]
       lea      r13d, [rsi+0x0C]
       movsxd   r12, r13d
       shl      r12, 5
       vmovups  ymm4, ymmword ptr [rbx+r12]
       vsubps   ymm1, ymm1, ymm3
       mov      r12d, esi
       imul     r12d, r8d
       add      r12d, edx
       shl      r12d, 3
       movsxd   r12, r12d
       vmovups  ymmword ptr [rcx+4*r12], ymm1
       vaddps   ymm1, ymm2, ymm3
       imul     r15d, r8d
       add      r15d, edx
       shl      r15d, 3
       movsxd   r15, r15d
       vmovups  ymmword ptr [rcx+4*r15], ymm1
       vsubps   ymm1, ymm3, ymm2
       imul     r14d, r8d
       add      r14d, edx
       shl      r14d, 3
       movsxd   r15, r14d
       vmovups  ymmword ptr [rcx+4*r15], ymm1
       vsubps   ymm1, ymm2, ymm4
       imul     r13d, r8d
       add      r13d, edx
       shl      r13d, 3
       movsxd   r15, r13d
       vmovups  ymmword ptr [rcx+4*r15], ymm1
       inc      esi
       cmp      esi, 4
       jl       G_M000_IG06
 
G_M000_IG07:                ;; offset=0x021F
       inc      edx
       cmp      edx, r8d
       jl       G_M000_IG03
 
G_M000_IG08:                ;; offset=0x022A
       xor      eax, eax
       mov      bword ptr [rbp-0x30], rax
 
G_M000_IG09:                ;; offset=0x0230
       mov      bword ptr [rbp-0x38], rax
       cmp      qword ptr [rbp-0x40], 0x1DE67DC0
       je       SHORT G_M000_IG10
       call     CORINFO_HELP_FAIL_FAST
 
G_M000_IG10:                ;; offset=0x0243
       nop      
 
G_M000_IG11:                ;; offset=0x0244
       vzeroupper 
       lea      rsp, [rbp-0x28]
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
RWD00  	dq	0000000200000000h, 0000000600000004h, 0000000300000001h, 0000000700000005h

; Total bytes of code 598


; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:TransformWinogradInputContiguous(System.ReadOnlySpan`1[float],System.Span`1[float],int,int,int,int,int) (Tier0-FullOpts)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
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
       sub      rsp, 104
       lea      rbp, [rsp+0x90]
       xor      eax, eax
       mov      qword ptr [rbp-0x58], rax
       mov      qword ptr [rbp-0x60], rax
       mov      qword ptr [rbp-0x88], 0x1EE88627
       mov      bword ptr [rbp-0x78], rdi
       mov      dword ptr [rbp-0x64], esi
       mov      bword ptr [rbp-0x80], rdx
       mov      dword ptr [rbp-0x68], ecx
       mov      ebx, r8d
       mov      r13d, r9d
       mov      r15d, dword ptr [rbp+0x10]
       mov      r14d, dword ptr [rbp+0x18]
       mov      r12d, dword ptr [rbp+0x20]
 
G_M000_IG02:                ;; offset=0x004B
       vmovups  ymm0, ymmword ptr [reloc @RWD00]
       vmovups  ymmword ptr [rbp-0x50], ymm0
       test     dword ptr [rsp], esp
       sub      rsp, 512
       lea      rdi, [rsp]
       mov      qword ptr [rbp-0x70], rdi
       mov      esi, 512
       call     [CORINFO_HELP_MEMZERO]
       mov      rdi, qword ptr [rbp-0x70]
       mov      rax, rdi
       xor      rcx, rcx
       cmp      dword ptr [rbp-0x64], 0
       cmovne   rcx, bword ptr [rbp-0x78]
       mov      bword ptr [rbp-0x58], rcx
       xor      rdx, rdx
       cmp      dword ptr [rbp-0x68], 0
       cmovne   rdx, bword ptr [rbp-0x80]
       mov      bword ptr [rbp-0x60], rdx
       xor      edi, edi
       cmp      edi, ebx
       jge      G_M000_IG08
 
G_M000_IG03:                ;; offset=0x00A4
       mov      esi, edi
       imul     esi, r13d
       imul     esi, r15d
       movsxd   rsi, esi
       lea      rsi, [rcx+4*rsi]
       xor      r8d, r8d
       movsxd   r9, r12d
       shl      r9, 2
       align    [0 bytes for IG04]
 
G_M000_IG04:                ;; offset=0x00BF
       lea      r10d, [r14+r8]
       imul     r10d, r15d
       movsxd   r10, r10d
       shl      r10, 2
       add      r10, rsi
       add      r10, r9
       vmovups  ymm0, ymmword ptr [rbp-0x50]
       vpermps  ymm1, ymm0, ymmword ptr [r10]
       vpermps  ymm2, ymm0, ymmword ptr [r10+0x20]
       vpermps  ymm3, ymm0, ymmword ptr [r10+0x08]
       vpermps  ymm4, ymm0, ymmword ptr [r10+0x28]
       vperm2f128 ymm5, ymm1, ymm2, 32
       vperm2f128 ymm1, ymm1, ymm2, 49
       vperm2f128 ymm2, ymm3, ymm4, 32
       vperm2f128 ymm3, ymm3, ymm4, 49
       vsubps   ymm4, ymm5, ymm2
       lea      r10d, [4*r8]
       movsxd   r11, r10d
       shl      r11, 5
       vmovups  ymmword ptr [rax+r11], ymm4
       vaddps   ymm4, ymm1, ymm2
       lea      r11d, [r10+0x01]
       movsxd   r11, r11d
       shl      r11, 5
       vmovups  ymmword ptr [rax+r11], ymm4
       vsubps   ymm2, ymm2, ymm1
       lea      r11d, [r10+0x02]
       movsxd   r11, r11d
       shl      r11, 5
       vmovups  ymmword ptr [rax+r11], ymm2
       vsubps   ymm1, ymm1, ymm3
       add      r10d, 3
       movsxd   r10, r10d
       shl      r10, 5
       vmovups  ymmword ptr [rax+r10], ymm1
       inc      r8d
       cmp      r8d, 4
       jl       G_M000_IG04
 
G_M000_IG05:                ;; offset=0x016D
       xor      esi, esi
       align    [0 bytes for IG06]
 
G_M000_IG06:                ;; offset=0x016F
       movsxd   r8, esi
       shl      r8, 5
       vmovups  ymm1, ymmword ptr [rax+r8]
       lea      r8d, [rsi+0x04]
       movsxd   r9, r8d
       shl      r9, 5
       vmovups  ymm2, ymmword ptr [rax+r9]
       lea      r9d, [rsi+0x08]
       movsxd   r10, r9d
       shl      r10, 5
       vmovups  ymm3, ymmword ptr [rax+r10]
       lea      r10d, [rsi+0x0C]
       movsxd   r11, r10d
       shl      r11, 5
       vmovups  ymm4, ymmword ptr [rax+r11]
       vsubps   ymm1, ymm1, ymm3
       mov      r11d, esi
       imul     r11d, ebx
       add      r11d, edi
       shl      r11d, 3
       movsxd   r11, r11d
       vmovups  ymmword ptr [rdx+4*r11], ymm1
       vaddps   ymm1, ymm2, ymm3
       imul     r8d, ebx
       add      r8d, edi
       shl      r8d, 3
       movsxd   r8, r8d
       vmovups  ymmword ptr [rdx+4*r8], ymm1
       vsubps   ymm1, ymm3, ymm2
       imul     r9d, ebx
       add      r9d, edi
       shl      r9d, 3
       movsxd   r8, r9d
       vmovups  ymmword ptr [rdx+4*r8], ymm1
       vsubps   ymm1, ymm2, ymm4
       imul     r10d, ebx
       add      r10d, edi
       shl      r10d, 3
       movsxd   r8, r10d
       vmovups  ymmword ptr [rdx+4*r8], ymm1
       inc      esi
       cmp      esi, 4
       jl       G_M000_IG06
 
G_M000_IG07:                ;; offset=0x021D
       inc      edi
       cmp      edi, ebx
       jl       G_M000_IG03
 
G_M000_IG08:                ;; offset=0x0227
       xor      eax, eax
       mov      bword ptr [rbp-0x58], rax
 
G_M000_IG09:                ;; offset=0x022D
       mov      bword ptr [rbp-0x60], rax
       cmp      qword ptr [rbp-0x88], 0x1EE88627
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


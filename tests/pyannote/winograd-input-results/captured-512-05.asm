; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:TransformWinogradInput(System.ReadOnlySpan`1[float],System.Span`1[float],int,int,int,int,int,int) (Tier0-FullOpts)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Tier-0 switched to FullOpts code
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 100
; No PGO data
; 0 inlinees with PGO data; 0 single block inlinees; 2 inlinees without PGO data

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       push     r15
       push     r14
       push     r13
       push     r12
       push     rbx
       sub      rsp, 72
       lea      rbp, [rsp+0x70]
       xor      eax, eax
       mov      qword ptr [rbp-0x48], rax
       mov      qword ptr [rbp-0x50], rax
       mov      qword ptr [rbp-0x70], 0x1DD1ABC5
       mov      bword ptr [rbp-0x60], rdi
       mov      dword ptr [rbp-0x54], esi
       mov      bword ptr [rbp-0x68], rdx
       mov      dword ptr [rbp-0x58], ecx
       mov      dword ptr [rbp-0x2C], r8d
       mov      r11d, dword ptr [rbp+0x10]
       mov      ebx, dword ptr [rbp+0x18]
       mov      r15d, dword ptr [rbp+0x20]
       mov      r10d, dword ptr [rbp+0x28]
 
G_M000_IG02:                ;; offset=0x0046
       test     dword ptr [rsp], esp
       sub      rsp, 512
       lea      rax, [rsp]
       vxorps   ymm0, ymm0, ymm0
       vmovdqu32 zmmword ptr [rax], zmm0
       vmovdqu32 zmmword ptr [rax+0x40], zmm0
       vmovdqu32 zmmword ptr [rax+0x80], zmm0
       vmovdqu32 zmmword ptr [rax+0xC0], zmm0
       vmovdqu32 zmmword ptr [rax+0x100], zmm0
       vmovdqu32 zmmword ptr [rax+0x140], zmm0
       vmovdqu32 zmmword ptr [rax+0x180], zmm0
       vmovdqu32 zmmword ptr [rax+0x1C0], zmm0
       mov      r13, rax
       test     dword ptr [rsp], esp
       sub      rsp, 512
       lea      rax, [rsp]
       vxorps   ymm0, ymm0, ymm0
       vmovdqu32 zmmword ptr [rax], zmm0
       vmovdqu32 zmmword ptr [rax+0x40], zmm0
       vmovdqu32 zmmword ptr [rax+0x80], zmm0
       vmovdqu32 zmmword ptr [rax+0xC0], zmm0
       vmovdqu32 zmmword ptr [rax+0x100], zmm0
       vmovdqu32 zmmword ptr [rax+0x140], zmm0
       vmovdqu32 zmmword ptr [rax+0x180], zmm0
       vmovdqu32 zmmword ptr [rax+0x1C0], zmm0
       mov      r12, rax
       test     dword ptr [rsp], esp
       sub      rsp, 512
       lea      rax, [rsp]
       vxorps   ymm0, ymm0, ymm0
       vmovdqu32 zmmword ptr [rax], zmm0
       vmovdqu32 zmmword ptr [rax+0x40], zmm0
       vmovdqu32 zmmword ptr [rax+0x80], zmm0
       vmovdqu32 zmmword ptr [rax+0xC0], zmm0
       vmovdqu32 zmmword ptr [rax+0x100], zmm0
       vmovdqu32 zmmword ptr [rax+0x140], zmm0
       vmovdqu32 zmmword ptr [rax+0x180], zmm0
       vmovdqu32 zmmword ptr [rax+0x1C0], zmm0
       mov      qword ptr [rbp-0x38], rax
       xor      eax, eax
       jmp      SHORT G_M000_IG06
       align    [0 bytes for IG07]
 
G_M000_IG03:                ;; offset=0x012F
       inc      edx
       cmp      edx, 4
       jge      SHORT G_M000_IG05
 
G_M000_IG04:                ;; offset=0x0136
       xor      r14d, r14d
       jmp      G_M000_IG13
 
G_M000_IG05:                ;; offset=0x013E
       mov      eax, dword ptr [rbp-0x3C]
       inc      eax
       cmp      eax, 8
       mov      ebx, dword ptr [rbp+0x18]
       mov      r15d, dword ptr [rbp+0x20]
       jge      G_M000_IG16
 
G_M000_IG06:                ;; offset=0x0153
       mov      dword ptr [rbp+0x20], r15d
       mov      dword ptr [rbp-0x3C], eax
       lea      r14d, [r15+rax]
       mov      eax, r14d
       cdq      
       idiv     edx:eax, ebx
       lea      edx, [2*rax-0x01]
       mov      dword ptr [rbp-0x40], edx
       mov      dword ptr [rbp+0x18], ebx
       mov      eax, r14d
       cdq      
       idiv     edx:eax, ebx
       lea      eax, [2*rdx-0x01]
       xor      edx, edx
       jmp      SHORT G_M000_IG04
 
G_M000_IG07:                ;; offset=0x0182
       xor      esi, esi
 
G_M000_IG08:                ;; offset=0x0184
       movsxd   rcx, ecx
       shl      rcx, 2
       lea      r15, [rcx+r13]
       test     esi, esi
       jne      G_M000_IG26
 
G_M000_IG09:                ;; offset=0x0197
       xor      ebx, ebx
 
G_M000_IG10:                ;; offset=0x0199
       mov      dword ptr [r15], ebx
       add      rcx, r12
       test     esi, esi
       jne      G_M000_IG27
 
G_M000_IG11:                ;; offset=0x01A7
       xor      esi, esi
 
G_M000_IG12:                ;; offset=0x01A9
       mov      dword ptr [rcx], esi
       inc      r14d
       cmp      r14d, 4
       mov      dword ptr [rbp-0x3C], edi
       jge      G_M000_IG03
 
G_M000_IG13:                ;; offset=0x01BB
       lea      ecx, [r14+4*rdx]
       mov      edi, dword ptr [rbp-0x3C]
       lea      ecx, [rdi+8*rcx]
       cmp      edi, r10d
       jge      SHORT G_M000_IG07
 
G_M000_IG14:                ;; offset=0x01CA
       mov      esi, dword ptr [rbp-0x40]
       lea      r15d, [rsi+rdx]
       cmp      r15d, r9d
       jae      SHORT G_M000_IG07
 
G_M000_IG15:                ;; offset=0x01D6
       lea      r15d, [rax+r14]
       cmp      r15d, r11d
       setb     r15b
       movzx    r15, r15b
       mov      esi, r15d
       jmp      SHORT G_M000_IG08
 
G_M000_IG16:                ;; offset=0x01EA
       xor      rax, rax
       cmp      dword ptr [rbp-0x54], 0
       cmovne   rax, bword ptr [rbp-0x60]
       mov      bword ptr [rbp-0x48], rax
       xor      rdi, rdi
       cmp      dword ptr [rbp-0x58], 0
       cmovne   rdi, bword ptr [rbp-0x68]
       mov      bword ptr [rbp-0x50], rdi
       mov      rcx, rdi
       xor      edi, edi
       mov      r8d, dword ptr [rbp-0x2C]
       cmp      edi, r8d
       jge      G_M000_IG22
 
G_M000_IG17:                ;; offset=0x021A
       mov      esi, edi
       imul     esi, r9d
       imul     esi, r11d
       movsxd   rsi, esi
       lea      rsi, [rax+4*rsi]
       xor      r10d, r10d
       mov      ebx, 4
       align    [0 bytes for IG18]
 
G_M000_IG18:                ;; offset=0x0233
       lea      r15d, [8*r10]
       vxorps   ymm0, ymm0, ymm0
       movsxd   r15, r15d
       vmovups  ymm1, ymmword ptr [r13+4*r15]
       vmovups  ymm2, ymmword ptr [r12+4*r15]
       vmovaps  ymm3, ymm2
       vgatherdps ymm0, dword ptr [rsi+4*xmm1], ymm3
       vxorps   ymm1, ymm1, ymm1
       vmovups  ymm2, ymmword ptr [r13+4*r15+0x20]
       vmovups  ymm3, ymmword ptr [r12+4*r15+0x20]
       vmovaps  ymm4, ymm3
       vgatherdps ymm1, dword ptr [rsi+4*xmm2], ymm4
       vxorps   ymm2, ymm2, ymm2
       vmovups  ymm3, ymmword ptr [r13+4*r15+0x40]
       vmovups  ymm4, ymmword ptr [r12+4*r15+0x40]
       vmovaps  ymm5, ymm4
       vgatherdps ymm2, dword ptr [rsi+4*xmm3], ymm5
       vxorps   ymm3, ymm3, ymm3
       vmovups  ymm4, ymmword ptr [r13+4*r15+0x60]
       vmovups  ymm5, ymmword ptr [r12+4*r15+0x60]
       vmovaps  ymm6, ymm5
       vgatherdps ymm3, dword ptr [rsi+4*xmm4], ymm6
       vsubps   ymm0, ymm0, ymm2
       movsxd   r15, r10d
       shl      r15, 5
       mov      rdx, qword ptr [rbp-0x38]
       vmovups  ymmword ptr [rdx+r15], ymm0
       vaddps   ymm0, ymm1, ymm2
       lea      r15d, [r10+0x01]
       movsxd   r15, r15d
       shl      r15, 5
       vmovups  ymmword ptr [rdx+r15], ymm0
       vsubps   ymm0, ymm2, ymm1
       lea      r15d, [r10+0x02]
       movsxd   r15, r15d
       shl      r15, 5
       vmovups  ymmword ptr [rdx+r15], ymm0
       vsubps   ymm0, ymm1, ymm3
       lea      r15d, [r10+0x03]
       movsxd   r15, r15d
       shl      r15, 5
       vmovups  ymmword ptr [rdx+r15], ymm0
       add      r10d, 4
       dec      ebx
       jne      G_M000_IG18
 
G_M000_IG19:                ;; offset=0x030D
       xor      esi, esi
       align    [0 bytes for IG20]
 
G_M000_IG20:                ;; offset=0x030F
       movsxd   r10, esi
       shl      r10, 5
       mov      rdx, qword ptr [rbp-0x38]
       vmovups  ymm0, ymmword ptr [rdx+r10]
       lea      r10d, [rsi+0x04]
       movsxd   rbx, r10d
       shl      rbx, 5
       vmovups  ymm1, ymmword ptr [rdx+rbx]
       lea      ebx, [rsi+0x08]
       movsxd   r15, ebx
       shl      r15, 5
       vmovups  ymm2, ymmword ptr [rdx+r15]
       lea      r15d, [rsi+0x0C]
       movsxd   r14, r15d
       shl      r14, 5
       vmovups  ymm3, ymmword ptr [rdx+r14]
       vsubps   ymm0, ymm0, ymm2
       mov      r14d, esi
       imul     r14d, r8d
       add      r14d, edi
       shl      r14d, 3
       movsxd   r14, r14d
       vmovups  ymmword ptr [rcx+4*r14], ymm0
       vaddps   ymm0, ymm1, ymm2
       imul     r10d, r8d
       add      r10d, edi
       shl      r10d, 3
       movsxd   r10, r10d
       vmovups  ymmword ptr [rcx+4*r10], ymm0
       vsubps   ymm0, ymm2, ymm1
       imul     ebx, r8d
       add      ebx, edi
       shl      ebx, 3
       movsxd   r10, ebx
       vmovups  ymmword ptr [rcx+4*r10], ymm0
       vsubps   ymm0, ymm1, ymm3
       imul     r15d, r8d
       add      r15d, edi
       shl      r15d, 3
       movsxd   r10, r15d
       vmovups  ymmword ptr [rcx+4*r10], ymm0
       inc      esi
       cmp      esi, 4
       jl       G_M000_IG20
 
G_M000_IG21:                ;; offset=0x03BD
       inc      edi
       cmp      edi, r8d
       jl       G_M000_IG17
 
G_M000_IG22:                ;; offset=0x03C8
       xor      eax, eax
       mov      bword ptr [rbp-0x48], rax
 
G_M000_IG23:                ;; offset=0x03CE
       mov      bword ptr [rbp-0x50], rax
       cmp      qword ptr [rbp-0x70], 0x1DD1ABC5
       je       SHORT G_M000_IG24
       call     CORINFO_HELP_FAIL_FAST
 
G_M000_IG24:                ;; offset=0x03E1
       nop      
 
G_M000_IG25:                ;; offset=0x03E2
       vzeroupper 
       lea      rsp, [rbp-0x28]
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
G_M000_IG26:                ;; offset=0x03F4
       mov      ebx, dword ptr [rbp-0x40]
       lea      r8d, [rbx+rdx]
       imul     r8d, r11d
       add      r8d, eax
       add      r8d, r14d
       mov      ebx, r8d
       jmp      G_M000_IG10
 
G_M000_IG27:                ;; offset=0x040D
       mov      esi, -1
       jmp      G_M000_IG12
 
; Total bytes of code 1047


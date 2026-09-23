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
       sub      rsp, 104
       lea      rbp, [rsp+0x90]
       xor      eax, eax
       mov      qword ptr [rbp-0x48], rax
       mov      qword ptr [rbp-0x50], rax
       mov      qword ptr [rbp-0x70], 0x1DE67DC0
       mov      r14, rdx
       mov      r11d, dword ptr [rbp+0x10]
       mov      ebx, dword ptr [rbp+0x18]
       mov      r15d, dword ptr [rbp+0x20]
       mov      r10d, dword ptr [rbp+0x28]
 
G_M000_IG02:                ;; offset=0x003A
       mov      eax, r15d
       cdq      
       idiv     edx:eax, ebx
       mov      r13d, eax
       lea      r12d, [2*r13-0x01]
       mov      eax, r15d
       cdq      
       idiv     edx:eax, ebx
       lea      edx, [2*rdx-0x01]
       mov      dword ptr [rbp-0x30], edx
       cmp      r10d, 8
       jne      SHORT G_M000_IG08
 
G_M000_IG03:                ;; offset=0x0061
       lea      eax, [r15+0x07]
       cdq      
       idiv     edx:eax, ebx
       cmp      eax, r13d
       jne      SHORT G_M000_IG08
 
G_M000_IG04:                ;; offset=0x006D
       test     r12d, r12d
       jl       SHORT G_M000_IG08
 
G_M000_IG05:                ;; offset=0x0072
       lea      edx, [r12+0x03]
       cmp      edx, r9d
       jge      SHORT G_M000_IG08
 
G_M000_IG06:                ;; offset=0x007C
       mov      r13d, dword ptr [rbp-0x30]
       test     r13d, r13d
       jl       SHORT G_M000_IG08
 
G_M000_IG07:                ;; offset=0x0085
       lea      edx, [r13+0x11]
       cmp      edx, r11d
       jl       G_M000_IG35
 
G_M000_IG08:                ;; offset=0x0092
       test     dword ptr [rsp], esp
       sub      rsp, 512
       lea      rax, [rsp+0x20]
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
       lea      rax, [rsp+0x20]
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
       lea      rax, [rsp+0x20]
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
       mov      bword ptr [rbp-0x68], r14
       jmp      SHORT G_M000_IG10
       align    [0 bytes for IG13]
 
G_M000_IG09:                ;; offset=0x0182
       mov      eax, dword ptr [rbp-0x3C]
       inc      eax
       cmp      eax, 8
       mov      ebx, dword ptr [rbp+0x18]
       mov      r15d, dword ptr [rbp+0x20]
       jge      G_M000_IG22
 
G_M000_IG10:                ;; offset=0x0197
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
       jmp      SHORT G_M000_IG12
 
G_M000_IG11:                ;; offset=0x01C6
       mov      ecx, dword ptr [rbp-0x58]
       mov      rdi, bword ptr [rbp-0x60]
       inc      edx
       cmp      edx, 4
       jge      SHORT G_M000_IG09
 
G_M000_IG12:                ;; offset=0x01D4
       xor      r14d, r14d
       mov      bword ptr [rbp-0x60], rdi
       mov      dword ptr [rbp-0x58], ecx
       jmp      SHORT G_M000_IG19
 
G_M000_IG13:                ;; offset=0x01E0
       xor      esi, esi
 
G_M000_IG14:                ;; offset=0x01E2
       movsxd   rcx, ecx
       shl      rcx, 2
       lea      r15, [rcx+r13]
       test     esi, esi
       jne      G_M000_IG33
 
G_M000_IG15:                ;; offset=0x01F5
       xor      ebx, ebx
 
G_M000_IG16:                ;; offset=0x01F7
       mov      dword ptr [r15], ebx
       add      rcx, r12
       test     esi, esi
       jne      G_M000_IG34
 
G_M000_IG17:                ;; offset=0x0205
       xor      esi, esi
 
G_M000_IG18:                ;; offset=0x0207
       mov      dword ptr [rcx], esi
       inc      r14d
       cmp      r14d, 4
       mov      dword ptr [rbp-0x3C], edi
       mov      esi, dword ptr [rbp-0x54]
       jge      SHORT G_M000_IG11
 
G_M000_IG19:                ;; offset=0x0218
       lea      ecx, [r14+4*rdx]
       mov      edi, dword ptr [rbp-0x3C]
       lea      ecx, [rdi+8*rcx]
       cmp      edi, r10d
       mov      dword ptr [rbp-0x54], esi
       jge      SHORT G_M000_IG13
 
G_M000_IG20:                ;; offset=0x022A
       mov      esi, dword ptr [rbp-0x40]
       lea      r15d, [rsi+rdx]
       cmp      r15d, r9d
       jae      SHORT G_M000_IG13
 
G_M000_IG21:                ;; offset=0x0236
       lea      r15d, [rax+r14]
       cmp      r15d, r11d
       setb     r15b
       movzx    r15, r15b
       mov      esi, r15d
       jmp      SHORT G_M000_IG14
 
G_M000_IG22:                ;; offset=0x024A
       mov      r14, bword ptr [rbp-0x68]
       xor      rax, rax
       test     esi, esi
       cmovne   rax, rdi
       mov      bword ptr [rbp-0x48], rax
       xor      rdi, rdi
       test     ecx, ecx
       cmovne   rdi, r14
       mov      bword ptr [rbp-0x50], rdi
       mov      rcx, rdi
       xor      edi, edi
       cmp      edi, r8d
       jge      G_M000_IG28
 
G_M000_IG23:                ;; offset=0x0274
       mov      esi, edi
       imul     esi, r9d
       imul     esi, r11d
       movsxd   rsi, esi
       lea      rsi, [rax+4*rsi]
       xor      r10d, r10d
       mov      ebx, 4
       align    [0 bytes for IG24]
 
G_M000_IG24:                ;; offset=0x028D
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
       jne      G_M000_IG24
 
G_M000_IG25:                ;; offset=0x0367
       xor      esi, esi
       align    [0 bytes for IG26]
 
G_M000_IG26:                ;; offset=0x0369
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
       jl       G_M000_IG26
 
G_M000_IG27:                ;; offset=0x0417
       inc      edi
       cmp      edi, r8d
       jl       G_M000_IG23
 
G_M000_IG28:                ;; offset=0x0422
       xor      eax, eax
       mov      bword ptr [rbp-0x48], rax
 
G_M000_IG29:                ;; offset=0x0428
       mov      bword ptr [rbp-0x50], rax
 
G_M000_IG30:                ;; offset=0x042C
       cmp      qword ptr [rbp-0x70], 0x1DE67DC0
       je       SHORT G_M000_IG31
       call     CORINFO_HELP_FAIL_FAST
 
G_M000_IG31:                ;; offset=0x043B
       nop      
 
G_M000_IG32:                ;; offset=0x043C
       vzeroupper 
       lea      rsp, [rbp-0x28]
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
G_M000_IG33:                ;; offset=0x044E
       mov      dword ptr [rbp-0x2C], r8d
       mov      ebx, dword ptr [rbp-0x40]
       lea      r8d, [rbx+rdx]
       imul     r8d, r11d
       add      r8d, eax
       add      r8d, r14d
       mov      ebx, r8d
       mov      r8d, dword ptr [rbp-0x2C]
       jmp      G_M000_IG16
 
G_M000_IG34:                ;; offset=0x046F
       mov      esi, -1
       jmp      G_M000_IG18
 
G_M000_IG35:                ;; offset=0x0479
       mov      dword ptr [rsp], r11d
       mov      dword ptr [rsp+0x08], r12d
       mov      dword ptr [rsp+0x10], r13d
       mov      rdx, r14
       call     [Lokad.Onnx.ConvBlockedSpatial:TransformWinogradInputContiguous(System.ReadOnlySpan`1[float],System.Span`1[float],int,int,int,int,int)]
       jmp      SHORT G_M000_IG30
 
; Total bytes of code 1170


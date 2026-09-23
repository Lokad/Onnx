; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:TransformWinogradInput(System.ReadOnlySpan`1[float],System.Span`1[float],int,int,int,int,int,int) (Tier0-FullOpts)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
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
       sub      rsp, 136
       lea      rbp, [rsp+0xB0]
       xor      eax, eax
       mov      qword ptr [rbp-0x48], rax
       mov      qword ptr [rbp-0x50], rax
       mov      qword ptr [rbp-0x88], 0x1DF9575E
       mov      r11, rdi
       mov      r13d, r8d
       mov      r8d, esi
       mov      r14d, r9d
       mov      r9, rdx
       mov      r15d, dword ptr [rbp+0x10]
       mov      r12d, dword ptr [rbp+0x18]
       mov      r10d, dword ptr [rbp+0x20]
       mov      ebx, dword ptr [rbp+0x28]
 
G_M000_IG02:                ;; offset=0x004C
       mov      eax, r10d
       cdq      
       idiv     edx:eax, r12d
       mov      edi, eax
       lea      esi, [2*rdi-0x01]
       mov      eax, r10d
       cdq      
       idiv     edx:eax, r12d
       lea      edx, [2*rdx-0x01]
       mov      dword ptr [rbp-0x30], edx
       cmp      ebx, 8
       jne      G_M000_IG13
 
G_M000_IG03:                ;; offset=0x0076
       mov      dword ptr [rbp+0x20], r10d
       lea      eax, [r10+0x07]
       cdq      
       idiv     edx:eax, r12d
       cmp      eax, edi
       jne      SHORT G_M000_IG12
 
G_M000_IG04:                ;; offset=0x0086
       test     esi, esi
       jl       SHORT G_M000_IG11
 
G_M000_IG05:                ;; offset=0x008A
       lea      edi, [rsi+0x03]
       cmp      edi, r14d
       jge      SHORT G_M000_IG10
 
G_M000_IG06:                ;; offset=0x0092
       mov      edi, dword ptr [rbp-0x30]
       test     edi, edi
       jl       SHORT G_M000_IG09
 
G_M000_IG07:                ;; offset=0x0099
       lea      edx, [rdi+0x11]
       cmp      edx, r15d
       jl       G_M000_IG41
 
G_M000_IG08:                ;; offset=0x00A5
       mov      bword ptr [rbp-0x78], r11
       mov      bword ptr [rbp-0x80], r9
       mov      dword ptr [rbp-0x54], r8d
       mov      dword ptr [rbp-0x58], ecx
       jmp      SHORT G_M000_IG14
       align    [0 bytes for IG19]
 
G_M000_IG09:                ;; offset=0x00B6
       mov      bword ptr [rbp-0x78], r11
       mov      bword ptr [rbp-0x80], r9
       mov      dword ptr [rbp-0x54], r8d
       mov      dword ptr [rbp-0x58], ecx
       jmp      SHORT G_M000_IG14
 
G_M000_IG10:                ;; offset=0x00C7
       mov      bword ptr [rbp-0x78], r11
       mov      bword ptr [rbp-0x80], r9
       mov      dword ptr [rbp-0x54], r8d
       mov      dword ptr [rbp-0x58], ecx
       jmp      SHORT G_M000_IG14
 
G_M000_IG11:                ;; offset=0x00D8
       mov      bword ptr [rbp-0x78], r11
       mov      bword ptr [rbp-0x80], r9
       mov      dword ptr [rbp-0x54], r8d
       mov      dword ptr [rbp-0x58], ecx
       jmp      SHORT G_M000_IG14
 
G_M000_IG12:                ;; offset=0x00E9
       mov      bword ptr [rbp-0x78], r11
       mov      bword ptr [rbp-0x80], r9
       mov      dword ptr [rbp-0x54], r8d
       mov      dword ptr [rbp-0x58], ecx
       jmp      SHORT G_M000_IG14
 
G_M000_IG13:                ;; offset=0x00FA
       mov      dword ptr [rbp+0x20], r10d
       mov      bword ptr [rbp-0x78], r11
       mov      bword ptr [rbp-0x80], r9
       mov      dword ptr [rbp-0x54], r8d
       mov      dword ptr [rbp-0x58], ecx
 
G_M000_IG14:                ;; offset=0x010D
       test     dword ptr [rsp], esp
       sub      rsp, 512
       lea      rdi, [rsp+0x20]
       mov      qword ptr [rbp-0x60], rdi
       mov      esi, 512
       call     [CORINFO_HELP_MEMZERO]
       mov      rdi, qword ptr [rbp-0x60]
       mov      qword ptr [rbp-0x38], rdi
       test     dword ptr [rsp], esp
       sub      rsp, 512
       lea      rdi, [rsp+0x20]
       mov      qword ptr [rbp-0x68], rdi
       mov      esi, 512
       call     [CORINFO_HELP_MEMZERO]
       mov      rdi, qword ptr [rbp-0x68]
       mov      qword ptr [rbp-0x40], rdi
       test     dword ptr [rsp], esp
       sub      rsp, 512
       lea      rdi, [rsp+0x20]
       mov      qword ptr [rbp-0x70], rdi
       mov      esi, 512
       call     [CORINFO_HELP_MEMZERO]
       mov      rdi, qword ptr [rbp-0x70]
       mov      rcx, rdi
       xor      edi, edi
       jmp      SHORT G_M000_IG16
 
G_M000_IG15:                ;; offset=0x0182
       inc      edi
       cmp      edi, 8
       mov      r12d, dword ptr [rbp+0x18]
       jge      G_M000_IG28
 
G_M000_IG16:                ;; offset=0x0191
       mov      esi, edi
       add      esi, dword ptr [rbp+0x20]
       mov      eax, esi
       cdq      
       idiv     edx:eax, r12d
       lea      r8d, [2*rax-0x01]
       mov      dword ptr [rbp+0x18], r12d
       mov      eax, esi
       cdq      
       idiv     edx:eax, r12d
       lea      eax, [2*rdx-0x01]
       xor      edx, edx
       jmp      SHORT G_M000_IG18
 
G_M000_IG17:                ;; offset=0x01B9
       inc      edx
       cmp      edx, 4
       jge      SHORT G_M000_IG15
 
G_M000_IG18:                ;; offset=0x01C0
       xor      esi, esi
       jmp      SHORT G_M000_IG25
 
G_M000_IG19:                ;; offset=0x01C4
       xor      r11d, r11d
 
G_M000_IG20:                ;; offset=0x01C7
       movsxd   r9, r9d
       shl      r9, 2
       mov      r10, qword ptr [rbp-0x38]
       lea      r12, [r10+r9]
       test     r11d, r11d
       jne      G_M000_IG39
 
G_M000_IG21:                ;; offset=0x01DF
       mov      dword ptr [rbp-0x2C], r13d
       xor      r13d, r13d
 
G_M000_IG22:                ;; offset=0x01E6
       mov      dword ptr [r12], r13d
       mov      r13, qword ptr [rbp-0x40]
       add      r9, r13
       test     r11d, r11d
       jne      G_M000_IG40
 
G_M000_IG23:                ;; offset=0x01FA
       xor      r11d, r11d
 
G_M000_IG24:                ;; offset=0x01FD
       mov      dword ptr [r9], r11d
       inc      esi
       cmp      esi, 4
       mov      r13d, dword ptr [rbp-0x2C]
       jge      SHORT G_M000_IG17
 
G_M000_IG25:                ;; offset=0x020B
       lea      r9d, [rsi+4*rdx]
       lea      r9d, [rdi+8*r9]
       cmp      edi, ebx
       jge      SHORT G_M000_IG19
 
G_M000_IG26:                ;; offset=0x0217
       lea      r11d, [r8+rdx]
       cmp      r11d, r14d
       jae      SHORT G_M000_IG19
 
G_M000_IG27:                ;; offset=0x0220
       lea      r11d, [rax+rsi]
       cmp      r11d, r15d
       setb     r11b
       movzx    r11, r11b
       jmp      SHORT G_M000_IG20
 
G_M000_IG28:                ;; offset=0x0231
       xor      rax, rax
       cmp      dword ptr [rbp-0x54], 0
       cmovne   rax, bword ptr [rbp-0x78]
       mov      bword ptr [rbp-0x48], rax
       xor      rdx, rdx
       cmp      dword ptr [rbp-0x58], 0
       cmovne   rdx, bword ptr [rbp-0x80]
       mov      bword ptr [rbp-0x50], rdx
       xor      edi, edi
       cmp      edi, r13d
       jge      G_M000_IG34
 
G_M000_IG29:                ;; offset=0x025A
       mov      esi, edi
       imul     esi, r14d
       imul     esi, r15d
       movsxd   rsi, esi
       lea      rsi, [rax+4*rsi]
       xor      r8d, r8d
       mov      r9d, 4
       align    [0 bytes for IG30]
 
G_M000_IG30:                ;; offset=0x0274
       lea      r10d, [8*r8]
       vxorps   ymm0, ymm0, ymm0
       movsxd   r10, r10d
       mov      rbx, qword ptr [rbp-0x38]
       vmovups  ymm1, ymmword ptr [rbx+4*r10]
       mov      r12, qword ptr [rbp-0x40]
       vmovups  ymm2, ymmword ptr [r12+4*r10]
       vmovaps  ymm3, ymm2
       vgatherdps ymm0, dword ptr [rsi+4*xmm1], ymm3
       vxorps   ymm1, ymm1, ymm1
       vmovups  ymm2, ymmword ptr [rbx+4*r10+0x20]
       vmovups  ymm3, ymmword ptr [r12+4*r10+0x20]
       vmovaps  ymm4, ymm3
       vgatherdps ymm1, dword ptr [rsi+4*xmm2], ymm4
       vxorps   ymm2, ymm2, ymm2
       vmovups  ymm3, ymmword ptr [rbx+4*r10+0x40]
       vmovups  ymm4, ymmword ptr [r12+4*r10+0x40]
       vmovaps  ymm5, ymm4
       vgatherdps ymm2, dword ptr [rsi+4*xmm3], ymm5
       vxorps   ymm3, ymm3, ymm3
       vmovups  ymm4, ymmword ptr [rbx+4*r10+0x60]
       vmovups  ymm5, ymmword ptr [r12+4*r10+0x60]
       vmovaps  ymm6, ymm5
       vgatherdps ymm3, dword ptr [rsi+4*xmm4], ymm6
       vsubps   ymm0, ymm0, ymm2
       movsxd   r10, r8d
       shl      r10, 5
       vmovups  ymmword ptr [rcx+r10], ymm0
       vaddps   ymm0, ymm1, ymm2
       lea      r10d, [r8+0x01]
       movsxd   r10, r10d
       shl      r10, 5
       vmovups  ymmword ptr [rcx+r10], ymm0
       vsubps   ymm0, ymm2, ymm1
       lea      r10d, [r8+0x02]
       movsxd   r10, r10d
       shl      r10, 5
       vmovups  ymmword ptr [rcx+r10], ymm0
       vsubps   ymm0, ymm1, ymm3
       lea      r10d, [r8+0x03]
       movsxd   r10, r10d
       shl      r10, 5
       vmovups  ymmword ptr [rcx+r10], ymm0
       add      r8d, 4
       dec      r9d
       jne      G_M000_IG30
 
G_M000_IG31:                ;; offset=0x0352
       xor      esi, esi
       align    [0 bytes for IG32]
 
G_M000_IG32:                ;; offset=0x0354
       movsxd   r8, esi
       shl      r8, 5
       vmovups  ymm0, ymmword ptr [rcx+r8]
       lea      r8d, [rsi+0x04]
       movsxd   r9, r8d
       shl      r9, 5
       vmovups  ymm1, ymmword ptr [rcx+r9]
       lea      r9d, [rsi+0x08]
       movsxd   r10, r9d
       shl      r10, 5
       vmovups  ymm2, ymmword ptr [rcx+r10]
       lea      r10d, [rsi+0x0C]
       movsxd   r11, r10d
       shl      r11, 5
       vmovups  ymm3, ymmword ptr [rcx+r11]
       vsubps   ymm0, ymm0, ymm2
       mov      r11d, esi
       imul     r11d, r13d
       add      r11d, edi
       shl      r11d, 3
       movsxd   r11, r11d
       vmovups  ymmword ptr [rdx+4*r11], ymm0
       vaddps   ymm0, ymm1, ymm2
       imul     r8d, r13d
       add      r8d, edi
       shl      r8d, 3
       movsxd   r8, r8d
       vmovups  ymmword ptr [rdx+4*r8], ymm0
       vsubps   ymm0, ymm2, ymm1
       imul     r9d, r13d
       add      r9d, edi
       shl      r9d, 3
       movsxd   r8, r9d
       vmovups  ymmword ptr [rdx+4*r8], ymm0
       vsubps   ymm0, ymm1, ymm3
       imul     r10d, r13d
       add      r10d, edi
       shl      r10d, 3
       movsxd   r8, r10d
       vmovups  ymmword ptr [rdx+4*r8], ymm0
       inc      esi
       cmp      esi, 4
       jl       G_M000_IG32
 
G_M000_IG33:                ;; offset=0x0402
       inc      edi
       cmp      edi, r13d
       jl       G_M000_IG29
 
G_M000_IG34:                ;; offset=0x040D
       xor      eax, eax
       mov      bword ptr [rbp-0x48], rax
 
G_M000_IG35:                ;; offset=0x0413
       mov      bword ptr [rbp-0x50], rax
 
G_M000_IG36:                ;; offset=0x0417
       cmp      qword ptr [rbp-0x88], 0x1DF9575E
       je       SHORT G_M000_IG37
       call     CORINFO_HELP_FAIL_FAST
 
G_M000_IG37:                ;; offset=0x0429
       nop      
 
G_M000_IG38:                ;; offset=0x042A
       vzeroupper 
       lea      rsp, [rbp-0x28]
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
G_M000_IG39:                ;; offset=0x043C
       mov      dword ptr [rbp-0x2C], r13d
       lea      r13d, [r8+rdx]
       imul     r13d, r15d
       add      r13d, eax
       add      r13d, esi
       jmp      G_M000_IG22
 
G_M000_IG40:                ;; offset=0x0453
       mov      r11d, -1
       jmp      G_M000_IG24
 
G_M000_IG41:                ;; offset=0x045E
       mov      dword ptr [rsp], r15d
       mov      dword ptr [rsp+0x08], esi
       mov      dword ptr [rsp+0x10], edi
       mov      rdi, r11
       mov      esi, r8d
       mov      rdx, r9
       mov      r8d, r13d
       mov      r9d, r14d
       call     [Lokad.Onnx.ConvBlockedSpatial:TransformWinogradInputContiguous(System.ReadOnlySpan`1[float],System.Span`1[float],int,int,int,int,int)]
       jmp      SHORT G_M000_IG36
 
; Total bytes of code 1153


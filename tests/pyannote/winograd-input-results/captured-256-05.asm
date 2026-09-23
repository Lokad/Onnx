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
       sub      rsp, 104
       lea      rbp, [rsp+0x90]
       xor      eax, eax
       mov      qword ptr [rbp-0x48], rax
       mov      qword ptr [rbp-0x50], rax
       mov      qword ptr [rbp-0x88], 0x1DD15316
       mov      bword ptr [rbp-0x78], rdi
       mov      dword ptr [rbp-0x54], esi
       mov      bword ptr [rbp-0x80], rdx
       mov      dword ptr [rbp-0x58], ecx
       mov      dword ptr [rbp-0x2C], r8d
       mov      r14d, r9d
       mov      r15d, dword ptr [rbp+0x10]
       mov      r12d, dword ptr [rbp+0x18]
       mov      ebx, dword ptr [rbp+0x28]
 
G_M000_IG02:                ;; offset=0x004B
       test     dword ptr [rsp], esp
       sub      rsp, 512
       lea      rdi, [rsp]
       mov      qword ptr [rbp-0x60], rdi
       mov      esi, 512
       call     [CORINFO_HELP_MEMZERO]
       mov      rdi, qword ptr [rbp-0x60]
       mov      qword ptr [rbp-0x38], rdi
       test     dword ptr [rsp], esp
       sub      rsp, 512
       lea      rdi, [rsp]
       mov      qword ptr [rbp-0x68], rdi
       mov      esi, 512
       call     [CORINFO_HELP_MEMZERO]
       mov      rdi, qword ptr [rbp-0x68]
       mov      qword ptr [rbp-0x40], rdi
       test     dword ptr [rsp], esp
       sub      rsp, 512
       lea      rdi, [rsp]
       mov      qword ptr [rbp-0x70], rdi
       mov      esi, 512
       call     [CORINFO_HELP_MEMZERO]
       mov      rdi, qword ptr [rbp-0x70]
       mov      rcx, rdi
       xor      edi, edi
       jmp      SHORT G_M000_IG06
       align    [0 bytes for IG07]
 
G_M000_IG03:                ;; offset=0x00BD
       inc      edx
       cmp      edx, 4
       jge      SHORT G_M000_IG05
 
G_M000_IG04:                ;; offset=0x00C4
       xor      r8d, r8d
       jmp      G_M000_IG13
 
G_M000_IG05:                ;; offset=0x00CC
       inc      edi
       cmp      edi, 8
       mov      r12d, dword ptr [rbp+0x18]
       jge      G_M000_IG16
 
G_M000_IG06:                ;; offset=0x00DB
       mov      r8d, edi
       add      r8d, dword ptr [rbp+0x20]
       mov      eax, r8d
       cdq      
       idiv     edx:eax, r12d
       lea      r9d, [2*rax-0x01]
       mov      dword ptr [rbp+0x18], r12d
       mov      eax, r8d
       cdq      
       idiv     edx:eax, r12d
       lea      eax, [2*rdx-0x01]
       xor      edx, edx
       jmp      SHORT G_M000_IG04
 
G_M000_IG07:                ;; offset=0x0107
       xor      r11d, r11d
 
G_M000_IG08:                ;; offset=0x010A
       movsxd   r10, r10d
       shl      r10, 2
       mov      rsi, qword ptr [rbp-0x38]
       lea      r12, [rsi+r10]
       test     r11d, r11d
       jne      G_M000_IG26
 
G_M000_IG09:                ;; offset=0x0122
       xor      r13d, r13d
 
G_M000_IG10:                ;; offset=0x0125
       mov      dword ptr [r12], r13d
       mov      r13, qword ptr [rbp-0x40]
       add      r10, r13
       test     r11d, r11d
       jne      G_M000_IG27
 
G_M000_IG11:                ;; offset=0x0139
       xor      r11d, r11d
 
G_M000_IG12:                ;; offset=0x013C
       mov      dword ptr [r10], r11d
       inc      r8d
       cmp      r8d, 4
       jge      G_M000_IG03
 
G_M000_IG13:                ;; offset=0x014C
       lea      r10d, [r8+4*rdx]
       lea      r10d, [rdi+8*r10]
       cmp      edi, ebx
       jge      SHORT G_M000_IG07
 
G_M000_IG14:                ;; offset=0x0158
       lea      r11d, [r9+rdx]
       cmp      r11d, r14d
       jae      SHORT G_M000_IG07
 
G_M000_IG15:                ;; offset=0x0161
       lea      r11d, [rax+r8]
       cmp      r11d, r15d
       setb     r11b
       movzx    r11, r11b
       jmp      SHORT G_M000_IG08
 
G_M000_IG16:                ;; offset=0x0172
       xor      rax, rax
       cmp      dword ptr [rbp-0x54], 0
       cmovne   rax, bword ptr [rbp-0x78]
       mov      bword ptr [rbp-0x48], rax
       xor      rdx, rdx
       cmp      dword ptr [rbp-0x58], 0
       cmovne   rdx, bword ptr [rbp-0x80]
       mov      bword ptr [rbp-0x50], rdx
       xor      edi, edi
       mov      r13d, dword ptr [rbp-0x2C]
       cmp      edi, r13d
       jge      G_M000_IG22
 
G_M000_IG17:                ;; offset=0x019F
       mov      esi, edi
       imul     esi, r14d
       imul     esi, r15d
       movsxd   rsi, esi
       lea      rsi, [rax+4*rsi]
       xor      r8d, r8d
       mov      r9d, 4
       align    [0 bytes for IG18]
 
G_M000_IG18:                ;; offset=0x01B9
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
       jne      G_M000_IG18
 
G_M000_IG19:                ;; offset=0x0297
       xor      esi, esi
       align    [0 bytes for IG20]
 
G_M000_IG20:                ;; offset=0x0299
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
       jl       G_M000_IG20
 
G_M000_IG21:                ;; offset=0x0347
       inc      edi
       cmp      edi, r13d
       jl       G_M000_IG17
 
G_M000_IG22:                ;; offset=0x0352
       xor      eax, eax
       mov      bword ptr [rbp-0x48], rax
 
G_M000_IG23:                ;; offset=0x0358
       mov      bword ptr [rbp-0x50], rax
       cmp      qword ptr [rbp-0x88], 0x1DD15316
       je       SHORT G_M000_IG24
       call     CORINFO_HELP_FAIL_FAST
 
G_M000_IG24:                ;; offset=0x036E
       nop      
 
G_M000_IG25:                ;; offset=0x036F
       vzeroupper 
       lea      rsp, [rbp-0x28]
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
G_M000_IG26:                ;; offset=0x0381
       lea      r13d, [r9+rdx]
       imul     r13d, r15d
       add      r13d, eax
       add      r13d, r8d
       jmp      G_M000_IG10
 
G_M000_IG27:                ;; offset=0x0394
       mov      r11d, -1
       jmp      G_M000_IG12
 
; Total bytes of code 927


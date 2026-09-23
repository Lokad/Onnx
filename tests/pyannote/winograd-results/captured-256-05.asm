; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:TransformWinogradInput(System.ReadOnlySpan`1[float],System.Span`1[float],int,int,int,int,int,int) (Tier0-FullOpts)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
; Tier-0 switched to FullOpts code
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 100
; No PGO data
; 0 inlinees with PGO data; 0 single block inlinees; 3 inlinees without PGO data

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
       mov      qword ptr [rbp-0x88], rax
       vxorps   xmm8, xmm8, xmm8
       vmovdqu  ymmword ptr [rbp-0x80], ymm8
       vmovdqu  ymmword ptr [rbp-0x60], ymm8
       vmovdqa  xmmword ptr [rbp-0x40], xmm8
       mov      qword ptr [rbp-0x30], 0x1DB4A1CA
       mov      bword ptr [rbp-0x50], rdi
       mov      qword ptr [rbp-0x48], rsi
       mov      bword ptr [rbp-0x60], rdx
       mov      qword ptr [rbp-0x58], rcx
       mov      ebx, dword ptr [rbp+0x28]
 
G_M000_IG02:                ;; offset=0x0051
       mov      r15d, dword ptr [rbp+0x20]
       mov      dword ptr [rbp-0x94], r15d
       mov      r14d, dword ptr [rbp+0x18]
       mov      dword ptr [rbp-0x90], r14d
       mov      r13d, dword ptr [rbp+0x10]
       mov      r12d, r9d
       mov      dword ptr [rbp-0x8C], r8d
 
G_M000_IG03:                ;; offset=0x0075
       vmovdqu  xmm0, xmmword ptr [rbp-0x60]
       vmovdqu  xmmword ptr [rbp-0x88], xmm0
 
G_M000_IG04:                ;; offset=0x0082
       vmovdqu  xmm0, xmmword ptr [rbp-0x50]
       vmovdqu  xmmword ptr [rbp-0x78], xmm0
 
G_M000_IG05:                ;; offset=0x008C
       mov      rcx, bword ptr [rbp-0x78]
       mov      bword ptr [rbp-0xA0], rcx
       mov      rdx, bword ptr [rbp-0x88]
       mov      bword ptr [rbp-0xA8], rdx
       mov      r8d, dword ptr [rbp-0x80]
       mov      dword ptr [rbp-0x68], r8d
       mov      esi, r8d
       shl      rsi, 2
       mov      rdi, rdx
       call     [System.SpanHelpers:ClearWithoutReferences(byref,nuint)]
       test     dword ptr [rsp], esp
       sub      rsp, 64
       lea      rax, [rsp]
       vxorps   ymm0, ymm0, ymm0
       vmovdqu  ymmword ptr [rax], ymm0
       vmovdqu  ymmword ptr [rax+0x20], ymm0
       mov      rcx, rax
       lea      rdi, [rbp-0x40]
       xor      esi, esi
       mov      r8d, dword ptr [rbp-0x70]
       cmp      esi, dword ptr [rbp-0x8C]
       jl       G_M000_IG18
 
G_M000_IG06:                ;; offset=0x00EE
       cmp      qword ptr [rbp-0x30], 0x1DB4A1CA
       je       SHORT G_M000_IG07
       call     CORINFO_HELP_FAIL_FAST
 
G_M000_IG07:                ;; offset=0x00FD
       nop      
 
G_M000_IG08:                ;; offset=0x00FE
       lea      rsp, [rbp-0x28]
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
G_M000_IG09:                ;; offset=0x010D
       lea      r11d, [4*rdx]
       mov      r15d, r11d
       vmovss   xmm0, dword ptr [rdi]
       vsubss   xmm0, xmm0, dword ptr [rdi+0x08]
       vmovss   dword ptr [rcx+4*r15], xmm0
       lea      r15d, [r11+0x01]
       vmovss   xmm0, dword ptr [rdi+0x04]
       vaddss   xmm0, xmm0, dword ptr [rdi+0x08]
       vmovss   dword ptr [rcx+4*r15], xmm0
       lea      r15d, [r11+0x02]
       vmovss   xmm0, dword ptr [rdi+0x08]
       vsubss   xmm0, xmm0, dword ptr [rdi+0x04]
       vmovss   dword ptr [rcx+4*r15], xmm0
       add      r11d, 3
       vmovss   xmm0, dword ptr [rdi+0x04]
       vsubss   xmm0, xmm0, dword ptr [rdi+0x0C]
       vmovss   dword ptr [rcx+4*r11], xmm0
       inc      edx
       cmp      edx, 4
       jge      SHORT G_M000_IG11
 
G_M000_IG10:                ;; offset=0x016A
       xor      r11d, r11d
       jmp      G_M000_IG21
       align    [0 bytes for IG12]
 
G_M000_IG11:                ;; offset=0x0172
       xor      eax, eax
 
G_M000_IG12:                ;; offset=0x0174
       vmovss   xmm0, dword ptr [rcx+4*rax]
       lea      edx, [rax+0x04]
       mov      r11d, edx
       vmovss   xmm1, dword ptr [rcx+4*r11]
       lea      r11d, [rax+0x08]
       mov      r15d, r11d
       vmovss   xmm2, dword ptr [rcx+4*r15]
       lea      r15d, [rax+0x0C]
       mov      ebx, r15d
       vmovss   xmm3, dword ptr [rcx+4*rbx]
       mov      r9d, dword ptr [rbp-0x8C]
       mov      ebx, eax
       imul     ebx, r9d
       add      ebx, esi
       lea      ebx, [r10+8*rbx]
       cmp      ebx, dword ptr [rbp-0x68]
       jae      G_M000_IG24
       vsubss   xmm0, xmm0, xmm2
       mov      r14, bword ptr [rbp-0xA8]
       vmovss   dword ptr [r14+4*rbx], xmm0
       imul     edx, r9d
       add      edx, esi
       lea      edx, [r10+8*rdx]
       mov      ebx, dword ptr [rbp-0x68]
       cmp      edx, ebx
       jae      G_M000_IG24
       vaddss   xmm0, xmm1, xmm2
       vmovss   dword ptr [r14+4*rdx], xmm0
       imul     r11d, r9d
       add      r11d, esi
       lea      edx, [r10+8*r11]
       cmp      edx, ebx
       jae      G_M000_IG24
       vsubss   xmm0, xmm2, xmm1
       vmovss   dword ptr [r14+4*rdx], xmm0
       imul     r15d, r9d
       add      r15d, esi
       lea      edx, [r10+8*r15]
       cmp      edx, ebx
       jae      G_M000_IG24
       vsubss   xmm0, xmm1, xmm3
       vmovss   dword ptr [r14+4*rdx], xmm0
       inc      eax
       cmp      eax, 4
       jl       G_M000_IG12
 
G_M000_IG13:                ;; offset=0x022F
       inc      r10d
       mov      ebx, dword ptr [rbp+0x28]
       cmp      r10d, ebx
       jge      SHORT G_M000_IG17
 
G_M000_IG14:                ;; offset=0x023A
       mov      dword ptr [rbp+0x28], ebx
       mov      r14d, dword ptr [rbp-0x90]
       mov      r15d, dword ptr [rbp-0x94]
 
G_M000_IG15:                ;; offset=0x024B
       lea      r11d, [r15+r10]
       mov      eax, r11d
       cdq      
       idiv     edx:eax, r14d
       lea      edx, [2*rax-0x01]
       mov      dword ptr [rbp-0x64], edx
       mov      eax, r11d
       cdq      
       idiv     edx:eax, r14d
       lea      eax, [2*rdx-0x01]
       xor      edx, edx
       jmp      G_M000_IG10
       align    [0 bytes for IG19]
 
G_M000_IG16:                ;; offset=0x0275
       mov      ebx, dword ptr [rbp+0x28]
 
G_M000_IG17:                ;; offset=0x0278
       inc      esi
       mov      r9d, dword ptr [rbp-0x8C]
       cmp      esi, r9d
       mov      r14d, dword ptr [rbp-0x90]
       mov      r15d, dword ptr [rbp-0x94]
       jge      G_M000_IG06
 
G_M000_IG18:                ;; offset=0x0298
       xor      r10d, r10d
       mov      dword ptr [rbp+0x28], ebx
       cmp      r10d, ebx
       jl       SHORT G_M000_IG15
       jmp      SHORT G_M000_IG16
 
G_M000_IG19:                ;; offset=0x02A5
       vxorps   xmm0, xmm0, xmm0
 
G_M000_IG20:                ;; offset=0x02A9
       vmovss   dword ptr [r15], xmm0
       inc      r11d
       cmp      r11d, 4
       jge      G_M000_IG09
 
G_M000_IG21:                ;; offset=0x02BB
       lea      r15, bword ptr [rdi+4*r11]
       mov      ebx, dword ptr [rbp-0x64]
       lea      r14d, [rbx+rdx]
       cmp      r14d, r12d
       jae      SHORT G_M000_IG19
 
G_M000_IG22:                ;; offset=0x02CB
       lea      r14d, [rax+r11]
       cmp      r14d, r13d
       jae      SHORT G_M000_IG19
 
G_M000_IG23:                ;; offset=0x02D4
       mov      r14d, esi
       imul     r14d, r12d
       add      r14d, ebx
       add      r14d, edx
       imul     r14d, r13d
       add      r14d, eax
       add      r14d, r11d
       cmp      r14d, r8d
       jae      SHORT G_M000_IG24
       mov      r9, bword ptr [rbp-0xA0]
       vmovss   xmm0, dword ptr [r9+4*r14]
       jmp      SHORT G_M000_IG20
 
G_M000_IG24:                ;; offset=0x02FF
       call     CORINFO_HELP_RNGCHKFAIL
       int3     
 
; Total bytes of code 773


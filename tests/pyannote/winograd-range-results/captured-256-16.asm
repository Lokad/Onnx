; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:OutputWinograd256(ptr,ptr,int,int,int,int,int,int) (Tier1)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
; Tier1 code
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 33536

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       push     r15
       push     r14
       push     r13
       push     r12
       push     rbx
       sub      rsp, 24
       lea      rbp, [rsp+0x40]
       mov      r10d, edx
       mov      ebx, dword ptr [rbp+0x10]
       mov      r11d, dword ptr [rbp+0x18]
 
G_M000_IG02:                ;; offset=0x001D
       mov      r15d, ecx
       imul     r15d, r8d
       mov      dword ptr [rbp-0x34], r15d
       xor      r14d, r14d
       cmp      r14d, r10d
       jl       SHORT G_M000_IG06
 
G_M000_IG03:                ;; offset=0x0030
       vzeroupper 
       add      rsp, 24
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
G_M000_IG04:                ;; offset=0x0042
       mov      qword ptr [rbp-0x30], rdi
       mov      dword ptr [rbp+0x10], ebx
       mov      r15d, dword ptr [rbp-0x34]
 
G_M000_IG05:                ;; offset=0x004D
       add      r14d, 8
       cmp      r14d, r10d
       mov      ebx, dword ptr [rbp+0x10]
       mov      rdi, qword ptr [rbp-0x30]
       jge      SHORT G_M000_IG03
 
G_M000_IG06:                ;; offset=0x005D
       xor      r13d, r13d
       cmp      r13d, r11d
       jge      SHORT G_M000_IG04
       align    [0 bytes for IG07]
 
G_M000_IG07:                ;; offset=0x0065
       lea      eax, [8*r14]
       cdqe     
       shl      rax, 2
       mov      qword ptr [rbp-0x30], rdi
       add      rax, rdi
       lea      edx, [8*r13]
       movsxd   rdx, edx
       lea      r12, [rax+4*rdx]
       lea      edx, [8*r10]
       mov      dword ptr [rbp-0x38], edx
       mov      dword ptr [rbp+0x10], ebx
       lea      eax, [rbx+r13]
       mov      dword ptr [rbp-0x40], eax
       cdq      
       idiv     edx:eax, r9d
       lea      edx, [rax+rax]
       mov      dword ptr [rbp-0x3C], edx
       mov      eax, dword ptr [rbp-0x40]
       cdq      
       idiv     edx:eax, r9d
       add      edx, edx
       vmovups  ymm0, ymmword ptr [r12]
       mov      eax, dword ptr [rbp-0x38]
       lea      ebx, [4*rax]
       movsxd   rbx, ebx
       vmovups  ymm1, ymmword ptr [r12+4*rbx]
       vaddps   ymm0, ymm0, ymm1
       lea      ebx, [8*rax]
       movsxd   rbx, ebx
       vmovups  ymm2, ymmword ptr [r12+4*rbx]
       vaddps   ymm0, ymm0, ymm2
       vsubps   ymm1, ymm1, ymm2
       lea      ebx, [rax+2*rax]
       lea      r15d, [4*rbx]
       movsxd   r15, r15d
       vsubps   ymm1, ymm1, ymmword ptr [r12+4*r15]
       movsxd   r15, eax
       vmovups  ymm2, ymmword ptr [r12+4*r15]
       lea      r15d, [rax+4*rax]
       movsxd   rdi, r15d
       vmovups  ymm3, ymmword ptr [r12+4*rdi]
       vaddps   ymm2, ymm2, ymm3
       lea      edi, [rax+8*rax]
       movsxd   rdi, edi
       vmovups  ymm4, ymmword ptr [r12+4*rdi]
       vaddps   ymm2, ymm2, ymm4
       vsubps   ymm3, ymm3, ymm4
       imul     edi, eax, 13
       movsxd   rdi, edi
       vsubps   ymm3, ymm3, ymmword ptr [r12+4*rdi]
       lea      edi, [rax+rax]
       movsxd   rdi, edi
       vmovups  ymm4, ymmword ptr [r12+4*rdi]
       lea      edi, [rbx+rbx]
       movsxd   rdi, edi
       vmovups  ymm5, ymmword ptr [r12+4*rdi]
       vaddps   ymm4, ymm4, ymm5
       add      r15d, r15d
       movsxd   rdi, r15d
       vmovups  ymm6, ymmword ptr [r12+4*rdi]
       vaddps   ymm4, ymm6, ymm4
       vsubps   ymm5, ymm5, ymm6
       imul     edi, eax, 14
       movsxd   rdi, edi
       vsubps   ymm5, ymm5, ymmword ptr [r12+4*rdi]
       movsxd   rdi, ebx
 
G_M000_IG08:                ;; offset=0x0173
       vmovups  ymm6, ymmword ptr [r12+4*rdi]
       lea      edi, [8*rax]
       sub      edi, eax
       movsxd   rdi, edi
       vmovups  ymm7, ymmword ptr [r12+4*rdi]
       vaddps   ymm6, ymm6, ymm7
       imul     edi, eax, 11
       movsxd   rdi, edi
       vmovups  ymm8, ymmword ptr [r12+4*rdi]
       vaddps   ymm6, ymm6, ymm8
       vsubps   ymm7, ymm7, ymm8
       mov      edi, eax
       shl      edi, 4
       sub      edi, eax
       movsxd   rax, edi
       vsubps   ymm7, ymm7, ymmword ptr [r12+4*rax]
       vaddps   ymm0, ymm2, ymm0
       vaddps   ymm0, ymm0, ymm4
       vsubps   ymm2, ymm2, ymm4
       vsubps   ymm2, ymm2, ymm6
       vaddps   ymm1, ymm1, ymm3
       vaddps   ymm1, ymm1, ymm5
       vsubps   ymm3, ymm3, ymm5
       vsubps   ymm3, ymm3, ymm7
       mov      edi, dword ptr [rbp-0x3C]
       mov      eax, edi
       imul     eax, r8d
       add      eax, edx
       shl      eax, 3
       cdqe     
       mov      r15d, dword ptr [rbp-0x34]
       mov      ebx, r14d
       imul     ebx, r15d
       movsxd   rbx, ebx
       shl      rbx, 2
       add      rbx, rsi
       lea      rax, [rbx+4*rax]
       vmovups  ymmword ptr [rax], ymm0
       inc      edx
       cmp      edx, r8d
       jge      SHORT G_M000_IG10
 
G_M000_IG09:                ;; offset=0x0209
       vmovups  ymmword ptr [rax+0x20], ymm2
 
G_M000_IG10:                ;; offset=0x020E
       inc      edi
       cmp      edi, ecx
       jge      SHORT G_M000_IG13
 
G_M000_IG11:                ;; offset=0x0214
       lea      edi, [8*r8]
       movsxd   rbx, edi
       vmovups  ymmword ptr [rax+4*rbx], ymm1
       cmp      edx, r8d
       jge      SHORT G_M000_IG13
 
G_M000_IG12:                ;; offset=0x0229
       add      edi, 8
       movsxd   rdx, edi
       vmovups  ymmword ptr [rax+4*rdx], ymm3
 
G_M000_IG13:                ;; offset=0x0234
       inc      r13d
       cmp      r13d, r11d
       jge      G_M000_IG05
 
G_M000_IG14:                ;; offset=0x0240
       mov      ebx, dword ptr [rbp+0x10]
       mov      rdi, qword ptr [rbp-0x30]
       jmp      G_M000_IG07
 
; Total bytes of code 588


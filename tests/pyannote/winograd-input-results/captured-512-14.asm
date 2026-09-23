; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:OutputWinograd512(ptr,ptr,int,int,int,int,int,int) (Tier1)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Tier1 code
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 32640

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
       add      r14d, 16
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
       mov      edx, r13d
       shl      edx, 4
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
       vmovups  zmm0, zmmword ptr [r12]
       mov      eax, dword ptr [rbp-0x38]
       lea      ebx, [4*rax]
       movsxd   rbx, ebx
       vmovups  zmm1, zmmword ptr [r12+4*rbx]
       vaddps   zmm0, zmm0, zmm1
       lea      ebx, [8*rax]
       movsxd   rbx, ebx
       vmovups  zmm2, zmmword ptr [r12+4*rbx]
       vaddps   zmm0, zmm0, zmm2
       vsubps   zmm1, zmm1, zmm2
       lea      ebx, [rax+2*rax]
       lea      r15d, [4*rbx]
       movsxd   r15, r15d
       vsubps   zmm1, zmm1, zmmword ptr [r12+4*r15]
       movsxd   r15, eax
       vmovups  zmm2, zmmword ptr [r12+4*r15]
       lea      r15d, [rax+4*rax]
       movsxd   rdi, r15d
       vmovups  zmm3, zmmword ptr [r12+4*rdi]
       vaddps   zmm2, zmm2, zmm3
       lea      edi, [rax+8*rax]
       movsxd   rdi, edi
       vmovups  zmm4, zmmword ptr [r12+4*rdi]
       vaddps   zmm2, zmm2, zmm4
       vsubps   zmm3, zmm3, zmm4
       imul     edi, eax, 13
       movsxd   rdi, edi
       vsubps   zmm3, zmm3, zmmword ptr [r12+4*rdi]
       lea      edi, [rax+rax]
       movsxd   rdi, edi
       vmovups  zmm4, zmmword ptr [r12+4*rdi]
       lea      edi, [rbx+rbx]
       movsxd   rdi, edi
       vmovups  zmm5, zmmword ptr [r12+4*rdi]
       vaddps   zmm4, zmm4, zmm5
       add      r15d, r15d
       movsxd   rdi, r15d
       vmovups  zmm6, zmmword ptr [r12+4*rdi]
       vaddps   zmm4, zmm6, zmm4
       vsubps   zmm5, zmm5, zmm6
       imul     edi, eax, 14
       movsxd   rdi, edi
       vsubps   zmm5, zmm5, zmmword ptr [r12+4*rdi]
 
G_M000_IG08:                ;; offset=0x018C
       movsxd   rdi, ebx
       vmovups  zmm6, zmmword ptr [r12+4*rdi]
       lea      edi, [8*rax]
       sub      edi, eax
       movsxd   rdi, edi
       vmovups  zmm7, zmmword ptr [r12+4*rdi]
       vaddps   zmm6, zmm6, zmm7
       imul     edi, eax, 11
       movsxd   rdi, edi
       vmovups  zmm8, zmmword ptr [r12+4*rdi]
       vaddps   zmm6, zmm6, zmm8
       vsubps   zmm7, zmm7, zmm8
       mov      edi, eax
       shl      edi, 4
       sub      edi, eax
       movsxd   rax, edi
       vsubps   zmm7, zmm7, zmmword ptr [r12+4*rax]
       vaddps   zmm0, zmm2, zmm0
       vaddps   zmm0, zmm0, zmm4
       vsubps   zmm2, zmm2, zmm4
       vsubps   zmm2, zmm2, zmm6
       vaddps   zmm1, zmm1, zmm3
       vaddps   zmm1, zmm1, zmm5
       vsubps   zmm3, zmm3, zmm5
       vsubps   zmm3, zmm3, zmm7
       mov      edi, dword ptr [rbp-0x3C]
       mov      eax, edi
       imul     eax, r8d
       add      eax, edx
       shl      eax, 4
       cdqe     
       mov      r15d, dword ptr [rbp-0x34]
       mov      ebx, r14d
       imul     ebx, r15d
       movsxd   rbx, ebx
       shl      rbx, 2
       add      rbx, rsi
       lea      rax, [rbx+4*rax]
       vmovups  zmmword ptr [rax], zmm0
       inc      edx
       cmp      edx, r8d
       jge      SHORT G_M000_IG10
 
G_M000_IG09:                ;; offset=0x023F
       vmovups  zmmword ptr [rax+0x40], zmm2
 
G_M000_IG10:                ;; offset=0x0246
       inc      edi
       cmp      edi, ecx
       jge      SHORT G_M000_IG13
 
G_M000_IG11:                ;; offset=0x024C
       mov      edi, r8d
       shl      edi, 4
       movsxd   rdi, edi
       vmovups  zmmword ptr [rax+4*rdi], zmm1
       cmp      edx, r8d
       jge      SHORT G_M000_IG13
 
G_M000_IG12:                ;; offset=0x0261
       lea      edx, [r8+0x01]
       shl      edx, 4
       movsxd   rdx, edx
       vmovups  zmmword ptr [rax+4*rdx], zmm3
 
G_M000_IG13:                ;; offset=0x0272
       inc      r13d
       cmp      r13d, r11d
       jge      G_M000_IG05
 
G_M000_IG14:                ;; offset=0x027E
       mov      ebx, dword ptr [rbp+0x10]
       mov      rdi, qword ptr [rbp-0x30]
       jmp      G_M000_IG07
 
; Total bytes of code 650


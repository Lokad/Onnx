; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:Kernel256(ptr,ptr,ptr,int,int,int,int,int,int,int) (Tier1-OSR)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
; Tier1-OSR code
; OSR variant for entry point 0x1ec
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 1

G_M000_IG01:                ;; offset=0x0000
       mov      rax, qword ptr [rbp]
       push     rax
       sub      rsp, 64
       mov      qword ptr [rsp+0x448], r15
       mov      qword ptr [rsp+0x440], r14
       mov      qword ptr [rsp+0x438], r13
       mov      qword ptr [rsp+0x430], r12
       mov      qword ptr [rsp+0x428], rbx
       lea      rbp, [rsp+0x40]
       mov      rcx, qword ptr [rbp+0x3E0]
       mov      rdi, qword ptr [rbp+0x3D0]
       mov      edx, dword ptr [rbp+0x3CC]
       mov      esi, dword ptr [rbp+0x3C8]
       mov      eax, dword ptr [rbp+0x428]
       mov      r9d, dword ptr [rbp+0x438]
       mov      r12d, dword ptr [rbp+0x3BC]
       mov      r13d, dword ptr [rbp+0x3A8]
       vmovups  ymm0, ymmword ptr [rbp+0x380]
       vmovups  ymm6, ymmword ptr [rbp+0x360]
       vmovups  ymm1, ymmword ptr [rbp+0x340]
       vmovups  ymm7, ymmword ptr [rbp+0x320]
       vmovups  ymm2, ymmword ptr [rbp+0x300]
       vmovups  ymm8, ymmword ptr [rbp+0x2E0]
       vmovups  ymm3, ymmword ptr [rbp+0x2C0]
       vmovups  ymm9, ymmword ptr [rbp+0x2A0]
       vmovups  ymm4, ymmword ptr [rbp+0x280]
       vmovups  ymm10, ymmword ptr [rbp+0x260]
       vmovups  ymm5, ymmword ptr [rbp+0x240]
       vmovups  ymm11, ymmword ptr [rbp+0x220]
       mov      rbx, qword ptr [rbp+0x218]
       mov      r15, qword ptr [rbp+0x210]
       mov      r10d, dword ptr [rbp+0x20C]
       mov      r14d, dword ptr [rbp+0x208]
       mov      r11d, dword ptr [rbp+0x204]
 
G_M000_IG02:                ;; offset=0x00EE
       jmp      G_M000_IG07
 
G_M000_IG03:                ;; offset=0x00F3
       inc      r14d
       cmp      r14d, 3
       jge      G_M000_IG09
 
G_M000_IG04:                ;; offset=0x0100
       xor      r11d, r11d
       mov      dword ptr [rbp+0x3CC], edx
       mov      qword ptr [rbp+0x3D0], rdi
       mov      dword ptr [rbp+0x3C8], esi
       mov      dword ptr [rbp+0x438], r9d
 
G_M000_IG05:                ;; offset=0x011D
       vmovups  ymm12, ymmword ptr [rbx]
       vmovups  ymm13, ymmword ptr [r15]
       mov      r8d, r10d
       sar      r8d, 31
       and      r8d, 7
       add      r8d, r10d
       sar      r8d, 3
       mov      r9d, dword ptr [rbp+0x3C0]
       imul     r8d, r9d
       mov      esi, dword ptr [rbp+0x3AC]
       mov      edi, esi
       imul     edi, eax
       add      edi, r8d
       add      edi, r14d
       imul     edi, r12d
       mov      r8d, r13d
       imul     r8d, eax
       add      edi, r8d
       add      edi, r11d
       shl      edi, 3
       movsxd   rdi, edi
       shl      rdi, 2
       add      rdi, rcx
       mov      r8d, r10d
       sar      r8d, 31
       and      r8d, 7
       add      r8d, r10d
       and      r8d, -8
       mov      edx, r10d
       sub      edx, r8d
       movsxd   rdx, edx
       lea      rdx, [rdi+4*rdx]
       vbroadcastss ymm14, dword ptr [rdx]
       vfmadd231ps ymm0, ymm12, ymm14
       vfmadd231ps ymm6, ymm13, ymm14
       lea      edi, [8*rax]
       movsxd   rdi, edi
       vbroadcastss ymm14, dword ptr [rdx+4*rdi]
       vfmadd231ps ymm1, ymm12, ymm14
       vfmadd231ps ymm7, ymm13, ymm14
       lea      edi, [rax+rax]
       shl      edi, 3
       movsxd   rdi, edi
       vbroadcastss ymm14, dword ptr [rdx+4*rdi]
       vfmadd231ps ymm2, ymm12, ymm14
       vfmadd231ps ymm8, ymm13, ymm14
       lea      edi, [rax+2*rax]
       shl      edi, 3
       movsxd   rdi, edi
       vbroadcastss ymm14, dword ptr [rdx+4*rdi]
       vfmadd231ps ymm3, ymm12, ymm14
       vfmadd231ps ymm9, ymm13, ymm14
       lea      edi, [4*rax]
       shl      edi, 3
       movsxd   rdi, edi
       vbroadcastss ymm14, dword ptr [rdx+4*rdi]
       vfmadd231ps ymm4, ymm12, ymm14
       vfmadd231ps ymm10, ymm13, ymm14
       lea      edi, [rax+4*rax]
       shl      edi, 3
       movsxd   rdi, edi
       vbroadcastss ymm14, dword ptr [rdx+4*rdi]
       vfmadd231ps ymm5, ymm12, ymm14
       vfmadd231ps ymm11, ymm13, ymm14
       add      rbx, 32
       add      r15, 32
       inc      r11d
       mov      dword ptr [rbp+0x3AC], esi
       mov      dword ptr [rbp+0x3C0], r9d
 
G_M000_IG06:                ;; offset=0x023A
       mov      edx, dword ptr [rbp+0x3CC]
       mov      esi, dword ptr [rbp+0x3C8]
       mov      rdi, qword ptr [rbp+0x3D0]
       mov      r9d, dword ptr [rbp+0x438]
 
G_M000_IG07:                ;; offset=0x0254
       cmp      r11d, 3
       jge      G_M000_IG03
 
G_M000_IG08:                ;; offset=0x025E
       mov      dword ptr [rbp+0x3CC], edx
       mov      qword ptr [rbp+0x3D0], rdi
       mov      dword ptr [rbp+0x3C8], esi
       mov      dword ptr [rbp+0x438], r9d
       jmp      G_M000_IG05
 
G_M000_IG09:                ;; offset=0x027D
       inc      r10d
 
G_M000_IG10:                ;; offset=0x0280
       cmp      r10d, edx
       jge      SHORT G_M000_IG12
 
G_M000_IG11:                ;; offset=0x0285
       xor      r11d, r11d
       mov      r14d, r11d
       jmp      G_M000_IG04
 
G_M000_IG12:                ;; offset=0x0290
       mov      r10d, dword ptr [rbp+0x3B0]
       mov      r11d, r10d
       sar      r11d, 31
       and      r11d, 7
       mov      dword ptr [rbp+0x3B0], r10d
       add      r11d, r10d
       sar      r11d, 3
       mov      r14d, dword ptr [rbp+0x3B8]
       mov      ebx, r11d
       imul     ebx, r14d
       mov      r15d, dword ptr [rbp+0x3AC]
       mov      dword ptr [rbp+0x438], r9d
       mov      r8d, r15d
       imul     r8d, r9d
       add      ebx, r8d
       add      ebx, r13d
       shl      ebx, 3
       movsxd   r10, ebx
       vmovups  ymmword ptr [rdi+4*r10], ymm0
       mov      r10d, dword ptr [rbp+0x3B0]
       add      r10d, 8
       cmp      r10d, esi
       jge      SHORT G_M000_IG14
 
G_M000_IG13:                ;; offset=0x02F5
       lea      r9d, [r11+0x01]
       imul     r9d, r14d
       add      r9d, r8d
       add      r9d, r13d
       shl      r9d, 3
       movsxd   r9, r9d
       vmovups  ymmword ptr [rdi+4*r9], ymm6
 
G_M000_IG14:                ;; offset=0x0310
       lea      r9d, [rbx+0x08]
       movsxd   r9, r9d
       vmovups  ymmword ptr [rdi+4*r9], ymm1
       cmp      r10d, esi
       jge      SHORT G_M000_IG16
 
G_M000_IG15:                ;; offset=0x0322
       lea      r9d, [r11+0x01]
       imul     r9d, r14d
       add      r9d, r8d
       add      r9d, r13d
       lea      r9d, [8*r9+0x08]
       movsxd   r9, r9d
       vmovups  ymmword ptr [rdi+4*r9], ymm7
 
G_M000_IG16:                ;; offset=0x0341
       lea      r9d, [rbx+0x10]
       movsxd   r9, r9d
       vmovups  ymmword ptr [rdi+4*r9], ymm2
       cmp      r10d, esi
       jge      SHORT G_M000_IG18
 
G_M000_IG17:                ;; offset=0x0353
       lea      r9d, [r11+0x01]
       imul     r9d, r14d
       add      r9d, r8d
       add      r9d, r13d
       lea      r9d, [8*r9+0x10]
       movsxd   r9, r9d
       vmovups  ymmword ptr [rdi+4*r9], ymm8
 
G_M000_IG18:                ;; offset=0x0372
       lea      r9d, [rbx+0x18]
       movsxd   r9, r9d
       vmovups  ymmword ptr [rdi+4*r9], ymm3
       cmp      r10d, esi
       jge      SHORT G_M000_IG20
 
G_M000_IG19:                ;; offset=0x0384
       lea      r9d, [r11+0x01]
       imul     r9d, r14d
       add      r9d, r8d
       add      r9d, r13d
       lea      r9d, [8*r9+0x18]
       movsxd   r9, r9d
       vmovups  ymmword ptr [rdi+4*r9], ymm9
 
G_M000_IG20:                ;; offset=0x03A3
       lea      r9d, [rbx+0x20]
       movsxd   r9, r9d
       vmovups  ymmword ptr [rdi+4*r9], ymm4
       cmp      r10d, esi
       jge      SHORT G_M000_IG22
 
G_M000_IG21:                ;; offset=0x03B5
       lea      r9d, [r11+0x01]
       imul     r9d, r14d
       add      r9d, r8d
       add      r9d, r13d
       lea      r9d, [8*r9+0x20]
       movsxd   r9, r9d
       vmovups  ymmword ptr [rdi+4*r9], ymm10
 
G_M000_IG22:                ;; offset=0x03D4
       add      ebx, 40
       movsxd   r9, ebx
       vmovups  ymmword ptr [rdi+4*r9], ymm5
       mov      dword ptr [rbp+0x3C8], esi
       cmp      r10d, esi
       jge      SHORT G_M000_IG24
 
G_M000_IG23:                ;; offset=0x03EB
       inc      r11d
       mov      dword ptr [rbp+0x3B8], r14d
       imul     r11d, r14d
       add      r8d, r11d
       add      r8d, r13d
       lea      r8d, [8*r8+0x28]
       movsxd   r8, r8d
       mov      qword ptr [rbp+0x3D0], rdi
       vmovups  ymmword ptr [rdi+4*r8], ymm11
       mov      rdi, qword ptr [rbp+0x3D0]
       mov      r14d, dword ptr [rbp+0x3B8]
 
G_M000_IG24:                ;; offset=0x0425
       add      r13d, 6
 
G_M000_IG25:                ;; offset=0x0429
       lea      r8d, [r13+0x06]
       mov      r9d, dword ptr [rbp+0x438]
       cmp      r8d, r9d
       jg       G_M000_IG41
 
G_M000_IG26:                ;; offset=0x043D
       mov      r8d, r15d
       imul     r8d, r9d
       lea      r8d, [r8+r13+0x06]
       cmp      r8d, dword ptr [rbp+0x3B4]
       jg       G_M000_IG41
 
G_M000_IG27:                ;; offset=0x0456
       mov      dword ptr [rbp+0x438], r9d
       vxorps   ymm0, ymm0, ymm0
       vxorps   ymm6, ymm6, ymm6
       vxorps   ymm1, ymm1, ymm1
       vxorps   ymm7, ymm7, ymm7
       vxorps   ymm2, ymm2, ymm2
       vxorps   ymm8, ymm8, ymm8
       vxorps   ymm3, ymm3, ymm3
       vxorps   ymm9, ymm9, ymm9
       vxorps   ymm4, ymm4, ymm4
       vxorps   ymm10, ymm10, ymm10
       vxorps   ymm5, ymm5, ymm5
       vxorps   ymm11, ymm11, ymm11
       mov      ebx, dword ptr [rbp+0x3B0]
       mov      r11d, ebx
       imul     r11d, edx
       lea      r11d, [r11+8*r11]
       movsxd   r11, r11d
       mov      r10, qword ptr [rbp+0x3D8]
       lea      r11, [r10+4*r11]
       mov      dword ptr [rbp+0x3CC], edx
       lea      r10d, [8*rdx]
       lea      r10d, [r10+8*r10]
       movsxd   r10, r10d
       lea      r10, [r11+4*r10]
       xor      r9d, r9d
       mov      dword ptr [rbp+0x3AC], r15d
       mov      dword ptr [rbp+0x3B8], r14d
       mov      dword ptr [rbp+0x3B0], ebx
       mov      rbx, r11
       mov      r15, r10
       mov      r10d, r9d
       mov      edx, dword ptr [rbp+0x3CC]
       mov      esi, dword ptr [rbp+0x3C8]
       mov      r9d, dword ptr [rbp+0x438]
       jmp      G_M000_IG10
       align    [0 bytes for IG28]
 
G_M000_IG28:                ;; offset=0x0501
       vfmadd231ps ymm0, ymm7, ymm1
       vfmadd231ps ymm6, ymm2, ymm1
 
G_M000_IG29:                ;; offset=0x050B
       add      r11, 32
       add      r10, 32
       lea      edi, [rsi+0x01]
       lea      edi, [rbx+8*rdi]
       movsxd   rdi, edi
       vbroadcastss ymm1, dword ptr [rcx+4*rdi]
       vmovups  ymm7, ymmword ptr [r11]
       vmovups  ymm2, ymmword ptr [r10]
       mov      edi, dword ptr [rbp+0x16C]
       test     edi, edi
       je       G_M000_IG36
 
G_M000_IG30:                ;; offset=0x053A
       vfmadd231ps ymm0, ymm7, ymm1
       vfmadd231ps ymm6, ymm2, ymm1
 
G_M000_IG31:                ;; offset=0x0544
       add      r11, 32
       add      r10, 32
       add      esi, 2
       lea      esi, [rbx+8*rsi]
       movsxd   rsi, esi
       vbroadcastss ymm1, dword ptr [rcx+4*rsi]
       vmovups  ymm7, ymmword ptr [r11]
       vmovups  ymm2, ymmword ptr [r10]
       test     edi, edi
       je       G_M000_IG37
 
G_M000_IG32:                ;; offset=0x056D
       vfmadd231ps ymm0, ymm7, ymm1
       vfmadd231ps ymm6, ymm2, ymm1
 
G_M000_IG33:                ;; offset=0x0577
       add      r11, 32
       add      r10, 32
       inc      r14d
       cmp      r14d, 3
       mov      esi, dword ptr [rbp-0x2C]
       jge      G_M000_IG45
 
G_M000_IG34:                ;; offset=0x058F
       add      esi, r14d
       imul     esi, r12d
       add      esi, dword ptr [rbp-0x34]
       mov      edi, r9d
       sar      edi, 31
       and      edi, 7
       add      edi, r9d
       and      edi, -8
       mov      ebx, r9d
       sub      ebx, edi
       lea      edi, [rbx+8*rsi]
       movsxd   rdi, edi
       vbroadcastss ymm1, dword ptr [rcx+4*rdi]
       vmovups  ymm7, ymmword ptr [r11]
       vmovups  ymm2, ymmword ptr [r10]
       cmp      dword ptr [rbp+0x16C], 0
       jne      G_M000_IG28
 
G_M000_IG35:                ;; offset=0x05D0
       vmulps   ymm7, ymm1, ymm7
       vaddps   ymm0, ymm7, ymm0
       vmulps   ymm1, ymm1, ymm2
       vaddps   ymm6, ymm1, ymm6
       jmp      G_M000_IG29
 
G_M000_IG36:                ;; offset=0x05E5
       vmulps   ymm7, ymm1, ymm7
       vaddps   ymm0, ymm7, ymm0
       vmulps   ymm1, ymm1, ymm2
       vaddps   ymm6, ymm1, ymm6
       jmp      G_M000_IG31
 
G_M000_IG37:                ;; offset=0x05FA
       vmulps   ymm8, ymm1, ymm7
       vaddps   ymm0, ymm8, ymm0
       vmulps   ymm3, ymm1, ymm2
       vaddps   ymm6, ymm3, ymm6
       jmp      G_M000_IG33
 
G_M000_IG38:                ;; offset=0x060F
       mov      r10d, dword ptr [rbp+0x3B0]
       mov      esi, r10d
       sar      esi, 31
       and      esi, 7
       add      esi, r10d
       sar      esi, 3
       mov      r14d, dword ptr [rbp+0x3B8]
       mov      r9d, esi
       imul     r9d, r14d
       mov      r11d, dword ptr [rbp+0x438]
       mov      ebx, r15d
       imul     ebx, r11d
       add      r9d, ebx
       add      r9d, r13d
       shl      r9d, 3
       movsxd   r9, r9d
       vmovups  ymmword ptr [rdi+4*r9], ymm0
       lea      r9d, [r10+0x08]
       mov      ebx, dword ptr [rbp+0x3C8]
       cmp      r9d, ebx
       jge      SHORT G_M000_IG40
 
G_M000_IG39:                ;; offset=0x0663
       inc      esi
       imul     esi, r14d
       mov      dword ptr [rbp+0x438], r11d
       mov      r9d, r15d
       imul     r9d, r11d
       add      esi, r9d
       add      esi, r13d
       shl      esi, 3
       movsxd   rsi, esi
       mov      qword ptr [rbp+0x3D0], rdi
       vmovups  ymmword ptr [rdi+4*rsi], ymm6
       mov      rdi, qword ptr [rbp+0x3D0]
       mov      r11d, dword ptr [rbp+0x438]
 
G_M000_IG40:                ;; offset=0x069D
       inc      r13d
       mov      dword ptr [rbp+0x3C0], r8d
       mov      dword ptr [rbp+0x3C8], ebx
       mov      dword ptr [rbp+0x3B0], r10d
       mov      r9d, r11d
 
G_M000_IG41:                ;; offset=0x06B7
       cmp      r13d, r9d
       jge      G_M000_IG43
 
G_M000_IG42:                ;; offset=0x06C0
       vxorps   ymm0, ymm0, ymm0
       vxorps   ymm6, ymm6, ymm6
       mov      r11d, edx
       imul     r11d, dword ptr [rbp+0x3B0]
       lea      r11d, [r11+8*r11]
       movsxd   r11, r11d
       mov      rbx, qword ptr [rbp+0x3D8]
       lea      r11, [rbx+4*r11]
       lea      r10d, [8*rdx]
       lea      r10d, [r10+8*r10]
       movsxd   r10, r10d
       lea      r10, [r11+4*r10]
       mov      dword ptr [rbp+0x438], r9d
       mov      ebx, r15d
       imul     ebx, r9d
       add      ebx, r13d
       cmp      ebx, dword ptr [rbp+0x3B4]
       setl     bl
       movzx    rbx, bl
       mov      dword ptr [rbp+0x16C], ebx
       xor      r9d, r9d
       mov      r8d, r15d
       imul     r8d, eax
       mov      dword ptr [rbp-0x30], r8d
       mov      r8d, r13d
       imul     r8d, eax
       mov      dword ptr [rbp-0x34], r8d
       cmp      r9d, edx
       mov      dword ptr [rbp+0x3B8], r14d
       jl       SHORT G_M000_IG47
       jmp      G_M000_IG50
 
G_M000_IG43:                ;; offset=0x0745
       inc      r15d
       mov      r11d, dword ptr [rbp+0x430]
       cmp      r15d, r11d
       jge      G_M000_IG51
 
G_M000_IG44:                ;; offset=0x0758
       xor      r13d, r13d
       mov      dword ptr [rbp+0x438], r9d
       mov      dword ptr [rbp+0x430], r11d
       jmp      G_M000_IG25
 
G_M000_IG45:                ;; offset=0x076E
       mov      rdi, qword ptr [rbp+0x3D0]
       inc      r9d
       cmp      r9d, edx
       jge      G_M000_IG38
 
G_M000_IG46:                ;; offset=0x0781
       mov      dword ptr [rbp+0x3C0], r8d
 
G_M000_IG47:                ;; offset=0x0788
       xor      r14d, r14d
       mov      esi, r9d
       sar      esi, 31
       and      esi, 7
       add      esi, r9d
       sar      esi, 3
       mov      r8d, dword ptr [rbp+0x3C0]
       imul     esi, r8d
       add      esi, dword ptr [rbp-0x30]
       mov      dword ptr [rbp-0x2C], esi
       mov      qword ptr [rbp+0x3D0], rdi
       jmp      G_M000_IG34
 
G_M000_IG48:                ;; offset=0x07B7
       xor      r13d, r13d
       mov      dword ptr [rbp+0x430], r11d
       test     r11d, r11d
       mov      dword ptr [rbp+0x3B0], r15d
       mov      r11d, dword ptr [rbp+0x430]
       jle      SHORT G_M000_IG51
 
G_M000_IG49:                ;; offset=0x07D4
       mov      r15d, r13d
       jmp      G_M000_IG44
 
G_M000_IG50:                ;; offset=0x07DC
       mov      r8d, dword ptr [rbp+0x3C0]
       jmp      G_M000_IG38
 
G_M000_IG51:                ;; offset=0x07E8
       mov      r15d, dword ptr [rbp+0x3B0]
       add      r15d, 16
       cmp      r15d, dword ptr [rbp+0x3C8]
       jl       SHORT G_M000_IG48
 
G_M000_IG52:                ;; offset=0x07FC
       vzeroupper 
       add      rsp, 0x428
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
; Total bytes of code 2065


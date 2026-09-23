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
       mov      r8d, dword ptr [rbp+0x438]
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
       jmp      G_M000_IG10
 
G_M000_IG03:                ;; offset=0x00F3
       inc      r10d
 
G_M000_IG04:                ;; offset=0x00F6
       cmp      r10d, edx
       jge      G_M000_IG12
 
G_M000_IG05:                ;; offset=0x00FF
       xor      r9d, r9d
       mov      r14d, r9d
       jmp      SHORT G_M000_IG07
 
G_M000_IG06:                ;; offset=0x0107
       inc      r14d
       cmp      r14d, 3
       jge      SHORT G_M000_IG03
 
G_M000_IG07:                ;; offset=0x0110
       xor      r9d, r9d
       mov      dword ptr [rbp+0x3CC], edx
       mov      qword ptr [rbp+0x3D0], rdi
       mov      dword ptr [rbp+0x3C8], esi
       mov      dword ptr [rbp+0x438], r8d
       mov      r11d, r9d
 
G_M000_IG08:                ;; offset=0x0130
       vmovups  ymm12, ymmword ptr [rbx]
       vmovups  ymm13, ymmword ptr [r15]
       mov      r9d, r10d
       sar      r9d, 31
       and      r9d, 7
       add      r9d, r10d
       sar      r9d, 3
       mov      r8d, dword ptr [rbp+0x3C0]
       imul     r9d, r8d
       mov      esi, dword ptr [rbp+0x3AC]
       mov      edi, esi
       imul     edi, eax
       add      edi, r9d
       add      edi, r14d
       imul     edi, r12d
       mov      r9d, r13d
       imul     r9d, eax
       add      edi, r9d
       add      edi, r11d
       shl      edi, 3
       movsxd   rdi, edi
       shl      rdi, 2
       add      rdi, rcx
       mov      r9d, r10d
       sar      r9d, 31
       and      r9d, 7
       add      r9d, r10d
       and      r9d, -8
       mov      edx, r10d
       sub      edx, r9d
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
       mov      dword ptr [rbp+0x3C0], r8d
 
G_M000_IG09:                ;; offset=0x024D
       mov      edx, dword ptr [rbp+0x3CC]
       mov      esi, dword ptr [rbp+0x3C8]
       mov      rdi, qword ptr [rbp+0x3D0]
       mov      r8d, dword ptr [rbp+0x438]
 
G_M000_IG10:                ;; offset=0x0267
       cmp      r11d, 3
       jge      G_M000_IG06
 
G_M000_IG11:                ;; offset=0x0271
       mov      dword ptr [rbp+0x3CC], edx
       mov      qword ptr [rbp+0x3D0], rdi
       mov      dword ptr [rbp+0x3C8], esi
       mov      dword ptr [rbp+0x438], r8d
       jmp      G_M000_IG08
 
G_M000_IG12:                ;; offset=0x0290
       mov      r10d, dword ptr [rbp+0x3B0]
       mov      r11d, r10d
       sar      r11d, 31
       and      r11d, 7
       add      r11d, r10d
       sar      r11d, 3
       mov      r14d, dword ptr [rbp+0x3B8]
       mov      ebx, r11d
       imul     ebx, r14d
       mov      r15d, dword ptr [rbp+0x3AC]
       mov      dword ptr [rbp+0x438], r8d
       mov      r9d, r15d
       imul     r9d, r8d
       add      ebx, r9d
       add      ebx, r13d
       shl      ebx, 3
       movsxd   r8, ebx
       vmovups  ymmword ptr [rdi+4*r8], ymm0
       mov      dword ptr [rbp+0x3B0], r10d
       lea      r8d, [r10+0x08]
       cmp      r8d, esi
       jge      SHORT G_M000_IG14
 
G_M000_IG13:                ;; offset=0x02EE
       lea      r10d, [r11+0x01]
       imul     r10d, r14d
       add      r10d, r9d
       add      r10d, r13d
       shl      r10d, 3
       movsxd   r10, r10d
       vmovups  ymmword ptr [rdi+4*r10], ymm6
 
G_M000_IG14:                ;; offset=0x0309
       lea      r10d, [rbx+0x08]
       movsxd   r10, r10d
       vmovups  ymmword ptr [rdi+4*r10], ymm1
       cmp      r8d, esi
       jge      SHORT G_M000_IG16
 
G_M000_IG15:                ;; offset=0x031B
       lea      r10d, [r11+0x01]
       imul     r10d, r14d
       add      r10d, r9d
       add      r10d, r13d
       lea      r10d, [8*r10+0x08]
       movsxd   r10, r10d
       vmovups  ymmword ptr [rdi+4*r10], ymm7
 
G_M000_IG16:                ;; offset=0x033A
       lea      r10d, [rbx+0x10]
       movsxd   r10, r10d
       vmovups  ymmword ptr [rdi+4*r10], ymm2
       cmp      r8d, esi
       jge      SHORT G_M000_IG18
 
G_M000_IG17:                ;; offset=0x034C
       lea      r10d, [r11+0x01]
       imul     r10d, r14d
       add      r10d, r9d
       add      r10d, r13d
       lea      r10d, [8*r10+0x10]
       movsxd   r10, r10d
       vmovups  ymmword ptr [rdi+4*r10], ymm8
 
G_M000_IG18:                ;; offset=0x036B
       lea      r10d, [rbx+0x18]
       movsxd   r10, r10d
       vmovups  ymmword ptr [rdi+4*r10], ymm3
       cmp      r8d, esi
       jge      SHORT G_M000_IG20
 
G_M000_IG19:                ;; offset=0x037D
       lea      r10d, [r11+0x01]
       imul     r10d, r14d
       add      r10d, r9d
       add      r10d, r13d
       lea      r10d, [8*r10+0x18]
       movsxd   r10, r10d
       vmovups  ymmword ptr [rdi+4*r10], ymm9
 
G_M000_IG20:                ;; offset=0x039C
       lea      r10d, [rbx+0x20]
       movsxd   r10, r10d
       vmovups  ymmword ptr [rdi+4*r10], ymm4
       cmp      r8d, esi
       jge      SHORT G_M000_IG22
 
G_M000_IG21:                ;; offset=0x03AE
       lea      r10d, [r11+0x01]
       imul     r10d, r14d
       add      r10d, r9d
       add      r10d, r13d
       lea      r10d, [8*r10+0x20]
       movsxd   r10, r10d
       vmovups  ymmword ptr [rdi+4*r10], ymm10
 
G_M000_IG22:                ;; offset=0x03CD
       add      ebx, 40
       movsxd   r10, ebx
       vmovups  ymmword ptr [rdi+4*r10], ymm5
       mov      dword ptr [rbp+0x3C8], esi
       cmp      r8d, esi
       jge      SHORT G_M000_IG24
 
G_M000_IG23:                ;; offset=0x03E4
       inc      r11d
       mov      dword ptr [rbp+0x3B8], r14d
       imul     r11d, r14d
       add      r9d, r11d
       add      r9d, r13d
       lea      r8d, [8*r9+0x28]
       movsxd   r8, r8d
       mov      qword ptr [rbp+0x3D0], rdi
       vmovups  ymmword ptr [rdi+4*r8], ymm11
       mov      rdi, qword ptr [rbp+0x3D0]
       mov      r14d, dword ptr [rbp+0x3B8]
 
G_M000_IG24:                ;; offset=0x041E
       add      r13d, 6
 
G_M000_IG25:                ;; offset=0x0422
       lea      r8d, [r13+0x06]
       mov      r9d, dword ptr [rbp+0x438]
       cmp      r8d, r9d
       jg       G_M000_IG28
 
G_M000_IG26:                ;; offset=0x0436
       mov      r8d, r15d
       imul     r8d, r9d
       lea      r8d, [r8+r13+0x06]
       cmp      r8d, dword ptr [rbp+0x3B4]
       jg       G_M000_IG28
 
G_M000_IG27:                ;; offset=0x044F
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
       mov      r8d, ebx
       imul     r8d, edx
       lea      r8d, [r8+8*r8]
       movsxd   r8, r8d
       mov      r10, qword ptr [rbp+0x3D8]
       lea      r8, [r10+4*r8]
       mov      dword ptr [rbp+0x3CC], edx
       lea      r10d, [8*rdx]
       lea      r10d, [r10+8*r10]
       movsxd   r10, r10d
       lea      r10, [r8+4*r10]
       xor      r9d, r9d
       mov      dword ptr [rbp+0x3AC], r15d
       mov      dword ptr [rbp+0x3B8], r14d
       mov      dword ptr [rbp+0x3B0], ebx
       mov      rbx, r8
       mov      r15, r10
       mov      r10d, r9d
       mov      edx, dword ptr [rbp+0x3CC]
       mov      esi, dword ptr [rbp+0x3C8]
       mov      r8d, dword ptr [rbp+0x438]
       jmp      G_M000_IG04
 
G_M000_IG28:                ;; offset=0x04FA
       mov      r8d, r15d
       imul     r8d, eax
       mov      dword ptr [rbp-0x30], r8d
       jmp      G_M000_IG35
       align    [0 bytes for IG29]
 
G_M000_IG29:                ;; offset=0x050A
       vmulps   ymm7, ymm1, ymm7
       vaddps   ymm0, ymm7, ymm0
       vmulps   ymm1, ymm1, ymm2
       vaddps   ymm6, ymm1, ymm6
       jmp      G_M000_IG43
 
G_M000_IG30:                ;; offset=0x051F
       vmulps   ymm8, ymm1, ymm7
       vaddps   ymm0, ymm8, ymm0
       vmulps   ymm3, ymm1, ymm2
       vaddps   ymm6, ymm3, ymm6
       jmp      G_M000_IG45
 
G_M000_IG31:                ;; offset=0x0534
       mov      r11d, dword ptr [rbp+0x3C0]
 
G_M000_IG32:                ;; offset=0x053B
       mov      r10d, dword ptr [rbp+0x3B0]
       mov      esi, r10d
       sar      esi, 31
       and      esi, 7
       add      esi, r10d
       sar      esi, 3
       mov      r14d, dword ptr [rbp+0x3B8]
       mov      r8d, esi
       imul     r8d, r14d
       mov      r9d, dword ptr [rbp+0x438]
       mov      ebx, r15d
       imul     ebx, r9d
       add      r8d, ebx
       add      r8d, r13d
       shl      r8d, 3
       movsxd   r8, r8d
       vmovups  ymmword ptr [rdi+4*r8], ymm0
       lea      r8d, [r10+0x08]
       mov      ebx, dword ptr [rbp+0x3C8]
       cmp      r8d, ebx
       jge      SHORT G_M000_IG34
 
G_M000_IG33:                ;; offset=0x058F
       inc      esi
       imul     esi, r14d
       mov      dword ptr [rbp+0x438], r9d
       mov      r8d, r15d
       imul     r8d, r9d
       add      esi, r8d
       add      esi, r13d
       shl      esi, 3
       movsxd   rsi, esi
       mov      qword ptr [rbp+0x3D0], rdi
       vmovups  ymmword ptr [rdi+4*rsi], ymm6
       mov      rdi, qword ptr [rbp+0x3D0]
       mov      r9d, dword ptr [rbp+0x438]
 
G_M000_IG34:                ;; offset=0x05C9
       inc      r13d
       mov      dword ptr [rbp+0x3C0], r11d
       mov      dword ptr [rbp+0x3C8], ebx
       mov      dword ptr [rbp+0x3B0], r10d
 
G_M000_IG35:                ;; offset=0x05E0
       cmp      r13d, r9d
       jge      G_M000_IG48
 
G_M000_IG36:                ;; offset=0x05E9
       vxorps   ymm0, ymm0, ymm0
       vxorps   ymm6, ymm6, ymm6
       mov      ebx, edx
       imul     ebx, dword ptr [rbp+0x3B0]
       lea      ebx, [rbx+8*rbx]
       movsxd   rbx, ebx
       mov      r10, qword ptr [rbp+0x3D8]
       lea      rbx, [r10+4*rbx]
       lea      r8d, [8*rdx]
       lea      r8d, [r8+8*r8]
       movsxd   r8, r8d
       lea      r8, [rbx+4*r8]
       mov      dword ptr [rbp+0x438], r9d
       mov      r10d, r15d
       imul     r10d, r9d
       add      r10d, r13d
       cmp      r10d, dword ptr [rbp+0x3B4]
       setl     r10b
       movzx    r10, r10b
       mov      dword ptr [rbp+0x16C], r10d
       xor      r9d, r9d
       mov      r11d, r13d
       imul     r11d, eax
       mov      dword ptr [rbp-0x34], r11d
       cmp      r9d, edx
       mov      dword ptr [rbp+0x3B8], r14d
       jl       SHORT G_M000_IG39
       jmp      G_M000_IG31
 
G_M000_IG37:                ;; offset=0x0664
       mov      rdi, qword ptr [rbp+0x3D0]
       inc      r9d
       cmp      r9d, edx
       jge      G_M000_IG32
 
G_M000_IG38:                ;; offset=0x0677
       mov      dword ptr [rbp+0x3C0], r11d
 
G_M000_IG39:                ;; offset=0x067E
       xor      r14d, r14d
       mov      esi, r9d
       sar      esi, 31
       and      esi, 7
       add      esi, r9d
       sar      esi, 3
       mov      r11d, dword ptr [rbp+0x3C0]
       imul     esi, r11d
       add      esi, dword ptr [rbp-0x30]
       mov      dword ptr [rbp-0x2C], esi
       mov      qword ptr [rbp+0x3D0], rdi
       jmp      G_M000_IG46
 
G_M000_IG40:                ;; offset=0x06AD
       vmulps   ymm7, ymm1, ymm7
       vaddps   ymm0, ymm7, ymm0
       vmulps   ymm1, ymm1, ymm2
       vaddps   ymm6, ymm1, ymm6
 
G_M000_IG41:                ;; offset=0x06BD
       add      rbx, 32
       add      r8, 32
       lea      edi, [rsi+0x01]
       lea      edi, [r10+8*rdi]
       movsxd   rdi, edi
       vbroadcastss ymm1, dword ptr [rcx+4*rdi]
       vmovups  ymm7, ymmword ptr [rbx]
       vmovups  ymm2, ymmword ptr [r8]
       mov      edi, dword ptr [rbp+0x16C]
       test     edi, edi
       je       G_M000_IG29
 
G_M000_IG42:                ;; offset=0x06EC
       vfmadd231ps ymm0, ymm7, ymm1
       vfmadd231ps ymm6, ymm2, ymm1
 
G_M000_IG43:                ;; offset=0x06F6
       add      rbx, 32
       add      r8, 32
       add      esi, 2
       lea      esi, [r10+8*rsi]
       movsxd   rsi, esi
       vbroadcastss ymm1, dword ptr [rcx+4*rsi]
       vmovups  ymm7, ymmword ptr [rbx]
       vmovups  ymm2, ymmword ptr [r8]
       test     edi, edi
       je       G_M000_IG30
 
G_M000_IG44:                ;; offset=0x071F
       vfmadd231ps ymm0, ymm7, ymm1
       vfmadd231ps ymm6, ymm2, ymm1
 
G_M000_IG45:                ;; offset=0x0729
       add      rbx, 32
       add      r8, 32
       inc      r14d
       cmp      r14d, 3
       mov      esi, dword ptr [rbp-0x2C]
       jge      G_M000_IG37
 
G_M000_IG46:                ;; offset=0x0741
       add      esi, r14d
       imul     esi, r12d
       add      esi, dword ptr [rbp-0x34]
       mov      edi, r9d
       sar      edi, 31
       and      edi, 7
       add      edi, r9d
       and      edi, -8
       mov      r10d, r9d
       sub      r10d, edi
       lea      edi, [r10+8*rsi]
       movsxd   rdi, edi
       vbroadcastss ymm1, dword ptr [rcx+4*rdi]
       vmovups  ymm7, ymmword ptr [rbx]
       vmovups  ymm2, ymmword ptr [r8]
       cmp      dword ptr [rbp+0x16C], 0
       je       G_M000_IG40
 
G_M000_IG47:                ;; offset=0x0783
       vfmadd231ps ymm0, ymm7, ymm1
       vfmadd231ps ymm6, ymm2, ymm1
       jmp      G_M000_IG41
 
G_M000_IG48:                ;; offset=0x0792
       inc      r15d
       mov      r8d, dword ptr [rbp+0x430]
       cmp      r15d, r8d
       jge      SHORT G_M000_IG52
 
G_M000_IG49:                ;; offset=0x07A1
       xor      r13d, r13d
       mov      dword ptr [rbp+0x438], r9d
       mov      dword ptr [rbp+0x430], r8d
       jmp      G_M000_IG25
 
G_M000_IG50:                ;; offset=0x07B7
       xor      r13d, r13d
       mov      dword ptr [rbp+0x430], r8d
       test     r8d, r8d
       mov      dword ptr [rbp+0x3B0], r15d
       mov      r8d, dword ptr [rbp+0x430]
       jle      SHORT G_M000_IG52
 
G_M000_IG51:                ;; offset=0x07D4
       mov      r15d, r13d
       jmp      SHORT G_M000_IG49
 
G_M000_IG52:                ;; offset=0x07D9
       mov      r15d, dword ptr [rbp+0x3B0]
       add      r15d, 16
       cmp      r15d, dword ptr [rbp+0x3C8]
       jl       SHORT G_M000_IG50
 
G_M000_IG53:                ;; offset=0x07ED
       vzeroupper 
       add      rsp, 0x428
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
; Total bytes of code 2050


; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:Kernel256(ptr,ptr,ptr,int,int,int,int,int,int,int) (Tier1)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
; Tier1 code
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 100

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       push     r15
       push     r14
       push     r13
       push     r12
       push     rbx
       sub      rsp, 136
       lea      rbp, [rsp+0xB0]
       mov      qword ptr [rbp-0x30], rsi
       mov      qword ptr [rbp-0x38], rdx
       mov      eax, dword ptr [rbp+0x18]
       mov      r11d, dword ptr [rbp+0x20]
       mov      r10d, dword ptr [rbp+0x28]
 
G_M000_IG02:                ;; offset=0x002C
       add      r9d, 2
       mov      dword ptr [rbp-0x44], r9d
       mov      ebx, dword ptr [rbp+0x10]
       add      ebx, 2
       mov      dword ptr [rbp-0x48], ebx
       mov      dword ptr [rbp+0x20], r11d
       mov      dword ptr [rbp+0x28], r10d
       mov      r15d, r11d
       imul     r15d, r10d
       mov      dword ptr [rbp-0x4C], r15d
       mov      r14d, r15d
       sar      r14d, 31
       and      r14d, 7
       add      r14d, r15d
       sar      r14d, 3
       shl      r14d, 3
       mov      dword ptr [rbp-0x50], r14d
       xor      r13d, r13d
       lea      r12d, [8*rcx]
       lea      r12d, [r12+8*r12]
       movsxd   r12, r12d
       shl      r12, 2
       mov      qword ptr [rbp-0x90], r12
       lea      esi, [rax+rax]
       shl      esi, 3
       movsxd   rsi, esi
       lea      r11d, [rax+2*rax]
       shl      r11d, 3
       movsxd   r11, r11d
       lea      r12d, [4*rax]
       shl      r12d, 3
       movsxd   r12, r12d
       lea      r14d, [rax+4*rax]
       shl      r14d, 3
       movsxd   r14, r14d
       mov      dword ptr [rbp-0x40], r8d
       cmp      r13d, r8d
       jl       G_M000_IG19
 
G_M000_IG03:                ;; offset=0x00C2
       vzeroupper 
       add      rsp, 136
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
G_M000_IG04:                ;; offset=0x00D7
       align    [0 bytes for IG05]
 
G_M000_IG05:                ;; offset=0x00D7
       vfmadd231ps ymm0, ymm3, ymm2
       vfmadd231ps ymm1, ymm4, ymm2
 
G_M000_IG06:                ;; offset=0x00E1
       add      r15, 32
       add      rdx, 32
       mov      r8d, dword ptr [rbp-0x70]
       inc      r8d
       lea      r8d, [r13+8*r8]
       movsxd   r8, r8d
       vbroadcastss ymm2, dword ptr [rdi+4*r8]
       vmovups  ymm3, ymmword ptr [r15]
       vmovups  ymm4, ymmword ptr [rdx]
       cmp      dword ptr [rbp-0x64], 0
       je       G_M000_IG44
 
G_M000_IG07:                ;; offset=0x0111
       vfmadd231ps ymm0, ymm3, ymm2
       vfmadd231ps ymm1, ymm4, ymm2
 
G_M000_IG08:                ;; offset=0x011B
       add      r15, 32
       add      rdx, 32
       mov      r8d, dword ptr [rbp-0x70]
       add      r8d, 2
       lea      r8d, [r13+8*r8]
       movsxd   r8, r8d
       vbroadcastss ymm2, dword ptr [rdi+4*r8]
       vmovups  ymm3, ymmword ptr [r15]
       vmovups  ymm4, ymmword ptr [rdx]
       mov      r8d, dword ptr [rbp-0x64]
       test     r8d, r8d
       je       G_M000_IG45
 
G_M000_IG09:                ;; offset=0x014F
       vfmadd231ps ymm0, ymm3, ymm2
       vfmadd231ps ymm1, ymm4, ymm2
 
G_M000_IG10:                ;; offset=0x0159
       add      r15, 32
       add      rdx, 32
       mov      r13d, dword ptr [rbp-0x6C]
       inc      r13d
       cmp      r13d, 3
       jge      G_M000_IG46
 
G_M000_IG11:                ;; offset=0x0172
       mov      r8d, r13d
       mov      r13d, dword ptr [rbp-0x7C]
 
G_M000_IG12:                ;; offset=0x0179
       mov      dword ptr [rbp-0x6C], r8d
       add      r13d, r8d
       imul     r13d, ebx
       add      r13d, dword ptr [rbp-0x9C]
       mov      dword ptr [rbp-0x70], r13d
       mov      r13d, dword ptr [rbp-0x68]
       sar      r13d, 31
       and      r13d, 7
       add      r13d, dword ptr [rbp-0x68]
       and      r13d, -8
       mov      dword ptr [rbp-0xA4], r13d
       mov      r13d, dword ptr [rbp-0x68]
       sub      r13d, dword ptr [rbp-0xA4]
       mov      r8d, dword ptr [rbp-0x70]
       lea      r8d, [r13+8*r8]
       movsxd   r8, r8d
       vbroadcastss ymm2, dword ptr [rdi+4*r8]
       vmovups  ymm3, ymmword ptr [r15]
       vmovups  ymm4, ymmword ptr [rdx]
       cmp      dword ptr [rbp-0x64], 0
       jne      G_M000_IG05
 
G_M000_IG13:                ;; offset=0x01DA
       vmulps   ymm3, ymm2, ymm3
       vaddps   ymm0, ymm3, ymm0
       vmulps   ymm2, ymm2, ymm4
       vaddps   ymm1, ymm2, ymm1
       jmp      G_M000_IG06
 
G_M000_IG14:                ;; offset=0x01EF
       mov      r10d, dword ptr [rbp-0x58]
       inc      r10d
       mov      r8d, dword ptr [rbp+0x20]
       cmp      r10d, r8d
       mov      dword ptr [rbp-0x58], r10d
       jge      SHORT G_M000_IG18
 
G_M000_IG15:                ;; offset=0x0203
       mov      dword ptr [rbp+0x20], r8d
 
G_M000_IG16:                ;; offset=0x0207
       xor      r13d, r13d
       mov      r10d, eax
       imul     r10d, dword ptr [rbp-0x58]
       mov      dword ptr [rbp-0x94], r10d
       jmp      G_M000_IG34
       align    [0 bytes for IG39]
 
G_M000_IG17:                ;; offset=0x021E
       mov      r8d, dword ptr [rbp+0x20]
 
G_M000_IG18:                ;; offset=0x0222
       mov      r13d, dword ptr [rbp-0x54]
       add      r13d, 16
       mov      r10d, dword ptr [rbp-0x40]
       cmp      r13d, r10d
       mov      dword ptr [rbp-0x40], r10d
       mov      dword ptr [rbp+0x20], r8d
       jge      G_M000_IG03
 
G_M000_IG19:                ;; offset=0x023F
       xor      r10d, r10d
       mov      dword ptr [rbp-0x58], r10d
       mov      dword ptr [rbp-0x54], r13d
       mov      r10d, r13d
       imul     r10d, ecx
       lea      r10d, [r10+8*r10]
       movsxd   r10, r10d
       mov      r13, qword ptr [rbp-0x30]
       lea      r10, [r13+4*r10]
       mov      qword ptr [rbp-0x88], r10
       mov      r13d, dword ptr [rbp-0x58]
       cmp      r13d, dword ptr [rbp+0x20]
       jl       SHORT G_M000_IG16
       jmp      SHORT G_M000_IG17
 
G_M000_IG20:                ;; offset=0x0274
       mov      ecx, dword ptr [rbp-0x3C]
 
G_M000_IG21:                ;; offset=0x0277
       mov      r13d, dword ptr [rbp-0x54]
       mov      edx, r13d
       sar      edx, 31
       and      edx, 7
       mov      dword ptr [rbp-0x54], r13d
       add      edx, r13d
       sar      edx, 3
       mov      r15d, dword ptr [rbp-0x4C]
       mov      r8d, edx
       imul     r8d, r15d
       mov      r9d, dword ptr [rbp-0xA0]
       add      r8d, r9d
       mov      r10d, dword ptr [rbp-0x5C]
       add      r8d, r10d
       shl      r8d, 3
       mov      dword ptr [rbp-0x74], r8d
       movsxd   r13, r8d
       mov      r15, qword ptr [rbp-0x38]
       vmovups  ymmword ptr [r15+4*r13], ymm0
       mov      r13d, dword ptr [rbp-0x54]
       add      r13d, 8
       mov      r15d, dword ptr [rbp-0x40]
       cmp      r13d, r15d
       jge      SHORT G_M000_IG23
 
G_M000_IG22:                ;; offset=0x02D0
       lea      r15d, [rdx+0x01]
       imul     r15d, dword ptr [rbp-0x4C]
       add      r15d, r9d
       add      r15d, r10d
       shl      r15d, 3
       movsxd   r15, r15d
       mov      r8, qword ptr [rbp-0x38]
       vmovups  ymmword ptr [r8+4*r15], ymm1
       mov      qword ptr [rbp-0x38], r8
 
G_M000_IG23:                ;; offset=0x02F4
       mov      r8d, dword ptr [rbp-0x74]
       lea      r15d, [r8+0x08]
       movsxd   r15, r15d
       mov      r8, qword ptr [rbp-0x38]
       vmovups  ymmword ptr [r8+4*r15], ymm2
       mov      r15d, dword ptr [rbp-0x40]
       cmp      r13d, r15d
       jge      SHORT G_M000_IG25
 
G_M000_IG24:                ;; offset=0x0312
       lea      r15d, [rdx+0x01]
       imul     r15d, dword ptr [rbp-0x4C]
       add      r15d, r9d
       add      r15d, r10d
       lea      r15d, [8*r15+0x08]
       movsxd   r15, r15d
       vmovups  ymmword ptr [r8+4*r15], ymm3
 
G_M000_IG25:                ;; offset=0x0332
       mov      r15d, dword ptr [rbp-0x74]
       add      r15d, 16
       movsxd   r15, r15d
       vmovups  ymmword ptr [r8+4*r15], ymm4
       mov      r15d, dword ptr [rbp-0x40]
       cmp      r13d, r15d
       jge      SHORT G_M000_IG27
 
G_M000_IG26:                ;; offset=0x034C
       lea      r15d, [rdx+0x01]
       imul     r15d, dword ptr [rbp-0x4C]
       add      r15d, r9d
       add      r15d, r10d
       lea      r15d, [8*r15+0x10]
       movsxd   r15, r15d
       vmovups  ymmword ptr [r8+4*r15], ymm5
 
G_M000_IG27:                ;; offset=0x036C
       mov      r15d, dword ptr [rbp-0x74]
       add      r15d, 24
       movsxd   r15, r15d
       vmovups  ymmword ptr [r8+4*r15], ymm6
       mov      r15d, dword ptr [rbp-0x40]
       cmp      r13d, r15d
       jge      SHORT G_M000_IG29
 
G_M000_IG28:                ;; offset=0x0386
       lea      r15d, [rdx+0x01]
       imul     r15d, dword ptr [rbp-0x4C]
       add      r15d, r9d
       add      r15d, r10d
       lea      r15d, [8*r15+0x18]
       movsxd   r15, r15d
       vmovups  ymmword ptr [r8+4*r15], ymm7
 
G_M000_IG29:                ;; offset=0x03A6
       mov      r15d, dword ptr [rbp-0x74]
       add      r15d, 32
       movsxd   r15, r15d
       vmovups  ymmword ptr [r8+4*r15], ymm8
       mov      r15d, dword ptr [rbp-0x40]
       cmp      r13d, r15d
       jge      SHORT G_M000_IG31
 
G_M000_IG30:                ;; offset=0x03C0
       lea      r15d, [rdx+0x01]
       imul     r15d, dword ptr [rbp-0x4C]
       add      r15d, r9d
       add      r15d, r10d
       lea      r15d, [8*r15+0x20]
       movsxd   r15, r15d
       vmovups  ymmword ptr [r8+4*r15], ymm9
 
G_M000_IG31:                ;; offset=0x03E0
       mov      r15d, dword ptr [rbp-0x74]
       add      r15d, 40
       movsxd   r15, r15d
       vmovups  ymmword ptr [r8+4*r15], ymm10
       mov      r15d, dword ptr [rbp-0x40]
       cmp      r13d, r15d
       jge      SHORT G_M000_IG33
 
G_M000_IG32:                ;; offset=0x03FA
       inc      edx
       imul     edx, dword ptr [rbp-0x4C]
       add      r9d, edx
       add      r9d, r10d
       lea      edx, [8*r9+0x28]
       movsxd   rdx, edx
       mov      qword ptr [rbp-0x38], r8
       vmovups  ymmword ptr [r8+4*rdx], ymm11
       mov      r8, qword ptr [rbp-0x38]
 
G_M000_IG33:                ;; offset=0x041F
       add      r10d, 6
       mov      qword ptr [rbp-0x38], r8
       mov      dword ptr [rbp-0x40], r15d
       mov      r13d, r10d
 
G_M000_IG34:                ;; offset=0x042E
       lea      r8d, [r13+0x06]
       cmp      r8d, dword ptr [rbp+0x28]
       jg       G_M000_IG43
 
G_M000_IG35:                ;; offset=0x043C
       mov      r8d, dword ptr [rbp-0x58]
       imul     r8d, dword ptr [rbp+0x28]
       mov      dword ptr [rbp-0xA0], r8d
       lea      r15d, [r8+r13+0x06]
       cmp      r15d, dword ptr [rbp-0x50]
       jg       G_M000_IG43
 
G_M000_IG36:                ;; offset=0x045B
       vxorps   ymm0, ymm0, ymm0
       vxorps   ymm1, ymm1, ymm1
       vxorps   ymm2, ymm2, ymm2
       vxorps   ymm3, ymm3, ymm3
       vxorps   ymm4, ymm4, ymm4
       vxorps   ymm5, ymm5, ymm5
       vxorps   ymm6, ymm6, ymm6
       vxorps   ymm7, ymm7, ymm7
       vxorps   ymm8, ymm8, ymm8
       vxorps   ymm9, ymm9, ymm9
       vxorps   ymm10, ymm10, ymm10
       vxorps   ymm11, ymm11, ymm11
       mov      r15, qword ptr [rbp-0x88]
       mov      r8, qword ptr [rbp-0x90]
       add      r8, r15
       xor      edx, edx
       mov      dword ptr [rbp-0x5C], r13d
       imul     r13d, eax
       mov      dword ptr [rbp-0x98], r13d
       mov      dword ptr [rbp-0x3C], ecx
       cmp      edx, ecx
       jge      G_M000_IG20
 
G_M000_IG37:                ;; offset=0x04BC
       xor      r9d, r9d
       mov      r10d, edx
       sar      r10d, 31
       and      r10d, 7
       add      r10d, edx
       sar      r10d, 3
       imul     r10d, dword ptr [rbp-0x44]
       add      r10d, dword ptr [rbp-0x94]
       mov      dword ptr [rbp-0x78], r10d
       mov      ecx, edx
       sar      ecx, 31
       and      ecx, 7
       add      ecx, edx
       and      ecx, -8
       mov      dword ptr [rbp-0x60], edx
       mov      r13d, edx
       sub      r13d, ecx
       movsxd   rcx, r13d
       shl      rcx, 2
 
G_M000_IG38:                ;; offset=0x04FE
       lea      r13d, [r10+r9]
       imul     r13d, ebx
       add      r13d, dword ptr [rbp-0x98]
       shl      r13d, 3
       mov      ebx, 3
 
G_M000_IG39:                ;; offset=0x0516
       vmovups  ymm12, ymmword ptr [r15]
       vmovups  ymm13, ymmword ptr [r8]
       movsxd   r10, r13d
       shl      r10, 2
       add      r10, rdi
       add      r10, rcx
       vbroadcastss ymm14, dword ptr [r10]
       vfmadd231ps ymm0, ymm12, ymm14
       vfmadd231ps ymm1, ymm13, ymm14
       lea      edx, [8*rax]
       movsxd   rdx, edx
       vbroadcastss ymm14, dword ptr [r10+4*rdx]
       vfmadd231ps ymm2, ymm12, ymm14
       vfmadd231ps ymm3, ymm13, ymm14
       vbroadcastss ymm14, dword ptr [r10+4*rsi]
       vfmadd231ps ymm4, ymm12, ymm14
       vfmadd231ps ymm5, ymm13, ymm14
       vbroadcastss ymm14, dword ptr [r10+4*r11]
       vfmadd231ps ymm6, ymm12, ymm14
       vfmadd231ps ymm7, ymm13, ymm14
       vbroadcastss ymm14, dword ptr [r10+4*r12]
       vfmadd231ps ymm8, ymm12, ymm14
       vfmadd231ps ymm9, ymm13, ymm14
       vbroadcastss ymm14, dword ptr [r10+4*r14]
       vfmadd231ps ymm10, ymm12, ymm14
       vfmadd231ps ymm11, ymm13, ymm14
       add      r15, 32
       add      r8, 32
       add      r13d, 8
       dec      ebx
       jne      G_M000_IG39
 
G_M000_IG40:                ;; offset=0x05AA
       inc      r9d
       cmp      r9d, 3
       mov      ebx, dword ptr [rbp-0x48]
       mov      r10d, dword ptr [rbp-0x78]
       jl       G_M000_IG38
 
G_M000_IG41:                ;; offset=0x05BE
       mov      edx, dword ptr [rbp-0x60]
       inc      edx
       mov      ecx, dword ptr [rbp-0x3C]
       cmp      edx, ecx
       jge      G_M000_IG21
 
G_M000_IG42:                ;; offset=0x05CE
       mov      dword ptr [rbp-0x3C], ecx
       jmp      G_M000_IG37
 
G_M000_IG43:                ;; offset=0x05D6
       mov      r8d, dword ptr [rbp-0x58]
       imul     r8d, dword ptr [rbp+0x28]
       jmp      G_M000_IG52
 
G_M000_IG44:                ;; offset=0x05E4
       vmulps   ymm3, ymm2, ymm3
       vaddps   ymm0, ymm3, ymm0
       vmulps   ymm2, ymm2, ymm4
       vaddps   ymm1, ymm2, ymm1
       jmp      G_M000_IG08
 
G_M000_IG45:                ;; offset=0x05F9
       vmulps   ymm3, ymm2, ymm3
       vaddps   ymm0, ymm3, ymm0
       vmulps   ymm2, ymm2, ymm4
       vaddps   ymm1, ymm2, ymm1
       jmp      G_M000_IG10
 
G_M000_IG46:                ;; offset=0x060E
       mov      r13d, dword ptr [rbp-0x68]
       inc      r13d
       cmp      r13d, ecx
       jge      SHORT G_M000_IG49
 
G_M000_IG47:                ;; offset=0x061A
       xor      r8d, r8d
       mov      dword ptr [rbp-0x68], r13d
       sar      r13d, 31
       and      r13d, 7
       add      r13d, dword ptr [rbp-0x68]
       sar      r13d, 3
       mov      r9d, dword ptr [rbp-0x44]
       imul     r13d, r9d
       mov      r10d, dword ptr [rbp-0x94]
       add      r13d, r10d
       mov      dword ptr [rbp-0x7C], r13d
       jmp      G_M000_IG12
 
G_M000_IG48:                ;; offset=0x064C
       mov      r9d, dword ptr [rbp-0x44]
       mov      r10d, dword ptr [rbp-0x94]
 
G_M000_IG49:                ;; offset=0x0657
       mov      r13d, dword ptr [rbp-0x54]
       mov      edx, r13d
       sar      edx, 31
       and      edx, 7
       mov      dword ptr [rbp-0x54], r13d
       add      edx, r13d
       sar      edx, 3
       mov      dword ptr [rbp-0x80], edx
       mov      r15d, dword ptr [rbp-0x4C]
       mov      r8d, edx
       imul     r8d, r15d
       mov      r13d, dword ptr [rbp-0xA0]
       add      r8d, r13d
       mov      r15d, dword ptr [rbp-0x5C]
       add      r8d, r15d
       shl      r8d, 3
       movsxd   r8, r8d
       mov      rdx, qword ptr [rbp-0x38]
       vmovups  ymmword ptr [rdx+4*r8], ymm0
       mov      r8d, dword ptr [rbp-0x54]
       add      r8d, 8
       mov      edx, dword ptr [rbp-0x40]
       cmp      r8d, edx
       jge      SHORT G_M000_IG51
 
G_M000_IG50:                ;; offset=0x06AE
       mov      r8d, dword ptr [rbp-0x80]
       inc      r8d
       imul     r8d, dword ptr [rbp-0x4C]
       add      r8d, r13d
       add      r8d, r15d
       shl      r8d, 3
       movsxd   r8, r8d
       mov      rdx, qword ptr [rbp-0x38]
       vmovups  ymmword ptr [rdx+4*r8], ymm1
       mov      qword ptr [rbp-0x38], rdx
 
G_M000_IG51:                ;; offset=0x06D5
       inc      r15d
       mov      r8d, r13d
       mov      r13d, r15d
 
G_M000_IG52:                ;; offset=0x06DE
       cmp      r13d, dword ptr [rbp+0x28]
       jge      G_M000_IG14
 
G_M000_IG53:                ;; offset=0x06E8
       vxorps   ymm0, ymm0, ymm0
       vxorps   ymm1, ymm1, ymm1
       mov      r15, qword ptr [rbp-0x88]
       mov      rdx, qword ptr [rbp-0x90]
       add      rdx, r15
       mov      dword ptr [rbp-0xA0], r8d
       mov      dword ptr [rbp-0x5C], r13d
       add      r8d, r13d
       cmp      r8d, dword ptr [rbp-0x50]
       setl     r8b
       movzx    r8, r8b
       mov      dword ptr [rbp-0x64], r8d
       xor      r13d, r13d
       mov      r8d, eax
       imul     r8d, dword ptr [rbp-0x5C]
       mov      dword ptr [rbp-0x9C], r8d
       cmp      r13d, ecx
       jl       G_M000_IG47
       jmp      G_M000_IG48
 
; Total bytes of code 1855


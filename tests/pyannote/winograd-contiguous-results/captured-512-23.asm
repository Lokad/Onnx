; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:Kernel512(ptr,ptr,ptr,int,int,int,int,int,int,int) (Tier1)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
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
       mov      r12d, ecx
       shl      r12d, 4
       lea      r12d, [r12+8*r12]
       movsxd   r12, r12d
       shl      r12, 2
       mov      qword ptr [rbp-0x90], r12
       lea      esi, [rax+rax]
       shl      esi, 4
       movsxd   rsi, esi
       imul     r11d, eax, 48
       movsxd   r11, r11d
       lea      r12d, [4*rax]
       shl      r12d, 4
       movsxd   r12, r12d
       imul     r14d, eax, 80
       movsxd   r14, r14d
       mov      dword ptr [rbp-0x40], r8d
       cmp      r13d, r8d
       jl       SHORT G_M000_IG07
 
G_M000_IG03:                ;; offset=0x00B5
       vzeroupper 
       add      rsp, 136
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
G_M000_IG04:                ;; offset=0x00CA
       mov      r8d, dword ptr [rbp-0x58]
       imul     r8d, dword ptr [rbp+0x28]
       jmp      G_M000_IG16
       align    [0 bytes for IG08]
 
G_M000_IG05:                ;; offset=0x00D8
       mov      r8d, dword ptr [rbp+0x20]
 
G_M000_IG06:                ;; offset=0x00DC
       mov      r13d, dword ptr [rbp-0x54]
       add      r13d, 32
       mov      r10d, dword ptr [rbp-0x40]
       cmp      r13d, r10d
       mov      dword ptr [rbp-0x40], r10d
       mov      dword ptr [rbp+0x20], r8d
       jge      SHORT G_M000_IG03
 
G_M000_IG07:                ;; offset=0x00F5
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
       jl       G_M000_IG29
       jmp      SHORT G_M000_IG05
 
G_M000_IG08:                ;; offset=0x012E
       vmulps   zmm3, zmm2, zmm3
       vaddps   zmm0, zmm3, zmm0
       vmulps   zmm2, zmm2, zmm4
       vaddps   zmm1, zmm2, zmm1
       jmp      G_M000_IG21
 
G_M000_IG09:                ;; offset=0x014B
       vmulps   zmm3, zmm2, zmm3
       vaddps   zmm0, zmm3, zmm0
       vmulps   zmm2, zmm2, zmm4
       vaddps   zmm1, zmm2, zmm1
       jmp      G_M000_IG23
 
G_M000_IG10:                ;; offset=0x0168
       mov      r8d, dword ptr [rbp-0x68]
       inc      r8d
       cmp      r8d, ecx
       mov      r13d, r8d
       jge      SHORT G_M000_IG13
 
G_M000_IG11:                ;; offset=0x0177
       xor      r8d, r8d
       mov      dword ptr [rbp-0x68], r13d
       sar      r13d, 31
       and      r13d, 15
       add      r13d, dword ptr [rbp-0x68]
       sar      r13d, 4
       mov      r9d, dword ptr [rbp-0x44]
       imul     r13d, r9d
       mov      r10d, dword ptr [rbp-0x94]
       add      r13d, r10d
       mov      dword ptr [rbp-0x78], r13d
       jmp      G_M000_IG25
 
G_M000_IG12:                ;; offset=0x01A9
       mov      r9d, dword ptr [rbp-0x44]
       mov      r10d, dword ptr [rbp-0x94]
 
G_M000_IG13:                ;; offset=0x01B4
       mov      r13d, dword ptr [rbp-0x54]
       mov      edx, r13d
       sar      edx, 31
       and      edx, 15
       mov      dword ptr [rbp-0x54], r13d
       add      edx, r13d
       sar      edx, 4
       mov      dword ptr [rbp-0x7C], edx
       mov      r15d, dword ptr [rbp-0x4C]
       mov      r8d, edx
       imul     r8d, r15d
       mov      r13d, dword ptr [rbp-0xA0]
       add      r8d, r13d
       mov      r15d, dword ptr [rbp-0x5C]
       add      r8d, r15d
       shl      r8d, 4
       movsxd   r8, r8d
       mov      rdx, qword ptr [rbp-0x38]
       vmovups  zmmword ptr [rdx+4*r8], zmm0
       mov      r8d, dword ptr [rbp-0x54]
       add      r8d, 16
       mov      edx, dword ptr [rbp-0x40]
       cmp      r8d, edx
       jge      SHORT G_M000_IG15
 
G_M000_IG14:                ;; offset=0x020C
       mov      r8d, dword ptr [rbp-0x7C]
       inc      r8d
       imul     r8d, dword ptr [rbp-0x4C]
       add      r8d, r13d
       add      r8d, r15d
       shl      r8d, 4
       movsxd   r8, r8d
       mov      rdx, qword ptr [rbp-0x38]
       vmovups  zmmword ptr [rdx+4*r8], zmm1
       mov      qword ptr [rbp-0x38], rdx
 
G_M000_IG15:                ;; offset=0x0234
       inc      r15d
       mov      r8d, r13d
       mov      r13d, r15d
 
G_M000_IG16:                ;; offset=0x023D
       cmp      r13d, dword ptr [rbp+0x28]
       jge      G_M000_IG27
 
G_M000_IG17:                ;; offset=0x0247
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
       jl       G_M000_IG11
       jmp      G_M000_IG12
 
G_M000_IG18:                ;; offset=0x029E
       vfmadd231ps zmm0, zmm3, zmm2
       vfmadd231ps zmm1, zmm4, zmm2
 
G_M000_IG19:                ;; offset=0x02AA
       add      r15, 64
       add      rdx, 64
       mov      r13d, dword ptr [rbp-0x6C]
       inc      r13d
       shl      r13d, 4
       add      r13d, dword ptr [rbp-0xA4]
       movsxd   r13, r13d
       vbroadcastss zmm2, dword ptr [rdi+4*r13]
       vmovups  zmm3, zmmword ptr [r15]
       vmovups  zmm4, zmmword ptr [rdx]
       cmp      dword ptr [rbp-0x64], 0
       je       G_M000_IG08
 
G_M000_IG20:                ;; offset=0x02E4
       vfmadd231ps zmm0, zmm3, zmm2
       vfmadd231ps zmm1, zmm4, zmm2
 
G_M000_IG21:                ;; offset=0x02F0
       add      r15, 64
       add      rdx, 64
       mov      r13d, dword ptr [rbp-0x6C]
       add      r13d, 2
       shl      r13d, 4
       add      r13d, dword ptr [rbp-0xA4]
       movsxd   r13, r13d
       vbroadcastss zmm2, dword ptr [rdi+4*r13]
       vmovups  zmm3, zmmword ptr [r15]
       vmovups  zmm4, zmmword ptr [rdx]
       mov      r13d, dword ptr [rbp-0x64]
       test     r13d, r13d
       je       G_M000_IG09
 
G_M000_IG22:                ;; offset=0x032E
       vfmadd231ps zmm0, zmm3, zmm2
       vfmadd231ps zmm1, zmm4, zmm2
 
G_M000_IG23:                ;; offset=0x033A
       add      r15, 64
       add      rdx, 64
       inc      r8d
       cmp      r8d, 3
       jge      G_M000_IG10
 
G_M000_IG24:                ;; offset=0x034F
       mov      r13d, dword ptr [rbp-0x78]
 
G_M000_IG25:                ;; offset=0x0353
       add      r13d, r8d
       imul     r13d, ebx
       add      r13d, dword ptr [rbp-0x9C]
       mov      dword ptr [rbp-0x6C], r13d
       shl      r13d, 4
       mov      dword ptr [rbp-0xA8], r13d
       mov      r13d, dword ptr [rbp-0x68]
       sar      r13d, 31
       and      r13d, 15
       add      r13d, dword ptr [rbp-0x68]
       and      r13d, -16
       mov      dword ptr [rbp-0xAC], r13d
       mov      r13d, dword ptr [rbp-0x68]
       sub      r13d, dword ptr [rbp-0xAC]
       mov      dword ptr [rbp-0xA4], r13d
       mov      r13d, dword ptr [rbp-0xA8]
       add      r13d, dword ptr [rbp-0xA4]
       movsxd   r13, r13d
       vbroadcastss zmm2, dword ptr [rdi+4*r13]
       vmovups  zmm3, zmmword ptr [r15]
       vmovups  zmm4, zmmword ptr [rdx]
       cmp      dword ptr [rbp-0x64], 0
       jne      G_M000_IG18
 
G_M000_IG26:                ;; offset=0x03CB
       vmulps   zmm3, zmm2, zmm3
       vaddps   zmm0, zmm3, zmm0
       vmulps   zmm2, zmm2, zmm4
       vaddps   zmm1, zmm2, zmm1
       jmp      G_M000_IG19
 
G_M000_IG27:                ;; offset=0x03E8
       mov      r10d, dword ptr [rbp-0x58]
       inc      r10d
       mov      r8d, dword ptr [rbp+0x20]
       cmp      r10d, r8d
       mov      dword ptr [rbp-0x58], r10d
       jge      G_M000_IG06
 
G_M000_IG28:                ;; offset=0x0400
       mov      dword ptr [rbp+0x20], r8d
 
G_M000_IG29:                ;; offset=0x0404
       xor      r13d, r13d
       mov      r10d, eax
       imul     r10d, dword ptr [rbp-0x58]
       mov      dword ptr [rbp-0x94], r10d
       jmp      G_M000_IG44
       align    [0 bytes for IG49]
 
G_M000_IG30:                ;; offset=0x041B
       mov      ecx, dword ptr [rbp-0x3C]
 
G_M000_IG31:                ;; offset=0x041E
       mov      r13d, dword ptr [rbp-0x54]
       mov      edx, r13d
       sar      edx, 31
       and      edx, 15
       mov      dword ptr [rbp-0x54], r13d
       add      edx, r13d
       sar      edx, 4
       mov      r15d, dword ptr [rbp-0x4C]
       mov      r8d, edx
       imul     r8d, r15d
       mov      r9d, dword ptr [rbp-0xA0]
       add      r8d, r9d
       mov      r10d, dword ptr [rbp-0x5C]
       add      r8d, r10d
       mov      dword ptr [rbp-0x70], r8d
       mov      r13d, r8d
       shl      r13d, 4
       movsxd   r13, r13d
       mov      r15, qword ptr [rbp-0x38]
       vmovups  zmmword ptr [r15+4*r13], zmm0
       mov      r13d, dword ptr [rbp-0x54]
       add      r13d, 16
       mov      r15d, dword ptr [rbp-0x40]
       cmp      r13d, r15d
       jge      SHORT G_M000_IG33
 
G_M000_IG32:                ;; offset=0x047B
       lea      r15d, [rdx+0x01]
       imul     r15d, dword ptr [rbp-0x4C]
       add      r15d, r9d
       add      r15d, r10d
       shl      r15d, 4
       movsxd   r15, r15d
       mov      r8, qword ptr [rbp-0x38]
       vmovups  zmmword ptr [r8+4*r15], zmm1
       mov      qword ptr [rbp-0x38], r8
 
G_M000_IG33:                ;; offset=0x04A0
       mov      r8d, dword ptr [rbp-0x70]
       lea      r15d, [r8+0x01]
       shl      r15d, 4
       movsxd   r15, r15d
       mov      r8, qword ptr [rbp-0x38]
       vmovups  zmmword ptr [r8+4*r15], zmm2
       mov      r15d, dword ptr [rbp-0x40]
       cmp      r13d, r15d
       jge      SHORT G_M000_IG35
 
G_M000_IG34:                ;; offset=0x04C3
       lea      r15d, [rdx+0x01]
       imul     r15d, dword ptr [rbp-0x4C]
       add      r15d, r9d
       lea      r15d, [r15+r10+0x01]
       shl      r15d, 4
       movsxd   r15, r15d
       vmovups  zmmword ptr [r8+4*r15], zmm3
 
G_M000_IG35:                ;; offset=0x04E2
       mov      r15d, dword ptr [rbp-0x70]
       add      r15d, 2
       shl      r15d, 4
       movsxd   r15, r15d
       vmovups  zmmword ptr [r8+4*r15], zmm4
       mov      r15d, dword ptr [rbp-0x40]
       cmp      r13d, r15d
       jge      SHORT G_M000_IG37
 
G_M000_IG36:                ;; offset=0x0501
       lea      r15d, [rdx+0x01]
       imul     r15d, dword ptr [rbp-0x4C]
       add      r15d, r9d
       lea      r15d, [r15+r10+0x02]
       shl      r15d, 4
       movsxd   r15, r15d
       vmovups  zmmword ptr [r8+4*r15], zmm5
 
G_M000_IG37:                ;; offset=0x0520
       mov      r15d, dword ptr [rbp-0x70]
       add      r15d, 3
       shl      r15d, 4
       movsxd   r15, r15d
       vmovups  zmmword ptr [r8+4*r15], zmm6
       mov      r15d, dword ptr [rbp-0x40]
       cmp      r13d, r15d
       jge      SHORT G_M000_IG39
 
G_M000_IG38:                ;; offset=0x053F
       lea      r15d, [rdx+0x01]
       imul     r15d, dword ptr [rbp-0x4C]
       add      r15d, r9d
       lea      r15d, [r15+r10+0x03]
       shl      r15d, 4
       movsxd   r15, r15d
       vmovups  zmmword ptr [r8+4*r15], zmm7
 
G_M000_IG39:                ;; offset=0x055E
       mov      r15d, dword ptr [rbp-0x70]
       add      r15d, 4
       shl      r15d, 4
       movsxd   r15, r15d
       vmovups  zmmword ptr [r8+4*r15], zmm8
       mov      r15d, dword ptr [rbp-0x40]
       cmp      r13d, r15d
       jge      SHORT G_M000_IG41
 
G_M000_IG40:                ;; offset=0x057D
       lea      r15d, [rdx+0x01]
       imul     r15d, dword ptr [rbp-0x4C]
       add      r15d, r9d
       lea      r15d, [r15+r10+0x04]
       shl      r15d, 4
       movsxd   r15, r15d
       vmovups  zmmword ptr [r8+4*r15], zmm9
 
G_M000_IG41:                ;; offset=0x059C
       mov      r15d, dword ptr [rbp-0x70]
       add      r15d, 5
       shl      r15d, 4
       movsxd   r15, r15d
       vmovups  zmmword ptr [r8+4*r15], zmm10
       mov      r15d, dword ptr [rbp-0x40]
       cmp      r13d, r15d
       jge      SHORT G_M000_IG43
 
G_M000_IG42:                ;; offset=0x05BB
       inc      edx
       imul     edx, dword ptr [rbp-0x4C]
       add      r9d, edx
       lea      edx, [r9+r10+0x05]
       shl      edx, 4
       movsxd   rdx, edx
       mov      qword ptr [rbp-0x38], r8
       vmovups  zmmword ptr [r8+4*rdx], zmm11
       mov      r8, qword ptr [rbp-0x38]
 
G_M000_IG43:                ;; offset=0x05DE
       add      r10d, 6
       mov      qword ptr [rbp-0x38], r8
       mov      dword ptr [rbp-0x40], r15d
       mov      r13d, r10d
 
G_M000_IG44:                ;; offset=0x05ED
       lea      r8d, [r13+0x06]
       cmp      r8d, dword ptr [rbp+0x28]
       jg       G_M000_IG04
 
G_M000_IG45:                ;; offset=0x05FB
       mov      r8d, dword ptr [rbp-0x58]
       imul     r8d, dword ptr [rbp+0x28]
       mov      dword ptr [rbp-0xA0], r8d
       lea      r15d, [r8+r13+0x06]
       cmp      r15d, dword ptr [rbp-0x50]
       jg       G_M000_IG04
 
G_M000_IG46:                ;; offset=0x061A
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
       jge      G_M000_IG30
 
G_M000_IG47:                ;; offset=0x067B
       xor      r9d, r9d
       mov      r10d, edx
       sar      r10d, 31
       and      r10d, 15
       add      r10d, edx
       sar      r10d, 4
       imul     r10d, dword ptr [rbp-0x44]
       add      r10d, dword ptr [rbp-0x94]
       mov      dword ptr [rbp-0x74], r10d
       mov      ecx, edx
       sar      ecx, 31
       and      ecx, 15
       add      ecx, edx
       and      ecx, -16
       mov      dword ptr [rbp-0x60], edx
       mov      r13d, edx
       sub      r13d, ecx
       movsxd   rcx, r13d
       shl      rcx, 2
 
G_M000_IG48:                ;; offset=0x06BD
       lea      r13d, [r10+r9]
       imul     r13d, ebx
       add      r13d, dword ptr [rbp-0x98]
       shl      r13d, 4
       mov      ebx, 3
 
G_M000_IG49:                ;; offset=0x06D5
       vmovups  zmm12, zmmword ptr [r15]
       vmovups  zmm13, zmmword ptr [r8]
       movsxd   r10, r13d
       shl      r10, 2
       add      r10, rdi
       add      r10, rcx
       vbroadcastss zmm14, dword ptr [r10]
       vfmadd231ps zmm0, zmm12, zmm14
       vfmadd231ps zmm1, zmm13, zmm14
       mov      edx, eax
       shl      edx, 4
       movsxd   rdx, edx
       vbroadcastss zmm14, dword ptr [r10+4*rdx]
       vfmadd231ps zmm2, zmm12, zmm14
       vfmadd231ps zmm3, zmm13, zmm14
       vbroadcastss zmm14, dword ptr [r10+4*rsi]
       vfmadd231ps zmm4, zmm12, zmm14
       vfmadd231ps zmm5, zmm13, zmm14
       vbroadcastss zmm14, dword ptr [r10+4*r11]
       vfmadd231ps zmm6, zmm12, zmm14
       vfmadd231ps zmm7, zmm13, zmm14
       vbroadcastss zmm14, dword ptr [r10+4*r12]
       vfmadd231ps zmm8, zmm12, zmm14
       vfmadd231ps zmm9, zmm13, zmm14
       vbroadcastss zmm14, dword ptr [r10+4*r14]
       vfmadd231ps zmm10, zmm12, zmm14
       vfmadd231ps zmm11, zmm13, zmm14
       add      r15, 64
       add      r8, 64
       add      r13d, 16
       dec      ebx
       jne      G_M000_IG49
 
G_M000_IG50:                ;; offset=0x077B
       inc      r9d
       cmp      r9d, 3
       mov      ebx, dword ptr [rbp-0x48]
       mov      r10d, dword ptr [rbp-0x74]
       jl       G_M000_IG48
 
G_M000_IG51:                ;; offset=0x078F
       mov      edx, dword ptr [rbp-0x60]
       inc      edx
       mov      ecx, dword ptr [rbp-0x3C]
       cmp      edx, ecx
       jge      G_M000_IG31
 
G_M000_IG52:                ;; offset=0x079F
       mov      dword ptr [rbp-0x3C], ecx
       jmp      G_M000_IG47
 
; Total bytes of code 1959


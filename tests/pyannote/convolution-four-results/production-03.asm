; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:Kernel512(ptr,ptr,ptr,int,int,int,int,int,int,int) (Tier1-OSR)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Tier1-OSR code
; OSR variant for entry point 0x1f7
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 1

G_M000_IG01:                ;; offset=0x0000
       mov      rax, qword ptr [rbp]
       push     rax
       sub      rsp, 64
       mov      qword ptr [rsp+0x7C8], r15
       mov      qword ptr [rsp+0x7C0], r14
       mov      qword ptr [rsp+0x7B8], r13
       mov      qword ptr [rsp+0x7B0], r12
       mov      qword ptr [rsp+0x7A8], rbx
       lea      rbp, [rsp+0x40]
       mov      rcx, qword ptr [rbp+0x760]
       mov      rdi, qword ptr [rbp+0x750]
       mov      edx, dword ptr [rbp+0x74C]
       mov      esi, dword ptr [rbp+0x748]
       mov      eax, dword ptr [rbp+0x7A8]
       mov      r9d, dword ptr [rbp+0x7B8]
       mov      r12d, dword ptr [rbp+0x73C]
       mov      r13d, dword ptr [rbp+0x728]
       vmovups  zmm0, zmmword ptr [rbp+0x6E0]
       vmovups  zmm6, zmmword ptr [rbp+0x6A0]
       vmovups  zmm1, zmmword ptr [rbp+0x660]
       vmovups  zmm7, zmmword ptr [rbp+0x620]
       vmovups  zmm2, zmmword ptr [rbp+0x5E0]
       vmovups  zmm8, zmmword ptr [rbp+0x5A0]
       vmovups  zmm3, zmmword ptr [rbp+0x560]
       vmovups  zmm9, zmmword ptr [rbp+0x520]
       vmovups  zmm4, zmmword ptr [rbp+0x4E0]
       vmovups  zmm10, zmmword ptr [rbp+0x4A0]
       vmovups  zmm5, zmmword ptr [rbp+0x460]
       vmovups  zmm11, zmmword ptr [rbp+0x420]
       mov      rbx, qword ptr [rbp+0x418]
       mov      r15, qword ptr [rbp+0x410]
       mov      r10d, dword ptr [rbp+0x40C]
       mov      r14d, dword ptr [rbp+0x408]
       mov      r11d, dword ptr [rbp+0x404]
 
G_M000_IG02:                ;; offset=0x0106
       jmp      G_M000_IG11
 
G_M000_IG03:                ;; offset=0x010B
       inc      r10d
 
G_M000_IG04:                ;; offset=0x010E
       cmp      r10d, edx
       jge      G_M000_IG13
 
G_M000_IG05:                ;; offset=0x0117
       mov      dword ptr [rbp+0x7B8], r9d
       xor      r9d, r9d
       mov      r14d, r9d
       jmp      SHORT G_M000_IG08
 
G_M000_IG06:                ;; offset=0x0126
       inc      r14d
       cmp      r14d, 3
       jge      SHORT G_M000_IG03
 
G_M000_IG07:                ;; offset=0x012F
       mov      dword ptr [rbp+0x7B8], r9d
 
G_M000_IG08:                ;; offset=0x0136
       xor      r9d, r9d
       mov      dword ptr [rbp+0x74C], edx
       mov      qword ptr [rbp+0x750], rdi
       mov      dword ptr [rbp+0x748], esi
       mov      r11d, r9d
 
G_M000_IG09:                ;; offset=0x014F
       vmovups  zmm12, zmmword ptr [rbx]
       vmovups  zmm13, zmmword ptr [r15]
       mov      r9d, r10d
       sar      r9d, 31
       and      r9d, 15
       add      r9d, r10d
       sar      r9d, 4
       mov      r8d, dword ptr [rbp+0x740]
       imul     r9d, r8d
       mov      esi, dword ptr [rbp+0x72C]
       mov      edi, esi
       imul     edi, eax
       add      edi, r9d
       add      edi, r14d
       imul     edi, r12d
       mov      r9d, r13d
       imul     r9d, eax
       add      edi, r9d
       add      edi, r11d
       shl      edi, 4
       movsxd   rdi, edi
       shl      rdi, 2
       add      rdi, rcx
       mov      r9d, r10d
       sar      r9d, 31
       and      r9d, 15
       add      r9d, r10d
       and      r9d, -16
       mov      edx, r10d
       sub      edx, r9d
       movsxd   rdx, edx
       lea      rdx, [rdi+4*rdx]
       vbroadcastss zmm14, dword ptr [rdx]
       vfmadd231ps zmm0, zmm12, zmm14
       vfmadd231ps zmm6, zmm13, zmm14
       mov      edi, eax
       shl      edi, 4
       movsxd   rdi, edi
       vbroadcastss zmm14, dword ptr [rdx+4*rdi]
       vfmadd231ps zmm1, zmm12, zmm14
       vfmadd231ps zmm7, zmm13, zmm14
       lea      edi, [rax+rax]
       shl      edi, 4
       movsxd   rdi, edi
       vbroadcastss zmm14, dword ptr [rdx+4*rdi]
       vfmadd231ps zmm2, zmm12, zmm14
       vfmadd231ps zmm8, zmm13, zmm14
       imul     edi, eax, 48
       movsxd   rdi, edi
       vbroadcastss zmm14, dword ptr [rdx+4*rdi]
       vfmadd231ps zmm3, zmm12, zmm14
       vfmadd231ps zmm9, zmm13, zmm14
       lea      edi, [4*rax]
       shl      edi, 4
       movsxd   rdi, edi
       vbroadcastss zmm14, dword ptr [rdx+4*rdi]
       vfmadd231ps zmm4, zmm12, zmm14
       vfmadd231ps zmm10, zmm13, zmm14
       imul     edi, eax, 80
       movsxd   rdi, edi
       vbroadcastss zmm14, dword ptr [rdx+4*rdi]
       vfmadd231ps zmm5, zmm12, zmm14
       vfmadd231ps zmm11, zmm13, zmm14
       add      rbx, 64
       add      r15, 64
       inc      r11d
       mov      dword ptr [rbp+0x72C], esi
       mov      dword ptr [rbp+0x740], r8d
 
G_M000_IG10:                ;; offset=0x0279
       mov      edx, dword ptr [rbp+0x74C]
       mov      esi, dword ptr [rbp+0x748]
       mov      rdi, qword ptr [rbp+0x750]
       mov      r9d, dword ptr [rbp+0x7B8]
 
G_M000_IG11:                ;; offset=0x0293
       cmp      r11d, 3
       jge      G_M000_IG06
 
G_M000_IG12:                ;; offset=0x029D
       mov      dword ptr [rbp+0x74C], edx
       mov      qword ptr [rbp+0x750], rdi
       mov      dword ptr [rbp+0x748], esi
       mov      dword ptr [rbp+0x7B8], r9d
       jmp      G_M000_IG09
 
G_M000_IG13:                ;; offset=0x02BC
       mov      r10d, dword ptr [rbp+0x730]
       mov      r11d, r10d
       sar      r11d, 31
       and      r11d, 15
       mov      dword ptr [rbp+0x730], r10d
       add      r11d, r10d
       sar      r11d, 4
       mov      r14d, dword ptr [rbp+0x738]
       mov      ebx, r11d
       imul     ebx, r14d
       mov      r15d, dword ptr [rbp+0x72C]
       mov      dword ptr [rbp+0x7B8], r9d
       mov      r10d, r15d
       imul     r10d, r9d
       add      ebx, r10d
       add      ebx, r13d
       mov      r9d, ebx
       shl      r9d, 4
       movsxd   r9, r9d
       vmovups  zmmword ptr [rdi+4*r9], zmm0
       mov      r9d, dword ptr [rbp+0x730]
       add      r9d, 16
       cmp      r9d, esi
       jge      SHORT G_M000_IG15
 
G_M000_IG14:                ;; offset=0x0326
       lea      r8d, [r11+0x01]
       imul     r8d, r14d
       add      r8d, r10d
       add      r8d, r13d
       shl      r8d, 4
       movsxd   r8, r8d
       vmovups  zmmword ptr [rdi+4*r8], zmm6
 
G_M000_IG15:                ;; offset=0x0342
       lea      r8d, [rbx+0x01]
       shl      r8d, 4
       movsxd   r8, r8d
       vmovups  zmmword ptr [rdi+4*r8], zmm1
       cmp      r9d, esi
       jge      SHORT G_M000_IG17
 
G_M000_IG16:                ;; offset=0x0359
       lea      r8d, [r11+0x01]
       imul     r8d, r14d
       add      r8d, r10d
       lea      r8d, [r8+r13+0x01]
       shl      r8d, 4
       movsxd   r8, r8d
       vmovups  zmmword ptr [rdi+4*r8], zmm7
 
G_M000_IG17:                ;; offset=0x0377
       lea      r8d, [rbx+0x02]
       shl      r8d, 4
       movsxd   r8, r8d
       vmovups  zmmword ptr [rdi+4*r8], zmm2
       cmp      r9d, esi
       jge      SHORT G_M000_IG19
 
G_M000_IG18:                ;; offset=0x038E
       lea      r8d, [r11+0x01]
       imul     r8d, r14d
       add      r8d, r10d
       lea      r8d, [r8+r13+0x02]
       shl      r8d, 4
       movsxd   r8, r8d
       vmovups  zmmword ptr [rdi+4*r8], zmm8
 
G_M000_IG19:                ;; offset=0x03AC
       lea      r8d, [rbx+0x03]
       shl      r8d, 4
       movsxd   r8, r8d
       vmovups  zmmword ptr [rdi+4*r8], zmm3
       cmp      r9d, esi
       jge      SHORT G_M000_IG21
 
G_M000_IG20:                ;; offset=0x03C3
       lea      r8d, [r11+0x01]
       imul     r8d, r14d
       add      r8d, r10d
       lea      r8d, [r8+r13+0x03]
       shl      r8d, 4
       movsxd   r8, r8d
       vmovups  zmmword ptr [rdi+4*r8], zmm9
 
G_M000_IG21:                ;; offset=0x03E1
       lea      r8d, [rbx+0x04]
       shl      r8d, 4
       movsxd   r8, r8d
       vmovups  zmmword ptr [rdi+4*r8], zmm4
       cmp      r9d, esi
       jge      SHORT G_M000_IG23
 
G_M000_IG22:                ;; offset=0x03F8
       lea      r8d, [r11+0x01]
       imul     r8d, r14d
       add      r8d, r10d
       lea      r8d, [r8+r13+0x04]
       shl      r8d, 4
       movsxd   r8, r8d
       vmovups  zmmword ptr [rdi+4*r8], zmm10
 
G_M000_IG23:                ;; offset=0x0416
       add      ebx, 5
       shl      ebx, 4
       movsxd   r8, ebx
       vmovups  zmmword ptr [rdi+4*r8], zmm5
       mov      dword ptr [rbp+0x748], esi
       cmp      r9d, esi
       jge      SHORT G_M000_IG25
 
G_M000_IG24:                ;; offset=0x0431
       inc      r11d
       mov      dword ptr [rbp+0x738], r14d
       imul     r11d, r14d
       add      r10d, r11d
       lea      r8d, [r10+r13+0x05]
       shl      r8d, 4
       movsxd   r8, r8d
       mov      qword ptr [rbp+0x750], rdi
       vmovups  zmmword ptr [rdi+4*r8], zmm11
       mov      rdi, qword ptr [rbp+0x750]
       mov      r14d, dword ptr [rbp+0x738]
 
G_M000_IG25:                ;; offset=0x046A
       add      r13d, 6
 
G_M000_IG26:                ;; offset=0x046E
       lea      r8d, [r13+0x06]
       mov      r9d, dword ptr [rbp+0x7B8]
       cmp      r8d, r9d
       jg       G_M000_IG42
 
G_M000_IG27:                ;; offset=0x0482
       mov      r8d, r15d
       imul     r8d, r9d
       lea      r8d, [r8+r13+0x06]
       cmp      r8d, dword ptr [rbp+0x734]
       jg       G_M000_IG42
 
G_M000_IG28:                ;; offset=0x049B
       mov      dword ptr [rbp+0x7B8], r9d
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
       mov      r11d, dword ptr [rbp+0x730]
       mov      ebx, r11d
       imul     ebx, edx
       lea      ebx, [rbx+8*rbx]
       movsxd   rbx, ebx
       mov      r8, qword ptr [rbp+0x758]
       lea      rbx, [r8+4*rbx]
       mov      dword ptr [rbp+0x74C], edx
       mov      r10d, edx
       shl      r10d, 4
       lea      r10d, [r10+8*r10]
       movsxd   r10, r10d
       lea      r10, [rbx+4*r10]
       xor      r9d, r9d
       mov      dword ptr [rbp+0x72C], r15d
       mov      dword ptr [rbp+0x738], r14d
       mov      dword ptr [rbp+0x730], r11d
       mov      r15, r10
       mov      r10d, r9d
       mov      edx, dword ptr [rbp+0x74C]
       mov      esi, dword ptr [rbp+0x748]
       mov      r9d, dword ptr [rbp+0x7B8]
       jmp      G_M000_IG04
 
G_M000_IG29:                ;; offset=0x0542
       vfmadd231ps zmm0, zmm7, zmm1
       vfmadd231ps zmm6, zmm2, zmm1
 
G_M000_IG30:                ;; offset=0x054E
       add      r11, 64
       add      r10, 64
       lea      edi, [r8+0x01]
       shl      edi, 4
       add      edi, ebx
       movsxd   rdi, edi
       vbroadcastss zmm1, dword ptr [rcx+4*rdi]
       vmovups  zmm7, zmmword ptr [r11]
       vmovups  zmm2, zmmword ptr [r10]
       test     r9d, r9d
       je       G_M000_IG37
 
G_M000_IG31:                ;; offset=0x057E
       vfmadd231ps zmm0, zmm7, zmm1
       vfmadd231ps zmm6, zmm2, zmm1
 
G_M000_IG32:                ;; offset=0x058A
       add      r11, 64
       add      r10, 64
       add      r8d, 2
       shl      r8d, 4
       add      r8d, ebx
       movsxd   rdi, r8d
       vbroadcastss zmm1, dword ptr [rcx+4*rdi]
       vmovups  zmm7, zmmword ptr [r11]
       vmovups  zmm2, zmmword ptr [r10]
       test     r9d, r9d
       je       G_M000_IG38
 
G_M000_IG33:                ;; offset=0x05BC
       vfmadd231ps zmm0, zmm7, zmm1
       vfmadd231ps zmm6, zmm2, zmm1
 
G_M000_IG34:                ;; offset=0x05C8
       add      r11, 64
       add      r10, 64
       inc      r14d
       cmp      r14d, 3
       mov      ebx, dword ptr [rbp+0x2C8]
       mov      r8d, dword ptr [rbp-0x2C]
       jge      G_M000_IG44
 
G_M000_IG35:                ;; offset=0x05E7
       add      r8d, r14d
       imul     r8d, r12d
       add      r8d, dword ptr [rbp-0x34]
       mov      edi, r8d
       shl      edi, 4
       mov      r9d, ebx
       sar      r9d, 31
       and      r9d, 15
       add      r9d, ebx
       and      r9d, -16
       mov      dword ptr [rbp+0x2C8], ebx
       sub      ebx, r9d
       add      edi, ebx
       movsxd   rdi, edi
       vbroadcastss zmm1, dword ptr [rcx+4*rdi]
       vmovups  zmm7, zmmword ptr [r11]
       vmovups  zmm2, zmmword ptr [r10]
       mov      r9d, dword ptr [rbp+0x2CC]
       test     r9d, r9d
       jne      G_M000_IG29
 
G_M000_IG36:                ;; offset=0x063B
       vmulps   zmm7, zmm1, zmm7
       vaddps   zmm0, zmm7, zmm0
       vmulps   zmm1, zmm1, zmm2
       vaddps   zmm6, zmm1, zmm6
       jmp      G_M000_IG30
 
G_M000_IG37:                ;; offset=0x0658
       vmulps   zmm7, zmm1, zmm7
       vaddps   zmm0, zmm7, zmm0
       vmulps   zmm1, zmm1, zmm2
       vaddps   zmm6, zmm1, zmm6
       jmp      G_M000_IG32
 
G_M000_IG38:                ;; offset=0x0675
       vmulps   zmm8, zmm1, zmm7
       vaddps   zmm0, zmm8, zmm0
       vmulps   zmm3, zmm1, zmm2
       vaddps   zmm6, zmm3, zmm6
       jmp      G_M000_IG34
 
G_M000_IG39:                ;; offset=0x0692
       mov      r9d, dword ptr [rbp+0x730]
       mov      r8d, r9d
       sar      r8d, 31
       and      r8d, 15
       add      r8d, r9d
       sar      r8d, 4
       mov      r14d, dword ptr [rbp+0x738]
       mov      r10d, r8d
       imul     r10d, r14d
       mov      r11d, dword ptr [rbp+0x7B8]
       mov      ebx, r15d
       imul     ebx, r11d
       add      r10d, ebx
       add      r10d, r13d
       shl      r10d, 4
       movsxd   r10, r10d
       vmovups  zmmword ptr [rdi+4*r10], zmm0
       lea      r10d, [r9+0x10]
       mov      ebx, dword ptr [rbp+0x748]
       cmp      r10d, ebx
       jge      SHORT G_M000_IG41
 
G_M000_IG40:                ;; offset=0x06EA
       inc      r8d
       imul     r8d, r14d
       mov      dword ptr [rbp+0x7B8], r11d
       mov      r10d, r15d
       imul     r10d, r11d
       add      r8d, r10d
       add      r8d, r13d
       shl      r8d, 4
       movsxd   r8, r8d
       mov      qword ptr [rbp+0x750], rdi
       vmovups  zmmword ptr [rdi+4*r8], zmm6
       mov      rdi, qword ptr [rbp+0x750]
       mov      r11d, dword ptr [rbp+0x7B8]
 
G_M000_IG41:                ;; offset=0x0728
       inc      r13d
       mov      dword ptr [rbp+0x740], esi
       mov      dword ptr [rbp+0x748], ebx
       mov      dword ptr [rbp+0x730], r9d
       mov      r9d, r11d
 
G_M000_IG42:                ;; offset=0x0741
       cmp      r13d, r9d
       jge      G_M000_IG50
 
G_M000_IG43:                ;; offset=0x074A
       vxorps   ymm0, ymm0, ymm0
       vxorps   ymm6, ymm6, ymm6
       mov      r11d, edx
       imul     r11d, dword ptr [rbp+0x730]
       lea      r11d, [r11+8*r11]
       movsxd   r11, r11d
       mov      rbx, qword ptr [rbp+0x758]
       lea      r11, [rbx+4*r11]
       mov      r10d, edx
       shl      r10d, 4
       lea      r10d, [r10+8*r10]
       movsxd   r10, r10d
       lea      r10, [r11+4*r10]
       mov      dword ptr [rbp+0x7B8], r9d
       imul     r9d, r15d
       add      r9d, r13d
       cmp      r9d, dword ptr [rbp+0x734]
       setl     r9b
       movzx    r9, r9b
       mov      dword ptr [rbp+0x2CC], r9d
       xor      ebx, ebx
       mov      r8d, r15d
       imul     r8d, eax
       mov      dword ptr [rbp-0x30], r8d
       mov      r8d, r13d
       imul     r8d, eax
       mov      dword ptr [rbp-0x34], r8d
       cmp      ebx, edx
       mov      dword ptr [rbp+0x738], r14d
       jl       SHORT G_M000_IG46
       jmp      G_M000_IG49
 
G_M000_IG44:                ;; offset=0x07CD
       mov      rdi, qword ptr [rbp+0x750]
       inc      ebx
       cmp      ebx, edx
       jge      G_M000_IG39
 
G_M000_IG45:                ;; offset=0x07DE
       mov      dword ptr [rbp+0x740], esi
 
G_M000_IG46:                ;; offset=0x07E4
       xor      r14d, r14d
       mov      r8d, ebx
       sar      r8d, 31
       and      r8d, 15
       add      r8d, ebx
       sar      r8d, 4
       mov      esi, dword ptr [rbp+0x740]
       imul     r8d, esi
       add      r8d, dword ptr [rbp-0x30]
       mov      dword ptr [rbp-0x2C], r8d
       mov      qword ptr [rbp+0x750], rdi
       jmp      G_M000_IG35
 
G_M000_IG47:                ;; offset=0x0817
       xor      r15d, r15d
       mov      dword ptr [rbp+0x7B0], r8d
       test     r8d, r8d
       mov      dword ptr [rbp+0x730], r11d
       mov      r8d, dword ptr [rbp+0x7B0]
       jle      SHORT G_M000_IG51
 
G_M000_IG48:                ;; offset=0x0834
       xor      r13d, r13d
       mov      dword ptr [rbp+0x7B8], r9d
       mov      dword ptr [rbp+0x7B0], r8d
       jmp      G_M000_IG26
 
G_M000_IG49:                ;; offset=0x084A
       mov      esi, dword ptr [rbp+0x740]
       jmp      G_M000_IG39
 
G_M000_IG50:                ;; offset=0x0855
       inc      r15d
       mov      r8d, dword ptr [rbp+0x7B0]
       cmp      r15d, r8d
       jl       SHORT G_M000_IG48
 
G_M000_IG51:                ;; offset=0x0864
       mov      r11d, dword ptr [rbp+0x730]
       add      r11d, 32
       cmp      r11d, dword ptr [rbp+0x748]
       jl       SHORT G_M000_IG47
 
G_M000_IG52:                ;; offset=0x0878
       vzeroupper 
       add      rsp, 0x7A8
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
; Total bytes of code 2189

